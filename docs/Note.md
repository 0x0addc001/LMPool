# LMPool System Notes

This document summarizes the current implementation in a form suitable for a paper draft:

- runtime flow
- component relationships
- control / data-plane interaction sequence
- implementation details

The codebase currently uses three layers:

- `LLMEngine`: launcher / supervisor
- `control_plane.py`: independent global control process
- `data_plane.py`: per-rank worker process

---

## 1. Runtime Flow

```mermaid
flowchart TD
    A[LLMEngine starts] --> B{enable_global_pool?}
    B -- no --> C[Start rank workers only]
    B -- yes --> D[Start Control Plane Process]
    D --> E[Start one DataPlaneProcess per rank]
    E --> F[Prompt enters LLMEngine]
    F --> G[ControlPlaneClient computes prefix hash]
    G --> H[Send route_request to control plane]
    H --> I[GlobalScheduler route]
    I --> J[GlobalBlockManager lookup and free-space check]
    J --> K[Return target rank + route info]
    K --> L[LLMEngine forwards Sequence to target worker]
    L --> M[DataPlaneProcess enqueues Sequence]
    M --> N[Local Scheduler schedules prefill/decode]
    N --> O{need swap or rebalance?}
    O -- yes --> P[Control plane plans rebalance]
    P --> Q[Workers execute swap out and swap in]
    Q --> R[Workers report new block state]
    O -- no --> S[ModelRunner runs inference]
    R --> T[GlobalBlockManager updates global page table]
    S --> U[Postprocess and finish]
    U --> V[LLMEngine collects outputs]
```

---

## 2. Component Diagram

```mermaid
flowchart LR
    subgraph Launcher["LLMEngine"]
        LE[LLMEngine]
    end

    subgraph Control["Control Plane Process"]
        CP[control_plane_process]
        GS[GlobalScheduler]
        GBM[GlobalBlockManager]
    end

    subgraph Worker["DataPlaneProcess per rank"]
        DP[data_plane_process]
        S[Scheduler]
        BM[BlockManager]
        MR[ModelRunner]
    end

    LE --> CP
    CP --> GS
    GS --> GBM
    DP --> S
    S --> BM
    S --> MR
    CP <--> DP
```

---

## 3. Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant L as LLMEngine
    participant C as ControlPlaneClient
    participant P as ControlPlaneProcess
    participant G as GlobalScheduler
    participant W as DataPlaneProcess
    participant S as Scheduler
    participant M as ModelRunner

    U->>L: generate(prompts)
    L->>C: route_sequence(seq)
    C->>P: route_request(seq meta + prefix hash)
    P->>G: route_sequence_meta(...)
    G->>P: target_rank + route_info
    P->>C: route_response
    L->>W: send Sequence to target rank
    W->>S: add_sequence(seq)
    S->>S: schedule prefill / decode
    alt local execution
        S->>M: run(local_seqs)
        M->>S: outputs
        S->>W: postprocess
    else rebalance needed
        S->>C: rebalance request
        C->>P: rebalance_request
        P->>W: rebalance_execute plan
        W->>M: execute_swap_out / execute_swap_in
        W->>P: rebalance_done
        P->>C: rebalance_response
    end
    W->>L: finished sequence tokens
```

---

## 4. System Implementation

### 4.1 Process Layout

The current implementation separates orchestration and execution.

- `LLMEngine` is the launcher and top-level supervisor. It starts the control-plane process and one data-plane process per rank.
- `control_plane_process` is a dedicated global coordinator. It owns routing decisions, global block-table updates, rebalance planning, and heartbeat tracking.
- `data_plane_process` is the per-rank execution loop. It owns local scheduling, KV-cache allocation, model execution, and swap execution.

This layout is intentional: the control plane can be described as an independent process in a systems paper, rather than as logic embedded inside rank 0.

### 4.2 Request Routing

For each prompt, the launcher creates a `Sequence`, computes its prefix hash through `ControlPlaneClient`, and sends a `route_request` to the control plane.

`GlobalScheduler.route_sequence_meta()` then decides the target GPU using the current global page table and local free-space snapshot from `GlobalBlockManager`.

The current routing logic is:

1. compute the hash of complete blocks only
2. lookup the prefix in the global page table
3. prefer the GPU with the best prefix-hit score
4. otherwise fall back to the GPU with the most free blocks
5. reserve blocks optimistically after routing

### 4.3 Local Scheduling

Each `DataPlaneProcess` keeps its own `Scheduler`, `BlockManager`, and `ModelRunner`.

- prefill: schedule waiting sequences, allocate blocks, and run model forward
- decode: append tokens, capture completion, and keep running sequences in order
- memory pressure: if local space is insufficient, request rebalance from the control plane

### 4.4 Global Page Table

`GlobalBlockManager` stores:

- `global_page_table`: hash to physical block locations
- `free_blocks_per_gpu`: per-GPU free capacity
- `block_access_time`: per-block timestamps for LRU selection
- `block_hash`: per-GPU block hash snapshot

The authoritative state lives in the control plane process. Workers report local snapshots through block-state messages, and the control plane updates the global view.

### 4.5 Swap / Rebalance

When a GPU cannot make progress because it lacks free blocks, the control plane builds a rebalance plan and dispatches it to the affected workers.

The current path is:

1. `GlobalScheduler.plan_rebalance()`
2. control plane broadcasts the plan to source and destination ranks
3. workers execute `ModelRunner.execute_swap_out()` / `execute_swap_in()`
4. workers update local `BlockManager` state
5. workers report the new block snapshot back to the control plane

### 4.6 KV Transfer

`kv_transfer.py` implements the physical migration primitive using NCCL `send` / `recv`.

The transfer is block-granular and layer-wise:

- source sends block ids
- destination allocates target block ids
- K and V tensors are transferred for each layer
- both sides synchronize before returning

### 4.7 Logging and Observability

The code currently emits structured `INFO`-level logs for:

- prefill and decode activity per rank
- routing decisions and route reasons
- swap / rebalance execution
- worker heartbeats and control-plane heartbeats
- finish and idle transitions

This is sufficient for development and for reproducing routing / swap traces in experiments.
