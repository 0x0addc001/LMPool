# LMPool: Distributed KV Cache Pooling for LLM Inference

[English](./README.md) | [简体中文](./README_zh.md)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Runtime Model](#3-runtime-model)
4. [Components](#4-components)
5. [Implementation](#5-implementation)
6. [Configuration & Running](#6-configuration--running)
7. [Tests](#7-tests)
8. [Benchmarks](#8-benchmarks)
9. [Current State & Future Work](#9-current-state--future-work)

---

## 1. Overview

LMPool abstracts the HBM of multiple GPUs into a logically unified global KV cache pool. Built on [Mini-vLLM](https://github.com/Wenyueh/MinivLLM)'s Paged Attention, it adds KV Cache-aware cross-GPU routing and NVLink KV transfer.

### 1.1 Problem

| Limitation | Symptom | Consequence |
| --- | --- | --- |
| No cross-GPU prefix reuse | Shared prefixes are stored repeatedly on different GPUs | Memory waste, lower throughput |
| No elasticity under pressure | Local HBM exhaustion leads to OOM or CPU fallback | Latency spikes or request aborts |
| Hot/cold imbalance | Cold blocks occupy HBM while hot blocks are displaced | Sustained latency degradation |

### 1.2 Solution

1. `GlobalBlockManager` maintains a cross-GPU global page table.
2. Block-level hash chains encode prefixes for reuse decisions.
3. `GlobalScheduler` routes requests and plans rebalance.
4. `kv_transfer` executes NCCL-based KV transfer.
5. `LLMEngine` runs as launcher / supervisor, with a dedicated control-plane process and per-rank data-plane workers.

---

## 2. Architecture

The implementation separates orchestration from execution:

- `LLMEngine`: launcher / supervisor
- `control_plane_process`: independent global control process
- `data_plane_process`: per-rank worker process

Each worker owns a local `Scheduler`, `BlockManager`, and `ModelRunner`. The control plane owns the authoritative `GlobalScheduler` and `GlobalBlockManager` state for routing and rebalance planning.

![fig_architecture.png](/assets/fig_architecture.png)

---

## 3. Runtime Model

1. `LLMEngine` receives prompts and builds `Sequence` objects.
2. `ControlPlaneClient` computes the complete-block prefix hash.
3. The control plane asks `GlobalScheduler` to choose a target rank.
4. `LLMEngine` forwards the `Sequence` to the chosen worker.
5. The worker runs prefill / decode locally through `Scheduler` and `ModelRunner`.
6. If memory pressure appears, the worker requests rebalance from the control plane.
7. The control plane dispatches a transfer plan and workers execute NCCL transfer.

---

## 4. Components

### 4.1 Control Plane

**Files**

- `src/lmpool/engine/control_plane.py`
- `src/lmpool/engine/global_scheduler.py`
- `src/lmpool/engine/global_block_manager.py`

The control plane receives route and rebalance requests, updates the global page table, tracks worker heartbeats, and returns decisions to the launcher and workers.

### 4.2 Data Plane

**Files**

- `src/lmpool/engine/data_plane.py`
- `src/lmpool/engine/scheduler.py`
- `src/lmpool/engine/block_manager.py`
- `src/lmpool/engine/model_runner.py`
- `src/lmpool/engine/kv_transfer.py`

Each data-plane process binds one GPU. It schedules prefill / decode locally, allocates KV blocks, runs model forward passes, and performs KV transfer when instructed by the control plane.

### 4.3 Sequence

**File**: `src/lmpool/engine/sequence.py`

`Sequence` carries token ids, block table state, completion tokens, and global-pool metadata such as remote-prefix state.

### 4.4 Global Scheduler

**File**: `src/lmpool/engine/global_scheduler.py`

`GlobalScheduler` is the cross-GPU decision layer. In the current architecture it runs behind the control-plane process. It exposes two main entry points:

- `route_sequence_meta()` for request routing
- `plan_rebalance()` for transfer planning

Routing policy:

1. compute the hash of complete blocks only
2. query the global page table for prefix hits
3. prefer the GPU with the best prefix-hit score after queue-pressure penalty
4. otherwise fall back to the least-congested GPU with more free blocks
5. reserve blocks optimistically after routing

Topology weights in the current routing policy are:

| Relationship | Weight |
| --- | --- |
| Same GPU | 2.0 |
| NVLink partner | 1.0 |

### 4.5 Global Block Manager

**File**: `src/lmpool/engine/global_block_manager.py`

`GlobalBlockManager` stores:

- `global_page_table`: hash to physical block locations
- `free_blocks_per_gpu`: per-GPU free capacity
- `block_access_time`: per-block timestamps for LRU selection
- `block_hash`: per-GPU block hash snapshot

The authoritative state lives in the control plane process. Workers report local snapshots back through block-state messages.

### 4.6 Local Scheduler

**File**: `src/lmpool/engine/scheduler.py`

The local scheduler manages `waiting` and `running` queues and coordinates with `BlockManager`.

- Prefill: schedule waiting sequences, allocate blocks, run model forward
- Decode: append tokens and continue running sequences
- Memory pressure: request rebalance from the control plane

### 4.7 Local Block Manager

**File**: `src/lmpool/engine/block_manager.py`

Each worker owns one `BlockManager` for its local KV cache blocks.

Main responsibilities:

- compute chained block hashes
- allocate blocks and reclaim cold cached blocks with local LRU
- append decode tokens
- maintain local prefix cache state

Complete hashed blocks remain cached after their active reference count reaches
zero. Partial blocks are released immediately; complete cached blocks remain
globally discoverable and evictable until capacity pressure reclaims them.

### 4.8 Model Runner

**File**: `src/lmpool/engine/model_runner.py`

`ModelRunner` holds model weights, CUDA graph captures, KV cache tensors, and the sampler. It is the execution point for forward inference and KV migration hooks.

### 4.9 KV Transfer

**File**: `src/lmpool/engine/kv_transfer.py`

Implements block migration with NCCL `send` / `recv`. Transfer is block-granular and layer-wise.

### 4.10 Sequence

**File**: `src/lmpool/engine/sequence.py`

`Sequence` carries:

- `is_remote_prefix`
- `remote_gpu_id`
- `pending_swap_in` legacy field name for pending transfer-in blocks

These fields survive cross-process transfer through `multiprocessing.Queue`.

---

## 5. [Implementation](./src/lmpool/)

---

## 6. Configuration & Running

### 6.1 Key Configuration Items

| Item | Type | Description |
| --- | --- | --- |
| `world_size` | `int` | Number of worker GPUs participating in the pool |
| `enable_global_pool` | `bool` | Enable global KV cache pooling |
| `use_control_plane_process` | `bool` | Start an independent control process |
| `gpu_memory_utilization` | `float` | Fraction of GPU memory usable |
| `heartbeat_interval` | `float` | Heartbeat period between control and data planes |
| `heartbeat_timeout` | `float` | Liveness timeout for control / worker detection |
| `nvlink_topo.pairs` | `List[Tuple[int, int]]` | Optional NVLink direct-connect GPU pairs; if omitted, the code best-effort parses `nvidia-smi topo -m` |
| `route_prefix_hit_weight` | `float` | Positive weight for reusable prefix blocks in global routing |
| `route_queue_pressure_weight` | `float` | Penalty weight for worker waiting/running queue pressure |
| `route_free_block_weight` | `float` | Small tie-breaker bonus for free KV blocks |
| `route_cache_queue_slack` | `float` | Maximum queue-pressure slack for using a cached prefix route |

### 6.2 Running

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Dual-GPU example
CUDA_VISIBLE_DEVICES=0,2 uv run python main.py

# Single-GPU example
CUDA_VISIBLE_DEVICES=0 uv run python main.py
```

---

## 7. [Tests](./tests/)

---

## 8. [Benchmarks](./benchmarks/)

The `benchmarks/` directory includes a shared-prefix workload benchmark with the following scenarios:

- `single-gpu`
- `multi-gpu`
- `multi-gpu-kv-routing`
- `multi-gpu-kv-transfer`
- `multi-gpu-lmpool`

Reported metrics:

- throughput in generated tokens/s
- goodput in generated tokens/s under `--goodput-e2e-sla-ms`
- mean / p95 TTFT
- mean / p95 TTPT
- mean / p95 end-to-end latency
- GPU utilization mean / p95
- GPU memory utilization mean / p95
- prefix hit rate
- route hit / prefix-owner hit
- transfer / copy count and rebalance success / failure counts

The current `multi-gpu` baseline uses online round-robin dispatch. Control-plane scenarios should use a bounded `--submit-window` such as `4` or `8` when measuring prefix reuse, because routing can only hit prefixes that previous requests have already prefetched and reported to the global page table.
The benchmark records TTFT from explicit first-token events emitted by data-plane workers. Local prefix hit is measured only on each request's initial prefill, excluding retry hits after preemption. Cached-token ratio, prefill attempts, preemptions, and redundant prefill tokens are reported separately. Route hit and prefix-owner hit are reported separately for control-plane scenarios.
The `locality` workload uses 16 distinct long shared-prefix groups by default,
with a deterministic shuffled request order. Override this with
`--locality-prefix-groups`; multiple groups prevent round-robin from matching
routing simply by replicating one hot prefix onto every GPU.
By default it ignores EOS so every request performs the configured decode work.
Use `--seed` for reproducibility and `--repetitions 3` or more to report
mean/standard-deviation results for paper experiments.

See `benchmarks/README.md` for usage.

---

## 9. Current State & Future Work

| Feature | State | Notes |
| --- | --- | --- |
| Multi-GPU async inference | Done | Multiple ranks independently schedule, execute, and sample |
| Control-plane routing | Done | `route_sequence_meta` is implemented |
| NVLink-aware eviction decision | Done | `select_eviction_candidates` is implemented |
| Benchmarks | Done | Shared-prefix benchmark with baseline comparisons |
| Tests | Done | Module-level unit tests and NCCL integration test |

Future work:

1. Expand benchmarks to longer traces and broader workloads.
2. Countinue to update README and comments
