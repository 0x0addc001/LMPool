# LMPool: Distributed KV Cache Pooling for LLM Inference

**Built on Mini-vLLM** | Prototype Stage

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Components](#3-components)
   - 3.1 [Global Scheduler](#31-global-scheduler)
   - 3.2 [Global Block Manager](#32-global-block-manager)
   - 3.3 [Local Scheduler](#33-local-scheduler)
   - 3.4 [Local Block Manager](#34-local-block-manager)
   - 3.5 [Model Runner](#35-model-runner)
   - 3.6 [KV Transfer](#36-kv-transfer)
   - 3.7 [Sequence](#37-sequence)
4. [Configuration & Running](#4-configuration--running)
5. [Current Status & Future Work](#5-current-status--future-work)

---

## 1. Overview

LMPool abstracts the HBM of multiple GPUs into a logically unified global KV cache pool. Built on Mini-vLLM's Paged Attention, it adds cross-GPU block-level prefix-aware routing and hot/cold-aware eviction.

### 1.1 Problem

In vLLM's original Paged Attention, each GPU manages its own memory independently, with three limitations:

| Limitation                | Symptom                                                      | Consequence                                |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| No cross-GPU prefix reuse | Multiple requests sharing the same prefix each allocate separate blocks | Memory waste, reduced effective throughput |
| No elasticity on OOM      | Local HBM exhausted → OOM or CPU swap                        | Latency spike or request abort             |
| Hot/cold imbalance        | Local HBM fills with cold blocks, hot blocks evicted to CPU  | Sustained latency degradation              |

### 1.2 Solution

Abstract multi-GPU HBM into a unified distributed memory pool:

1. **Logical unification**: `GlobalBlockManager` maintains a cross-GPU global page table recording the physical location of every KV block
2. **Prefix deduplication**: Block-level hash chains encode prefixes; cross-GPU lookup ensures identical prefixes are stored only once
3. **Hot/cold awareness**: LRU eviction + topology-prioritized swap (NVLink > PIX > NODE)
4. **Control/data plane separation**: `GlobalScheduler` makes decisions; `kv_transfer` executes NCCL transfers

---

## 2. Architecture

Each GPU process runs an independent `LLMEngine` instance with its own `Scheduler`, `BlockManager`, and `ModelRunner`. `GlobalBlockManager` (authoritative copy on rank 0) and `GlobalScheduler` are layered on top as cross-GPU coordination.

```
┌──────────────────────────────────────────────────────┐
│                    Control Plane                     │
│  ┌──────────────────┐  ┌─────────────────────────┐   │
│  │ GlobalScheduler  │  │  GlobalBlockManager     │   │
│  │ - route_sequence │  │  - global_page_table    │   │
│  │ - rebalance      │  │  - free_blocks_per_gpu  │   │
│  └────────┬─────────┘  └────────────┬────────────┘   │
└───────────┼─────────────────────────┼────────────────┘
            │                         │
┌───────────┼─────────────────────────┼────────────────┐
│           ▼        Data Plane       ▼                │
│  ┌──────────────┐  NVLink/NCCL  ┌──────────────┐     │
│  │    GPU 0     │◄─────────────►│    GPU 1     │     │
│  │ - Scheduler  │  swap_out/in  │ - Scheduler  │     │
│  │ - BlockMgr   │               │ - BlockMgr   │     │
│  │ - ModelRunner│               │ - ModelRunner│     │
│  └──────────────┘               └──────────────┘     │
└──────────────────────────────────────────────────────┘
```

---

## 3. Components

### 3.1 Global Scheduler

**File**: `src/lmpool/engine/global_scheduler.py` → `GlobalScheduler`

`GlobalScheduler` is the cross-GPU decision layer. At initialization it holds references to `GlobalBlockManager` (for page table queries), the local `BlockManager` (for hash computation), and optionally `ModelRunner` (for executing KV transfers). It exposes two primary entry points: request routing (`route_sequence`) and memory rebalancing (`rebalance`).

#### 3.1.1 Routing (`route_sequence`)

##### 3.1.1.1 Routing Decision

Determines which GPU a newly arriving sequence should execute on. The algorithm proceeds through six steps in priority order:

1. **Compute prefix hash**: Calls `_compute_prefix_hash(seq)`, which hashes only the sequence's complete blocks (partial tail blocks excluded). For the *i*-th complete block, it calls `BlockManager.compute_hash(block_tokens, prev_hash)`, chaining the hash of block *i-1* into the input of the current block. If the sequence has no complete blocks, returns `None` and routing jumps directly to Step 6.

2. **Query global page table**: Calls `gbm.lookup_prefix(prefix_hash)`, returning `List[BlockLocation]` where each entry carries `(gpu_id, block_id, hash, last_access_time)`. If the result is empty, jumps to Step 6.

3. **Aggregate hit counts by GPU**: Iterates over the `BlockLocation` list, accumulating `gpu_hit_count[gpu_id] += 1`.

4. **Weighted scoring**: For each candidate GPU: `score = hit_count × topo_weight`. Topology weights are computed by `_get_topo_weight(my_rank, target_gpu)`:

| Relationship        | Weight |
| ------------------- | ------ |
| Same GPU            | 3.0    |
| NVLink partner      | 2.0    |
| Same socket (PIX)   | 1.5    |
| Cross-socket (NODE) | 1.0    |

5. **Select highest-scoring GPU with sufficient free blocks**: Iterates over scored candidates. Only GPUs satisfying `gbm.get_free_blocks_count(gpu_id) >= seq.num_blocks` are qualified. If the highest-scoring candidate has insufficient free blocks, it is still returned—the expectation is that a subsequent `rebalance()` will free up space.

6. **Fallback: select GPU with most free blocks**: When there is no prefix hash or no hits at all, `_select_most_free_gpu` scans all ranks and returns the GPU with the largest free block count, preferring the local rank on ties.

##### 3.1.1.2 Hash Chaining & Block-Level Prefix Matching

The hash of the *i*-th block encodes the complete token content from block 0 through block *i*:

```
hash_0 = xxhash64(tokens[0 : block_size])
hash_1 = xxhash64(hash_0.to_bytes(8) + tokens[block_size : 2*block_size])
...
hash_k = xxhash64(hash_{k-1}.to_bytes(8) + tokens[k*block_size : (k+1)*block_size])
```

Two sequences sharing a prefix of length *k × block_size* will necessarily produce the same `hash_k`. Querying `global_page_table[hash_k]` finds all GPUs holding that prefix—no token-level comparison needed.

Only complete blocks are hashed. Partial blocks are always assigned hash `-1`, preventing spurious hits as sequences grow.

#### 3.1.2 Swapping (`rebalance`)

##### 3.1.2.1 Swapping Decision

`rebalance(gpu_id, needed_blocks)` is called when a GPU needs `needed_blocks` free blocks but has insufficient local capacity. It orchestrates cross-GPU swap to free up space.

1. **Get eviction candidates**: Calls `gbm.select_eviction_candidates(gpu_id, needed_blocks)`, returning `[(local_block_id, target_gpu_id), ...]` (three-tier eviction strategy; see §3.2.3).

2. **Quantity check**: If `len(candidates) < needed_blocks`, returns `False` immediately.

3. **Group by target GPU**: Groups the candidate list by `target_gpu` into `groups: dict[int, List[int]]`.

4. **Execute NCCL transfer**: For each group, the rank owning `gpu_id` calls `_execute_swap_out(blocks, gpu_id, target_gpu)` (delegated to `ModelRunner.execute_swap_out`), and the rank owning `target_gpu` calls `_execute_swap_in_accept(blocks, gpu_id, target_gpu)` (delegated to `ModelRunner.execute_swap_in`). Both sides synchronize via `dist.barrier()` before and after the transfer.

5. **Update global page table**: For each evicted block, calls `gbm.free_global(gpu_id, [local_block])`, then updates free counts and page table entries via `gbm.record_block_transfer()`.

6. **Preemption fallback**: When `rebalance()` still cannot satisfy the space requirement (e.g., all remote GPUs are also full), `preempt_for_rebalance(running_sequences, gpu_id, needed_blocks)` serves as the last resort: it iterates through `running_sequences` in order, marks the shortest sequences as `WAITING`, releases all their blocks, until the cumulative freed count meets `needed_blocks`. Preempted sequences have their `block_table` cleared and `num_cached_tokens` reset to 0, awaiting rescheduling.

---

### 3.2 Global Block Manager

**File**: `src/lmpool/engine/global_block_manager.py` → `GlobalBlockManager`

`GlobalBlockManager` is the authoritative registry of the distributed KV cache pool. Rank 0 (configurable via `master_rank`) holds the master copy of all state; other ranks hold local caches and refresh periodically.

#### 3.2.1 Properties

| Property              | Type                             | Description                                        |
| --------------------- | -------------------------------- | -------------------------------------------------- |
| `global_page_table`   | `Dict[int, List[BlockLocation]]` | `prefix_hash →` physical locations of all replicas |
| `free_blocks_per_gpu` | `List[int]`                      | Free block count per GPU                           |
| `block_access_time`   | `List[Dict[int, float]]`         | Per-GPU, per-block last access timestamps (LRU)    |
| `block_hash`          | `List[Dict[int, int]]`           | Per-GPU `block_id → hash` mapping                  |
| `master_rank`         | `int`                            | Rank of the authoritative master (default 0)       |
| `nvlink_pairs`        | `Set[Tuple[int, int]]`           | NVLink direct-connect GPU pairs                    |
| `socket_groups`       | `List[List[int]]`                | Per-CPU-socket GPU groupings                       |

`BlockLocation` dataclass carries four attributes—`gpu_id`, `block_id`, `hash`, `last_access_time`—and serves as the canonical representation of a KV block's physical location.

#### 3.2.2 Distributed Memory Allocation

##### 3.2.2.1 Prefix Deduplication

When the local `BlockManager` allocates blocks for a new sequence, the hash of the last complete block is committed to `GlobalBlockManager` via `_commit_alloc(gpu_id, block_ids, hashes)`.

`_commit_alloc(gpu_id, block_ids, hashes)` is the core write path. For each block, in order:

1. `free_blocks_per_gpu[gpu_id] -= 1`
2. `block_access_time[gpu_id][bid] = now` (for LRU)
3. `block_hash[gpu_id][bid] = h`
4. Appends `BlockLocation(gpu_id, bid, h, now)` to `global_page_table[h]`

If the hash already exists in the page table (i.e., replicas of the same prefix exist on other GPUs), a new `BlockLocation` entry is added—both replicas are registered, and routing can choose the topologically nearest one.

`BlockManager.allocate(seq)` drives this process: it iterates over all blocks of the new sequence, computes chained hashes, checks `hash_to_block_id` for local hits—reusing on hit (`ref_count++` and incrementing `seq.num_cached_tokens`), allocating a new block on miss. Finally, it registers the last complete block's hash globally via `gbm._commit_alloc`.

##### 3.2.2.2 Hot/Cold-Aware Allocation

When local free blocks are insufficient, `BlockManager.allocate_with_swap(seq)` is called instead of direct `allocate`:

1. **Check local free**: if `can_allocate(seq)` is true, calls `allocate()` directly and returns
2. **Compute shortage**: `shortage = seq.num_blocks - len(free_block_ids)`
3. **Get eviction candidates**: calls `gbm.select_eviction_candidates(rank, shortage)`, returning topology-aware `(local_block, target_gpu)` pairs (see §3.2.3)
4. **Release cold blocks**: for each candidate block with `ref_count == 0`, calls `_deallocate_block` to return it to `free_block_ids`, then calls `gbm.record_block_transfer()` to update the global page table
5. **Normal allocation**: after sufficient space is freed, calls `allocate(seq)`

The decode-stage counterpart is `append_with_swap(seq)`, which follows the same logic but only needs to free 1 block at a time.

#### 3.2.3 Hot/Cold-Aware Eviction (`select_eviction_candidates`)

`select_eviction_candidates(gpu_id, num_blocks) → List[Tuple[int, int]]` uses a three-tier progressive strategy to find a swap target for each cold block:

**Tier 1 — Select locally coldest blocks**

Sorts `block_access_time[gpu_id]` by timestamp ascending, takes the top `num_blocks` as `cold_blocks`.

**Tier 2 — Find a target GPU with free space for each cold block**

Iterates over candidate targets in the topology-priority order returned by `_get_target_gpu_order(gpu_id)`:

- **Priority 1**: NVLink direct-connect partner
- **Priority 2**: Same-socket GPUs (sorted by free block count descending)
- **Priority 3**: Cross-socket GPUs (sorted by free block count descending)

Stops at the first target where `free_blocks_per_gpu[target] > 0`, and temporarily decrements that target's free count by 1 to prevent subsequent blocks from occupying the same slot.

**Tier 3 — Recursive eviction / overwrite**

If all target GPUs have zero free blocks:

- Calls `_select_remote_victim(target)` to pick the LRU-coldest block on the topologically nearest target
- Removes that victim's entry from `block_access_time`, `block_hash`, and `global_page_table`—effectively recursive eviction
- The target thus gains one free slot, ready to receive the local cold block
- If the remote side is also completely empty (no blocks to choose from), falls back to direct overwrite—the target block count stays unchanged, old data is overwritten when the transfer arrives

`_get_target_gpu_order(gpu_id)` constructs the order as:

```python
ordered = []
# Tier 1: NVLink direct-connect partner (highest bandwidth)
partner = nvlink_partner.get(gpu_id)
if partner: ordered.append(partner)

# Tier 2: Same-socket GPUs, sorted by free block count descending
same_socket.sort(key=lambda g: free_blocks_per_gpu[g], reverse=True)
ordered.extend(same_socket)

# Tier 3: Cross-socket GPUs, sorted by free block count descending
other_socket.sort(key=lambda g: free_blocks_per_gpu[g], reverse=True)
ordered.extend(other_socket)
```

#### 3.2.4 Prefix Lookup (`lookup_prefix`)

`lookup_prefix(prefix_hash) → List[BlockLocation]`

1. If `prefix_hash` is not in `global_page_table`, returns `[]` immediately
2. Retrieves all matching `BlockLocation` entries, scores them by NVLink affinity: NVLink partner blocks score 2.0, same-socket blocks score 1.5, others score 1.0; higher scores sort first
3. Returns the sorted list for the router to directly pick the optimal option

Note: `lookup_prefix`'s sorting weights measure the distance of the target GPU from the **caller's** perspective (the current rank), which has the same direction as the eviction target ordering but a different reference point.

#### 3.2.5 Global Page Table Synchronization

`GlobalBlockManager` uses a **master-push synchronization model**:

**`update_gpu_state(gpu_id, free_blocks, block_hashes)`**

Master-only state ingestion boundary. Workers call this method via `Scheduler._sync_local_state_to_global()` after every allocation, append, preemption, or sequence completion. It atomically replaces the master's view of `gpu_id`'s state: first clears all old entries for that GPU from the global page table, then re-inserts all entries from the new `block_hashes` snapshot.

**`reserve_blocks(gpu_id, num_blocks)`**

After a request is routed to a remote GPU but before the remote worker has called `update_gpu_state`, optimistically decrements `free_blocks_per_gpu[gpu_id]` by `num_blocks`, preventing over-routing to the same GPU during the brief state-latency window.

**`broadcast_page_table()`**

First calls `gather_local_state()`—collects each rank's current `free_blocks_per_gpu` value via `dist.all_gather_into_tensor`; then broadcasts the complete `(global_page_table, free_blocks_per_gpu, block_access_time, block_hash, master_rank)` tuple from `master_rank` to all ranks via `dist.broadcast_object_list`. Non-master ranks overwrite their local cache.

**`maybe_sync()`**

An internal counter increments; every `sync_interval` (default 10) scheduler cycles, calls `broadcast_page_table()` once. Currently commented out in the `Scheduler`; re-enabling it is the first step to activating true cross-GPU prefix reuse.

#### 3.2.6 Master Failover

`GlobalBlockManager` includes a complete failure detection skeleton, not yet enabled in production paths:

1. `check_master_health()`: The master refreshes its own `master_heartbeat` timestamp; non-master ranks call `_broadcast_master_heartbeat()` to receive the heartbeat. If `now - heartbeat > heartbeat_timeout` (default 100 s), triggers an election
2. `_elect_new_master()`: Simplified rotation strategy: `new_master = (old_master + 1) % world_size`
3. `set_master_rank(new_master)`: Manually designate a new management node for post-disaster reconfiguration

#### 3.2.7 Block Lifecycle

```
                         ┌─────────────────┐
                         │   FREE (idle)   │
                         └────────┬────────┘
                                  │ _commit_alloc / allocate
                                  ▼
                         ┌─────────────────┐
                         │   ALLOCATED     │◄──── ref_count > 0 (shared)
                         └────────┬────────┘
                                  │ ref_count → 0
                                  ▼
                         ┌─────────────────┐
                         │   DEALLOCATED   │──► back to FREE
                         └─────────────────┘
                                  │
                    swap_out      │      swap_in
                    ┌─────────────┼─────────────┐
                    ▼             │              ▼
            ┌──────────┐          │       ┌──────────┐
            │ REMOTE   │          │       │  LOCAL   │
            │ GPU      │          │       │  RESTORE │
            └──────────┘          │       └──────────┘
                                  │
                        overwrite │
                                  │
                                  ▼
                            ┌──────────┐
                            │ OVERWRITE│ (discard old data)
                            └──────────┘
```

---

### 3.3 Local Scheduler

**File**: `src/lmpool/engine/scheduler.py` → `Scheduler`

The local scheduler manages `waiting` and `running` double-ended queues, making memory allocation decisions in coordination with `BlockManager`. Injecting `global_scheduler` activates two extension hooks.

#### 3.3.1 Prefill Phase

**Remote routing**: For each sequence at the front of `waiting`, if `remote_gpu_id` is not set, calls `global_scheduler.route_sequence(seq)`. If the returned target is a different GPU, pops the sequence from `waiting`, sets its status to `RUNNING`, adds it to `scheduled_sequences` without local block allocation, and calls `gbm.reserve_blocks(target_gpu, seq.num_blocks)` to optimistically decrement the target GPU's free count. Actual block allocation happens on the target rank.

**Swap-assisted allocation**: If a locally-routed sequence cannot be allocated due to insufficient free blocks, calls `block_manager.allocate_with_swap(seq)`, which internally triggers `gbm.select_eviction_candidates` and evicts cold blocks to free up space. If this also fails, the prefill loop breaks and the sequence remains in `waiting`.

#### 3.3.2 Decode Phase

**Rebalance on append failure**: When `block_manager.can_append(seq)` returns `False`, calls `global_scheduler.rebalance(self.rank, 1)`. If rebalance succeeds, pushes the sequence back to the front of `running` for retry next iteration; if it fails, executes the original preemption logic (shortest running sequence is preempted).

#### 3.3.3 State Synchronization

After every allocation, append, preemption, or sequence completion, `_sync_local_state_to_global()` pushes a `(free_count, block_hashes)` snapshot of the local `BlockManager` to the master `GlobalBlockManager` via `gbm.update_gpu_state(rank, free_count, block_hashes)`.

---

### 3.4 Local Block Manager

**File**: `src/lmpool/engine/block_manager.py` → `BlockManager`

Each GPU process owns one `BlockManager` instance managing that GPU's physical KV cache blocks: `free_block_ids` (deque), `used_block_ids` (set), a local `hash_to_block_id` prefix cache, and a reference to `GlobalBlockManager`.

**`compute_hash(token_ids, prefix_hash_value)`**

Computes `xxhash64` over the binary representation of the token IDs as a numpy `int32` array. If `prefix_hash_value != -1`, feeds its 8-byte little-endian encoding into the hasher first, achieving hash chaining.

**`allocate(seq)`**

Iterates over all blocks of the sequence. For each complete block, computes the chained hash and checks `hash_to_block_id` for a local hit: on hit, `ref_count++` and accumulates `seq.num_cached_tokens`; on miss, takes a new block from `free_block_ids`, calls `block.update(h, token_ids)` and writes to `hash_to_block_id`. Partial tail blocks always allocate a new block with hash `-1`. After all blocks are processed, if `gbm` is not `None`, calls `gbm._commit_alloc` to register the last complete block.

**`deallocate(seq)`**

For each block in `seq.block_table`: `ref_count -= 1`; blocks reaching 0 are returned to `free_block_ids` and removed from `hash_to_block_id`. Finally clears `seq.block_table` and `seq.num_cached_tokens`.

**`append(seq)`**

Called after appending a new token:

1. If the new token exactly fills a block (`num_tokens % block_size == 0`): computes and stores that block's hash, writes to `hash_to_block_id`, and notifies `gbm._commit_alloc`
2. If the new token is the first of a new block (`num_tokens % block_size == 1`): allocates a new block from `free_block_ids`
3. Otherwise: token is written to the existing partial block, no operation needed

**`can_allocate(seq)`**: `len(free_block_ids) >= seq.num_blocks`

**`can_append(seq)`**: If the next token would open a new block, checks `free_block_ids` is non-empty; otherwise returns `True` directly.

---

### 3.5 Model Runner

**File**: `src/lmpool/engine/model_runner.py` → `ModelRunner`

`ModelRunner` holds model weights, CUDA graph captures, the KV cache tensor, and the sampler. It is the sole execution point for KV cache physical memory allocation and cross-GPU data transfer.

#### 3.5.1 KV Cache Allocation (`allocate_kv_cache`)

After the warmup forward pass, computes `available_mem = free_mem × gpu_memory_utilization - (peak_warmup - current)`, then divides by the per-block byte cost (`block_size × 2 × num_layers × num_kv_heads × head_dim × dtype_bytes`) to obtain `num_available_kv_blocks`.

In multi-GPU mode, uses `dist.all_reduce(..., op=MIN)` to let all ranks adopt the most conservative block count, ensuring the block table size is globally consistent.

The KV cache is allocated as a single tensor `(2, num_layers, max_cached_blocks, block_size, num_kv_heads, head_dim)`, then sliced per-layer and assigned to each attention module's `k_cache` / `v_cache` attributes.

#### 3.5.2 Weight Loading & Broadcast

Rank 0 loads weights from disk, then between two `dist.barrier()` calls, rank 0 iterates over `model.parameters()` and calls `dist.broadcast(param.data, src=0)` for each parameter, ensuring all ranks hold identical weights before entering the main loop.

#### 3.5.3 CUDA Graph Capture (`capture_cudagraph`)

Captures one CUDA graph per decode batch size `[1, 2, 4, 8, 16, 32, ...]` (up to `max_num_seqs`), stored in `self.graphs[batch_size]`. During inference, `run_model` finds the smallest captured graph that fits the current batch, updates input tensors in-place, and replays it, eliminating CPU launch overhead.

#### 3.5.4 Remote Block Fetch (`_swap_in_remote_blocks(seq)`)

Before the model forward in `run()`, for each sequence with non-empty `pending_swap_in`: scans model modules to locate the `k_cache` tensor, calls `kv_transfer.swap_in(remote_gpu, remote_blocks, local_gpu, kv_cache, ...)`, writes the returned `local_blocks` into `seq.block_table` at the corresponding prefix positions, and finally clears `pending_swap_in`, `is_remote_prefix`, and `remote_gpu_id`.

#### 3.5.5 Swap Execution (`execute_swap_out` / `execute_swap_in`)

Both methods first call `_get_kv_cache()` to locate the KV cache tensor, then delegate to `kv_transfer.swap_out` and `kv_transfer.swap_in` respectively. Called by `GlobalScheduler._execute_swap_out` / `_execute_swap_in_accept`.

---

### 3.6 KV Transfer

**File**: `src/lmpool/engine/kv_transfer.py`

Implements two cross-GPU block migration primitives based on NCCL `send`/`recv`. Tag encoding `block_id × 10000 + layer_idx × 2 + is_k` ensures K/V tensors never collide even during concurrent multi-block transfers.

#### 3.6.1 Eviction (`swap_out`)

Moves cold KV blocks from the source GPU to a target GPU:

1. **Block index negotiation**: Source sends the list of blocks to evict via `_send_block_list`; if `target_free_blocks` is specified, sends that as well; target replies with the allocated target block IDs
2. **Per-layer, per-block transfer**: For each layer `0..num_layers-1`, source reads `layer_kv[0, src_block]` (K) and `layer_kv[1, src_block]` (V), calls `dist.send`; target allocates zero buffers, calls `dist.recv`, then `copy_` into `layer_kv[0/1, dst_block]`
3. **Global barrier**: Calls `dist.all_reduce` on a scalar zero tensor to ensure both sides complete before returning

#### 3.6.2 Fetch (`swap_in`)

Pulls KV blocks from a remote GPU to local. The protocol is the mirror of `swap_out`: local sends the desired remote block list → remote replies with mapping → K/V data flows in reverse direction per layer.

In single-GPU scenarios, both sides are the same rank, and NCCL calls degenerate to no-ops (`target_blocks = blocks_to_evict`).

---

### 3.7 Sequence

**File**: `src/lmpool/engine/sequence.py` → `Sequence`

Three fields added for global pooling:

| Field              | Type        | Description                                                  |
| ------------------ | ----------- | ------------------------------------------------------------ |
| `is_remote_prefix` | `bool`      | Whether the sequence uses a prefix KV cache from a remote GPU |
| `remote_gpu_id`    | `int`       | The GPU rank holding the remote prefix; -1 means all blocks are local |
| `pending_swap_in`  | `List[int]` | List of remote physical block IDs waiting to be pulled locally |

All three fields are included in `__getstate__` / `__setstate__`, ensuring they survive cross-process transfer via `multiprocessing.Queue`.

---

## 4. Configuration & Running

### 4.1 Key Configuration Items

| Item                              | Type                    | Description                                                  |
| --------------------------------- | ----------------------- | ------------------------------------------------------------ |
| `world_size`                      | `int`                   | Number of GPUs participating in the pool                     |
| `enable_global_pool`              | `bool`                  | Enable global KV cache pooling                               |
| `gpu_memory_utilization`          | `float`                 | Fraction of GPU memory usable (lower values trigger swap earlier) |
| `swap_threshold`                  | `float`                 | GPU memory usage ratio threshold for triggering swap         |
| `global_page_table_sync_interval` | `int`                   | Page table broadcast interval in scheduler cycles            |
| `nvlink_topo.pairs`               | `List[Tuple[int, int]]` | NVLink direct-connect GPU pairs                              |
| `nvlink_topo.sockets`             | `List[List[int]]`       | Per-CPU-socket GPU groupings                                 |

### 4.2 Running

```bash
# Dual-GPU NVLink test
CUDA_VISIBLE_DEVICES=0,2 uv run python main.py

# Single-GPU baseline
CUDA_VISIBLE_DEVICES=0 uv run python main.py
```

---

## 5. Current Status & Future Work

### 5.1 Current Status

| Feature                          | Status            | Notes                                                        |
| -------------------------------- | ----------------- | ------------------------------------------------------------ |
| Peer-to-peer multi-GPU inference | ✅ Complete        | Both ranks independently schedule, execute, and sample       |
| Cross-GPU sequence routing       | ✅ Complete        | `route_sequence` operational                                 |
| Global page table sync           | ❌ Disabled        | `maybe_sync` commented out; ranks have independent page tables |
| `swap_out`                       | ✅ Triggered       | Logs confirm execution                                       |
| `swap_in`                        | 🔄 In progress     | End-to-end pending verification                              |
| Prefix reuse (dedup)             | ❌ Not effective   | Page tables out of sync + remote allocation does not reuse existing blocks |
| Topology-aware eviction          | ✅ Code ready      | `select_eviction_candidates` implements three-tier strategy  |
| RadixTree prefix tree            | ❌ Not implemented | Current hash-chain approach sufficient; future optimization  |

### 5.2 Future Work

1. **Re-enable `maybe_sync()`**: The key missing link—this will allow `lookup_prefix` to find cross-GPU prefix hits
2. **Verify swap end-to-end**: Construct NCCL send/recv scenarios between two ranks, confirm no deadlocks and data correctness
3. **Implement prefix block reuse**: Modify `BlockManager.allocate` to accept `BlockLocation` hints, directly reference existing physical blocks instead of reallocating—closing the final gap in prefix reuse
4. **Construct high-concurrency long-prefix benchmarks**: Quantify cache hit rate and TTFT improvement under shared-prefix scenarios