![fig_architecture.png](/assets/fig_logo_dark.png)

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
2. `ControlPlaneClient` computes the cumulative hash chain for every complete prefix block.
3. The control plane asks `GlobalScheduler` to choose a target rank.
4. `LLMEngine` forwards the `Sequence` to the chosen worker.
5. The worker runs prefill / decode locally through `Scheduler` and `ModelRunner`.
6. If memory pressure appears, the worker requests rebalance from the control plane.
7. The control plane dispatches a
   `prepare -> execute -> publish -> finalize` transfer plan; workers retain
   the source until every destination has published valid KV.

---

## 4. Components

### 4.1 Control Plane

**Files**

- `src/lmpool/engine/control_plane.py`
- `src/lmpool/engine/global_scheduler.py`
- `src/lmpool/engine/global_block_manager.py`

The control plane receives route and rebalance requests, updates the global page table, tracks worker heartbeats, and returns decisions to the launcher and workers. A control epoch rejects replies from an obsolete control process, while per-worker epochs and monotonic snapshot versions reject stale worker state. A timed-out worker is removed from routing and page-table lookup until it publishes a fresh full snapshot.

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

1. compute a cumulative hash chain over complete blocks only
2. query the global page table and measure each GPU's longest contiguous prefix from block zero
3. prefer the GPU with the best reusable-block score after queue-pressure penalty
4. otherwise fall back to the least-congested GPU with enough effective
   capacity (`free + dependency-safe reclaimable cache`)
5. reserve only the blocks not already covered by the selected GPU's contiguous prefix

Ingress reservations are tracked by sequence until first prefill commits, so
concurrent requests cannot promise the same reclaimable blocks more than once.

Foreground transfer uses reason-aware exponential backoff for structural
failures such as `no_plan`, `no_target_space`, and `stale_source`. Repeated
requests are suppressed until capacity state has had time to change.

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
- dependency-safe reclaimable capacity derived from the ready-block prefix DAG
- per-sequence optimistic block reservations for routed but uncommitted requests
- per-block access frequency and recency for LFU-first, LRU-second selection
- `block_hash`: per-GPU block hash snapshot
- `block_generation`: physical block reuse generation used to reject stale
  transfer plans that refer to a recycled block id
- `block_parent_hash`: parent links used to preserve valid prefix chains during eviction

The authoritative state lives in the control plane process. Workers report
versioned local snapshots back through block-state messages. Control restart
requests a full snapshot from every worker before that rank becomes routable.

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
- allocate blocks and reclaim cold cached blocks with leaf-constrained LFU/LRU
- append decode tokens
- maintain local prefix cache state

Complete hashed blocks remain cached after their active reference count reaches
zero. Partial blocks are released immediately; complete cached blocks remain
globally discoverable and evictable until capacity pressure reclaims them.
Eviction only removes prefix-chain leaves, so a retained descendant never loses
an ancestor required for contiguous reuse. Access frequency orders eligible
leaves first and recency breaks ties.

### 4.8 Model Runner

**File**: `src/lmpool/engine/model_runner.py`

`ModelRunner` holds model weights, CUDA graph captures, KV cache tensors, and the sampler. It is the execution point for forward inference and KV migration hooks.

### 4.9 KV Transfer

**File**: `src/lmpool/engine/kv_transfer.py`

Implements block migration with NCCL `send` / `recv`. Blocks remain the placement
unit, while every layer and all K/V blocks in one plan are packed into one
contiguous P2P payload and sent with one blocking P2P operation.

Transfer uses an idempotent four-stage protocol. `prepare` validates the
source hash and physical-block generation, locks the source against local
reclamation, and reserves target blocks. `execute` copies KV into target blocks
that remain hidden from local/global prefix lookup. `publish` exposes every
valid target while all sources remain locked. Only after all publish ACKs does
`finalize` release source blocks for move-style plans and report the new page
table; `abort` unlocks the source and drops hidden reservations. These locks
cover transfer state transitions only and are not acquired by model forward or
decode. A control-process epoch change aborts unfinished local plans before the
worker publishes its recovery snapshot.

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
| `route_load_bypass_threshold` | `float` | Minimum token-equivalent cost advantage required to bypass a prefix owner |
| `route_prefill_cost_weight` | `float` | Cost per missing prefix token in the completion-cost model |
| `route_reclaim_cost_weight` | `float` | Additional cost for admitting through locally reclaimable KV capacity |
| `foreground_transfer_cost_weight` | `float` | Multiplier on the time-domain transfer cost |
| `foreground_transfer_min_benefit_ratio` | `float` | Minimum predicted saved-prefill-ms / transfer-ms ratio |
| `foreground_transfer_bandwidth_gib_s` | `float` | Measured effective bandwidth used by transfer admission |
| `foreground_transfer_fixed_latency_ms` | `float` | Fixed protocol and coordination cost per plan |
| `foreground_transfer_interference_multiplier` | `float` | Packing, unpacking, and inference interference multiplier |
| `foreground_prefill_token_time_ms` | `float` | Estimated recomputation time per uncached prompt token |
| `foreground_future_reuse_discount` | `float` | Discount from historical leaf-prefix accesses to future reuse |
| `foreground_transfer_ewma_alpha` | `float` | EWMA weight for observed source-side transfer overhead |
| `enable_kv_transfer_prewarm` | `bool` | Initialize configured NVLink P2P communicators before worker readiness |
| `route_cache_queue_slack` | `float` | Maximum completion-cost slack for using a cached prefix route |

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

The `benchmarks/` directory exposes three paper-oriented executable entries:

- `benchmark_kv_routing.py`: locality/routing-only ablation
- `benchmark_kv_transfer.py`: isolated NCCL/NVLink KV data path
- `benchmark_e2e.py`: five-configuration session-handoff comparison

Use the reproducible command matrix in
[`benchmarks/PAPER_RUNBOOK.md`](./benchmarks/PAPER_RUNBOOK.md) for paper runs.

The end-to-end script reports the following scenarios:

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
- data-plane request-hit rate and token-reuse ratio
- control-plane request-hit / owner-selection / matched-block ratios
- transfer / copy count and rebalance success / failure counts

The current `multi-gpu` baseline uses online round-robin dispatch. Control-plane scenarios should use a bounded `--submit-window` such as `4` or `8` when measuring prefix reuse, because routing can only hit prefixes that previous requests have already prefetched and reported to the global page table.
The benchmark records TTFT from explicit first-token events emitted by data-plane workers. Local prefix hit is measured only on each request's initial prefill, excluding retry hits after preemption. Cached-token ratio, prefill attempts, preemptions, and redundant prefill tokens are reported separately. Route hit and prefix-owner hit are reported separately for control-plane scenarios.
All five scenarios use the same requested per-rank KV capacity through
`--kv-block-budget`. The prefix diagnostics table also reports each rank's
actual runtime block capacity, workload theoretical hit upper bound, matched
route-block ratio, and stale-route rate.
For `memory-skew`, deterministic warm-up, pressure, and reuse phases also
report blocks sent, retained at the source, released at the source, chain
transfer plans, hot-prefix transfer ratio, and reuse-phase request/token hits. This distinguishes an
attempted foreground transfer from one that actually relieves capacity and
preserves reusable KV. It also reports actual bytes, source-side transfer time,
effective GiB/s, predicted transfer cost, and predicted saved prefill time.
For an isolated transfer result, `session-handoff` builds prefixes only on the
source side and then continues the same sessions across the NVLink pair.
`--handoff-prefix-groups` controls how many independent sessions cross the pair.
`--handoff-warmup-prompts` controls the cache-building phase length; setting it
to the prefix-group count leaves the remaining requests for repeated reuse,
instead of allowing a 50/50 warm-up phase to dilute the measured benefit.
Workers report per-block access frequency and recency to the control plane.
The control plane derives maximal hot prefix chains from these snapshots and
keeps a persistent candidate queue per NVLink pair. Ingress supplies counts of
not-yet-submitted prefix demand at a workload phase boundary; candidates are
coalesced by directed NVLink pair into bounded batches and dispatched only
while both ranks have low queue pressure. One batch uses one
prepare/execute/publish/finalize transaction and one contiguous KV payload. The phase waits for accepted
placement plans, and that wait remains
inside serving elapsed time. This makes background transfer proactive rather
than waiting for a reuse request to trigger a late copy.
Rejected placement decisions are memoized by prefix, pair, effective reuse
demand, and target capacity, so unchanged block-state reports do not repeatedly
run the same admission calculation. Benchmark JSON separates prompt, cached,
and actually executed uncached prefill tokens, placement wait time, and
per-NVLink-pair candidate lifecycle counters. Completed plans also feed their
dispatch-to-commit latency into the pair-specific transfer cost model.
Workers additionally feed completed uncached-prefill time into a per-rank EWMA,
so placement admission compares observed recomputation cost against the same
all-layer transport protocol used during startup calibration. A completed
replica creates a forecast-bound placement lease. The lease assigns half of
the forecast reuse demand to each valid copy; odd remainders alternate between
source and replica across adjacent prefixes. Matching requests consume these
explicit quotas. This preserves prefix reuse while allowing both GPUs in each
NVLink pair to execute the reuse phase instead of serializing it on either side.
Admission counts only the first avoidable target cold miss when no eviction is
predicted, rather than multiplying the same saving by every future reuse.
Each configured NVLink pair uses a dedicated NCCL process group, and a prepared
plan skips block-ID negotiation. Data-plane workers wait on ingress and control
queue connections together, preventing an idle ingress wait from delaying a
transfer command. The first dispatch-to-commit sample is blended with the
calibrated prior instead of replacing it with cold-start jitter.
Foreground transfer ranks complete prefix chains by reuse value per missing
target block, with recency as a tie-breaker. Admission uses a conservative
wall-clock model and does not sum one request's access count across every block
in its chain; the target inherits source frequency metadata after transfer.
For topology-blind and transfer-only memory-skew scenarios, reuse requests are
placed on the opposite side of each NVLink pair. Serving timing begins after
worker readiness and representative-payload P2P prewarm. Completed source
transfers feed measured excess latency back into a per-pair conservative EWMA,
and each pair admits at most one foreground plan at a time. A completed transfer
adds or moves page-table locations, so later routing may select the new prefix
owner; routing still applies its load cost and is not forced to follow it.
Routing load reservations include expected decode work. When one prefix owner
is materially busier than its NVLink partner, routing may spill directly and
let that request seed the partner. Proactive replicas are planned separately
from completed access observations and queued ingress demand, so routing never
keeps the current request on an overloaded owner in anticipation of an
unfinished copy.
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
