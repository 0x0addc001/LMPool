<p align="center">
  <img src="./assets/fig_logo_dark.png" width="40%" height="40%" alt="LMPool">
</p>

<p align="center">
  <a href="./README.md"><b>English</b></a> |
  <a href="./README_zh.md"><b>简体中文</b></a>
</p>

# LMPool: KV-Aware Routing and NVLink Transfer for Multi-GPU LLM Serving

LMPool is a research prototype built on [Mini-vLLM](https://github.com/Wenyueh/MinivLLM). It coordinates the physically local PagedAttention KV caches of multiple data-parallel LLM replicas through a dedicated control plane.

The design follows two principles:

1. **Routing for cache locality.** Send a request to a GPU that already holds its useful KV prefix when the saved prefill work outweighs queue and capacity pressure.
2. **Transfer for cache fluidity.** When KV placement and request load diverge, copy or move valuable blocks over direct NVLink if the expected reuse amortizes data movement.

In short: route to avoid transfer; use fast transfer only when movement is unavoidable and profitable. LMPool is a logically coordinated KV pool, not transparent shared HBM. Each worker remains the physical owner of its blocks and KV tensor.

## Architecture

![LMPool architecture](./assets/fig_architecture_dark.png)

For `N` GPUs, the runtime contains one user/launcher process, one independent control-plane process, and `N` identical data-plane processes, including a separate worker for rank 0.

| Component | Role |
| --- | --- |
| `LLMEngine` | User API, master-side ingress, process launch/supervision, request forwarding, completion aggregation |
| `control_plane_process` | Owns `GlobalScheduler` and `GlobalBlockManager`; handles routing, global metadata, transfer plans, leases, and heartbeat |
| `ControlPlaneClient` | Queue-based protocol endpoint used by LLMEngine and workers to communicate with the control process |
| `data_plane_process` | One per GPU; owns the local `Scheduler`, `BlockManager`, and `ModelRunner` |
| `KV Transfer` | Packs complete KV blocks and performs paired NCCL send/receive on direct NVLink pairs |

### Request Flow

1. LLMEngine creates a `Sequence` and sends its complete-block prefix hashes and load metadata to the control plane.
2. `GlobalScheduler` chooses a target rank before the full sequence enters a worker queue.
3. The selected data-plane process performs local prefill and decode.
4. The worker publishes versioned block snapshots after state changes; the control plane replaces that rank's global metadata atomically.
5. A real local block shortage may trigger a cost-gated foreground transfer. Predictable future reuse may trigger a background copy and placement lease.
6. Finished sequences and per-rank metrics return to LLMEngine.

## Core Mechanisms

### KV-Aware Routing

`GlobalScheduler.route_sequence_meta()` matches the longest contiguous chain of complete prefix blocks. Candidates are the initial GPU and only its directly connected NVLink partners; unrelated GPUs are excluded.

Topology affinity is `2` for the same GPU, `1` for a direct NVLink partner, and `0` otherwise. Selection combines missing prefill work, waiting/running load, decode-weighted work, effective free capacity, and reclaim cost. A prefix owner can be bypassed when it is sufficiently overloaded and the extra recomputation cost is bounded. Route reservations prevent concurrent requests from consuming the same stale capacity estimate.

### Global and Local Block State

The control process owns an authoritative global page table:

```text
prefix hash -> [(gpu, physical block, generation, readiness), ...]
```

It also tracks per-GPU free/reclaimable capacity, parent dependencies, recency/frequency, in-flight blocks, worker epochs, and snapshot versions. Workers do not share this Python object. Each local `BlockManager` remains authoritative for allocation, reference counts, block readiness, and the physical KV tensor, then reports a versioned snapshot to the control process.

Cached complete blocks are reclaimed with dependency-safe LFU-first/LRU-second ordering. Live blocks are not move-eviction victims. They may be copied when replication has sufficient expected value.

### NVLink KV Transfer

Foreground transfer requests only the actual shortage, not an entire sequence. Background placement batches hot prefix chains for predicted reuse. Both are admitted only when source validity, destination capacity, minimum batch size, and the estimated saved-prefill/transfer-cost ratio are acceptable.

Each plan follows an idempotent transaction:

```text
prepare -> execute -> publish -> finalize
                    \-> abort on failure
```

`prepare` reserves concrete destination blocks. `execute` sends one packed all-layer K/V payload per direction. `publish` makes received blocks visible. `finalize` releases source blocks only for move semantics; copy semantics retain both replicas. Hash and physical-block generation checks reject stale plans.

### Consistency and Liveness

- The control plane serializes authoritative state changes in one event loop.
- Per-client receive locks prevent multiple callers from consuming each other's queue responses.
- Worker epoch and monotonic snapshot versions reject stale state.
- Transfer plan IDs and phases are idempotent; reservations and in-flight markers prevent concurrent reuse/reclamation.
- Bidirectional heartbeat detects worker/control failure. LLMEngine can restart the control process and request full worker snapshots.

This is failure detection and metadata recovery, not replicated high availability. Launcher failure still stops the service, and a failed worker can lose its physical cache.

## Repository Layout

```text
src/lmpool/engine/
  llm_engine.py             launcher, ingress, supervision
  control_plane.py          protocol, client, control-process event loop
  global_scheduler.py       route and transfer-plan decisions
  global_block_manager.py   global page table and block snapshots
  data_plane.py             per-GPU worker and transfer-phase executor
  scheduler.py              local prefill/decode scheduling
  block_manager.py          local allocation and prefix cache
  model_runner.py           model/KV execution and metrics
  kv_transfer.py            packed NCCL transfer primitives
  sequence.py               request and block-table state

benchmarks/                 publishable microbenchmarks and E2E workloads
tests/                      module, protocol, benchmark, and integration tests
docs/paper/                 paper source and bibliography
```

Legacy internal functions named `swap_out`/`swap_in` remain in `kv_transfer.py`; public documentation uses **transfer** because the operation can be either a copy or a move and does not imply a CPU swap tier.

## Installation and Basic Run

Python 3.11 and CUDA-capable PyTorch are required.

```bash
uv sync --group dev
```

Set `world_size`, the model path, and logical NVLink pairs in `main.py` to match visible GPUs, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 UV_CACHE_DIR=/tmp/uvcache uv run python main.py
```

GPU IDs inside `nvlink_topo.pairs` are logical IDs after `CUDA_VISIBLE_DEVICES` remapping. If topology is omitted, LMPool attempts to parse `nvidia-smi topo -m` and retains only `NV#` links. Verify the physical topology before every experiment; do not infer a pair from socket or NUMA placement.

## Evaluation

Three complete benchmark entry points are retained:

| Entry | Claim |
| --- | --- |
| `benchmarks/benchmark_kv_transfer.py` | NCCL/NVLink payload latency, bandwidth, and data equality |
| `benchmarks/benchmark_kv_routing.py` | Routing-only locality and prefill-reuse benefit |
| `benchmarks/benchmark_e2e.py` | Five-way system comparison under load-skew, memory-skew, and session-handoff workloads |

The exact dual-model paper matrix, fixed variables, offline model paths, acceptance criteria, and commands are in [benchmarks/PAPER_RUNBOOK.md](./benchmarks/PAPER_RUNBOOK.md). Metric and workload definitions are in [benchmarks/README.md](./benchmarks/README.md).

### Current Paper Batch

Artifacts: [`benchmarks/results/paper/20260719T072508Z`](./benchmarks/results/paper/20260719T072508Z)

The batch uses five repetitions, six RTX 3090 GPUs arranged as three NV4 pairs, BF16 Qwen3-0.6B/Qwen3-1.7B, 256-token KV blocks, and equal per-worker block budgets.

- **Transfer microbenchmark:** four-block batches sustain 19.0-23.2 GiB/s; eight-block batches sustain 26.1-30.1 GiB/s. Every payload validation passes.
- **Routing workload:** routing raises cached prompt-token ratio from about 44% to 72% and reduces uncached prefill tokens by about 50% for both models. Throughput improves by 2.2-2.7%, while mean TTFT falls by 10.6-20.2%.
- **Session handoff:** full LMPool improves throughput by 4.2%/7.1%, lowers mean TTFT by 33.2%/42.6%, and lowers mean E2E latency by 9.9%/13.7% for Qwen3-0.6B/Qwen3-1.7B relative to round-robin multi-GPU.
- **Boundary results:** steady load skew admits no transfer and stays near the multi-GPU baseline. The current memory-skew trace triggers few foreground plans and does not improve throughput; it is a negative/boundary result, not evidence of universal transfer benefit.

The paper reports all four observations. Session handoff is the main end-to-end transfer result; memory/load skew are not silently omitted.

## Tests

The test tree mirrors engine and benchmark modules. It covers allocation/reclamation, chained hashes, route hit/miss/capacity cases, load bypass, direct-pair filtering, page-table epochs/snapshots, reservations, transactional transfer phases, queue concurrency, process lifecycle, model/KV dtype, benchmark schemas/plots, and end-to-end completion.

CPU and simulated tests:

```bash
CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/uvcache uv run pytest -q
```

Two-rank NCCL data-equality/deadlock test on one physical NVLink pair:

```bash
RUN_NCCL_INTEGRATION=1 CUDA_VISIBLE_DEVICES=0,1 UV_CACHE_DIR=/tmp/uvcache \
  uv run pytest -q tests/test_kv_transfer.py -s
```

See [tests/README.md](./tests/README.md) for the test-to-module map and hardware gates.

## Scope and Limitations

- Current transfer decisions use direct same-node NVLink pairs only; no PCIe/NUMA fallback is scored.
- The global page table coordinates metadata; it does not provide transparent remote block addressing.
- The paper workloads are deterministic synthetic traces, not production datasets.
- The current evidence supports routing and session handoff. It does not show that transfer improves every memory- or request-skew workload.
- The prototype has heartbeat and control-process restart but no replicated controller or launcher HA.
- Cross-node RDMA, CPU/SSD cache tiers, persistent KV cache, and heterogeneous model replicas are out of scope.

## Paper

The [paper directory](./docs/paper/README.md) contains the source, verified bibliography, reproducible architecture figure, and build instructions. [example_paper.tex](./docs/paper/example_paper.tex) covers the motivation, detailed design, dataset/workload profiling, test methodology, evaluation, limitations, and related work synchronized with the current code and the paper artifact batch.
