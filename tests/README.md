# Tests

This directory contains module-level tests for the current engine layout.

The coverage is organized around the runtime structure:

- control plane
- data plane
- global block manager
- global scheduler
- end-to-end orchestration
- local scheduler
- local block manager
- model runner
- KV transfer
- sequence
- launcher / supervisor

## Test Files

- `tests/conftest.py`
  - pytest path setup
  - adds the repository root and `src/` to `sys.path` so tests can import
    `lmpool.*` and reusable benchmark helpers directly

- `tests/test_benchmark_e2e.py`
  - balanced multi-prefix locality workload generation
  - deterministic seeded request ordering
  - memory-skew and two-phase session-handoff trace construction
  - common KV-budget and transfer-placement validation
  - schema-v2 JSON output and repeated-run confidence intervals

- `tests/test_benchmark_utils.py`
  - Qwen3-0.6B / Qwen3-1.7B dynamic runtime-config resolution
  - model/KV dtype byte consistency
  - exact benchmark metadata capture

- `tests/test_benchmark_kv_routing.py`
  - routing benchmark's independent argument surface
  - exact shared KV-budget propagation
  - routing-only transfer-path isolation

- `tests/test_benchmark_kv_transfer.py`
  - transfer block-count sweep parsing and validation
  - model-shaped transfer-contract resolution
  - machine-readable schema-v2 JSON export contract

- `tests/test_control_plane.py`
  - control-plane routing
  - idempotent prepare / execute / publish / finalize rebalance handling
  - epoch-aware, versioned block-state ingestion and worker isolation
  - concurrent control-plane client response demultiplexing
  - forecast placement, negative-cache, and replica-lease routing

- `tests/test_data_plane.py`
  - worker process loop
  - sequence reception and forwarding
  - local scheduling and finished / idle reporting
  - event-driven queue handling without unreliable `Queue.empty()` checks

- `tests/test_global_block_manager.py`
  - global page table maintenance
  - prefix lookup
  - NVLink topology parsing
  - eviction candidate selection
  - transfer bookkeeping, block generations, and unavailable-rank filtering

- `tests/test_global_scheduler.py`
  - route scoring
  - prefix-hit selection
  - free-space fallback
  - rebalance plan generation

- `tests/test_e2e.py`
  - LLMEngine ingress routing
  - worker result aggregation
  - runtime metric propagation

- `tests/test_kv_transfer.py`
  - per-layer and all-layer contiguous KV payload helpers
  - event-driven ingress/control queue wake-up for idle data-plane workers
  - dedicated-pair NCCL planned-transfer fast path
  - NCCL transfer round trip
  - hardware-gated integration path

- `tests/test_llm_engine.py`
  - launcher / supervisor orchestration
  - prompt ingress
  - worker message collection
  - serialized stepping and idempotent shutdown

- `tests/test_model_runner.py`
  - KV cache allocation
  - prefill / decode input preparation
  - transfer forwarding
  - KV cache lookup helpers
  - model-config dtype resolution

- `tests/test_rotary_embedding.py`
  - configured FP16/BF16 rotary-cache dtype
  - low-precision query/key dtype preservation through RoPE

- `tests/test_scheduler.py`
  - local waiting / running queue transitions
  - prefill scheduling
  - decode scheduling
  - decode-headroom admission and preemption avoidance
  - local rebalance trigger path

- `tests/test_block_manager.py`
  - local allocation, prefix caching, and dependency-safe reclamation
  - physical block generation checks against ABA reuse
  - transfer source locking and commit-time target publication

- `tests/test_sequence.py`
  - `Sequence` state and serialization
  - block accounting
  - remaining decode-block accounting and runtime counters
  - remote-prefix metadata

## Running

Run the full suite:

```bash
UV_CACHE_DIR=/tmp/uvcache uv run pytest -q
```

Run a single module:

```bash
UV_CACHE_DIR=/tmp/uvcache uv run pytest -q tests/test_model_runner.py
```

## Hardware-Gated Test

`tests/test_kv_transfer.py` contains an NCCL integration test that only runs when all of the following are true:

- `RUN_NCCL_INTEGRATION=1`
- CUDA is available
- at least 2 CUDA devices are visible

If those conditions are not met, the NCCL test is skipped.

## Notes

- Most tests use light-weight fakes or monkeypatching to avoid full model initialization.
- The suite is structured to keep unit tests fast while still preserving a small number of end-to-end integration checks.
