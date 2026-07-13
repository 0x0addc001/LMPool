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
  - adds `src/` to `sys.path` so tests can import `lmpool.*` directly

- `tests/test_control_plane.py`
  - control-plane routing
  - rebalance request / response handling
  - block-state ingestion
  - control-plane client message flow

- `tests/test_data_plane.py`
  - worker process loop
  - sequence reception and forwarding
  - local scheduling and finished / idle reporting

- `tests/test_global_block_manager.py`
  - global page table maintenance
  - prefix lookup
  - NVLink topology parsing
  - eviction candidate selection
  - transfer bookkeeping

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
  - KV transfer helpers
  - NCCL transfer round trip
  - hardware-gated integration path

- `tests/test_llm_engine.py`
  - launcher / supervisor orchestration
  - prompt ingress
  - worker message collection

- `tests/test_model_runner.py`
  - KV cache allocation
  - prefill / decode input preparation
  - transfer forwarding
  - KV cache lookup helpers

- `tests/test_scheduler.py`
  - local waiting / running queue transitions
  - prefill scheduling
  - decode scheduling
  - local rebalance trigger path

- `tests/test_sequence.py`
  - `Sequence` state and serialization
  - block accounting
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
