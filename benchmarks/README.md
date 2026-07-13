# Benchmarks

`benchmark_e2e.py` runs an online shared-prefix workload across the current LMPool engine stack. It compares five configurations in one horizontal summary table and can export the same results as JSON and a PNG figure.

Canonical entries:

- `benchmark_e2e.py`: full end-to-end comparison.
- `benchmark_kv_routing.py`: routing-focused comparison.
- `benchmark_kv_transfer.py`: direct KV transfer microbenchmark.

Compatibility entries `shared_prefix_benchmark.py` and `kv_transfer_benchmark.py` are kept for older commands.

## Scenarios

- `single-gpu`: one GPU, no global KV pool. Prefix-hit rate is measured with the local `BlockManager` as a single-card cache reference.
- `multi-gpu`: multiple GPUs with online round-robin request dispatch, no global KV sharing and no control-plane routing.
- `multi-gpu-kv-routing`: control-plane routing and global page-table lookup enabled, with foreground rebalance and background copy disabled. This is the routing-only baseline.
- `multi-gpu-kv-transfer`: global pool enabled with a smaller KV block budget, requests dispatched round-robin to stress transfer / rebalance overhead.
- `multi-gpu-lmpool`: full LMPool path with control-plane routing, global page table, data-plane workers, and rebalance support.

## Metrics

- `tput(tok/s)`: generated output tokens per second.
- `goodput`: generated output tokens per second for requests whose end-to-end latency is within `--goodput-e2e-sla-ms`.
- `ttft(ms)`: mean time from request submission to the first generated token event reported by the data-plane worker.
- `ttpt(ms)`: mean per-output-token latency proxy, computed as request E2E latency divided by the number of generated output tokens.
- `e2e(ms)`: mean end-to-end request latency.
- `p95(e2e)`: p95 end-to-end request latency.
- `gpu util`: mean GPU utilization sampled from `nvidia-smi`.
- `mem util`: mean GPU memory utilization sampled from `nvidia-smi`.
- `route hit`: control-plane routing-time prefix-hit rate, including validated route-cache hits. Round-robin baselines report zero because they do not query the routing policy.
- `owner hit`: fraction of requests routed to a GPU that already owned the matched prefix block.
- `local hit`: request-level worker-side local prefix-cache hit rate. A request counts as a local hit if any prefill attempt observes cached prefix tokens. This is reported for all scenarios, including round-robin baselines, so the number is comparable across dispatch policies without being diluted by preemption/retry events.
- `transfers`: number of KV blocks actually transferred by the data plane during rebalance.
- `copies`: number of transferred KV blocks that used copy-style transfer and kept the source block live.
- `fg ok` / `fg fail`: number of successful / failed foreground rebalance requests. Foreground rebalance is the current-request path that tries to free local KV blocks with move-style transfer.
- `bg ok` / `bg fail`: number of successful / failed background speculative copy plans. Background copy is the non-blocking path that replicates hot prefix blocks to an NVLink peer for future requests.
- `pinned`: rebalance failures caused by source blocks still being referenced (`ref_count > 0`), which are safe copy candidates but not safe move/eviction victims.
- `no space`: rebalance failures caused by no NVLink target having enough free blocks.
- `no plan`: rebalance failures where the control plane could not build an executable plan.
- `bg space`: background copy failures caused by the target rank not having enough free blocks during prepare.

## Example

If physical GPU 0 and 2 are NVLink-connected, expose them as two logical devices and pass the logical pair `0,1`:

```bash
CUDA_VISIBLE_DEVICES=0,2 UV_CACHE_DIR=/tmp/uvcache uv run python benchmarks/benchmark_e2e.py \
  --num-prompts 32 \
  --prompt-repeat 10 \
  --max-tokens 16 \
  --temperature 0.6 \
  --output-json /tmp/shared_prefix_benchmark.json \
  --nvlink-pairs 0,1 \
  --submit-window 4 \
  --goodput-e2e-sla-ms 120000 \
  --output-figure /tmp/shared_prefix_benchmark.png
```

The script prints `saved json: ...` and `saved figure: ...` after successful export. Parent directories are created automatically.

## Parameters

- `--num-prompts`: total number of requests in the benchmark.
- `--prompt-repeat`: lengthens the shared prefix by repeating a fixed instruction block; larger values make prefix reuse easier to observe.
- `--max-tokens`: maximum generated tokens per request.
- `--temperature`: sampling temperature.
- `--output-json`: write raw scenario results to JSON. Parent directories are created automatically, and the script prints `saved json: ...` on success.
- `--model-name-or-path`: model name or local model path. The default config targets `Qwen/Qwen3-0.6B`.
- `--nvlink-pairs`: logical NVLink pairs after `CUDA_VISIBLE_DEVICES` remapping, e.g. `0,1` or `0,1;2,3`. Quote values containing semicolons, e.g. `--nvlink-pairs "0,2;1,3;4,5;6,7"`. Pass an empty string to let the engine try `nvidia-smi topo -m` detection.
- `--world-size`: number of data-plane worker ranks for multi-GPU scenarios. The default is `2`; use `--world-size 8` for eight visible GPUs.
- `--routing-max-cached-blocks`: KV block budget for `multi-gpu-kv-routing`.
- `--eviction-max-cached-blocks`: KV block budget for `multi-gpu-kv-transfer`.
- `--goodput-e2e-sla-ms`: end-to-end latency SLA for counting goodput tokens.
- `--skip-pool`: skip `multi-gpu-lmpool`.
- `--output-figure`: write the summary figure to PNG. Parent directories are created automatically, and the script prints `saved figure: ...` on success.
- `--submit-window`: maximum in-flight requests. Use `4` or `8` to let earlier requests populate the global page table before later routing decisions; use `0` or a negative value for burst submission of all requests.
- `--disable-background-copy`: disable background speculative copy-style transfer.
- `--background-copy-max-blocks`: maximum prefix blocks copied by one background plan. Use `1` for correctness debugging and try `2` when measuring possible locality gains.
- `--background-copy-cooldown-s`: cooldown before the same prefix can trigger another copy on the same source-target pair. Try `0.5` when evaluating background copy impact.

Routing load-score defaults are set in `MODEL_CONFIG` inside `shared_prefix_benchmark.py`:

- `route_load_weight`: multiplier for token-aware load in the route score.
- `route_waiting_token_weight`: weight for queued prefill tokens.
- `route_running_token_weight`: weight for tokens already owned by running sequences.
- `route_running_sequence_weight`: fixed load weight per active running sequence.
- `route_load_bypass_threshold`: how much higher a prefix owner's load can be before routing bypasses locality and chooses a less-loaded candidate.

## Notes

- Prefix-hit rates depend on online timing and cache placement. With `--submit-window 0`, all requests are submitted in a burst before workers have finished prefill, so control-plane routing has less opportunity to use newly reported page-table state.
- `multi-gpu` is an online round-robin baseline, not static offline sharding.
- `multi-gpu-kv-transfer` also uses round-robin dispatch, but enables the global pool with a smaller block budget to exercise transfer / rebalance behavior.
- Rebalance requests are based on the actual block shortage, not the full sequence block count. Move-style eviction plans filter out `ref_count > 0` blocks; those pinned blocks are candidates for future copy-style replication, but they are not released from the source GPU.
- To evaluate background speculative copy itself, start with a less constrained KV budget such as `--eviction-max-cached-blocks 32 --background-copy-max-blocks 2 --background-copy-cooldown-s 0.5`. A budget of `8` is useful for failure analysis, but it often leaves too little target space for copy replication to improve local hits.
- For eight-GPU runs, pass both `CUDA_VISIBLE_DEVICES=...` and `--world-size 8`. NVLink pairs use the logical GPU indices after CUDA remapping.

## KV Transfer Microbenchmark

Use `benchmark_kv_transfer.py` to validate the data-plane transfer primitive directly:

```bash
CUDA_VISIBLE_DEVICES=0,2 UV_CACHE_DIR=/tmp/uvcache uv run python benchmarks/benchmark_kv_transfer.py \
  --num-layers 28 \
  --block-size 256 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --num-transfer-blocks 2 \
  --iterations 5 \
  --warmup 1
```

It reports mean / p95 transfer latency, transferred bytes per iteration, effective GiB/s, and data validation status. This benchmark isolates transfer from routing and model execution.
