# Benchmarks

`shared_prefix_benchmark.py` runs an online shared-prefix workload across the current LMPool engine stack. It compares five configurations in one horizontal summary table and can export the same results as JSON and a PNG figure.

## Scenarios

- `single-gpu`: one GPU, no global KV pool. Prefix-hit rate is measured with the local `BlockManager` as a single-card cache reference.
- `multi-gpu`: multiple GPUs with online round-robin request dispatch, no global KV sharing and no control-plane routing.
- `multi-gpu-kv-routing`: control-plane routing and global page-table lookup enabled, with a larger KV block budget to focus on routing and prefix-hit behavior.
- `multi-gpu-kv-swapping`: global pool enabled with a smaller KV block budget, requests dispatched round-robin to stress swap / rebalance overhead.
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
- `route hit`: control-plane routing-time prefix-hit rate. Round-robin baselines report zero because they do not query the routing policy.
- `owner hit`: fraction of requests routed to a GPU that already owned the matched prefix block.
- `local hit`: worker-side local prefix-cache hit rate observed during prefill. This is reported for all scenarios, including round-robin baselines, so the number is comparable across dispatch policies.
- `swaps`: number of KV blocks actually transferred by the data plane during rebalance.
- `reb ok` / `reb fail`: number of successful / failed rebalance requests.
- `pinned`: rebalance failures caused by source blocks still being referenced (`ref_count > 0`), which are safe copy candidates but not safe move/eviction victims.
- `no space`: rebalance failures caused by no NVLink target having enough free blocks.
- `no plan`: rebalance failures where the control plane could not build an executable plan.

## Example

If physical GPU 0 and 2 are NVLink-connected, expose them as two logical devices and pass the logical pair `0,1`:

```bash
CUDA_VISIBLE_DEVICES=0,2 UV_CACHE_DIR=/tmp/uvcache uv run python benchmarks/shared_prefix_benchmark.py \
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
- `--nvlink-pairs`: logical NVLink pairs after `CUDA_VISIBLE_DEVICES` remapping, e.g. `0,1` or `0,1;2,3`. Pass an empty string to let the engine try `nvidia-smi topo -m` detection.
- `--routing-max-cached-blocks`: KV block budget for `multi-gpu-kv-routing`.
- `--eviction-max-cached-blocks`: KV block budget for `multi-gpu-kv-swapping`.
- `--goodput-e2e-sla-ms`: end-to-end latency SLA for counting goodput tokens.
- `--skip-pool`: skip `multi-gpu-lmpool`.
- `--output-figure`: write the summary figure to PNG. Parent directories are created automatically, and the script prints `saved figure: ...` on success.
- `--submit-window`: maximum in-flight requests. Use `4` or `8` to let earlier requests populate the global page table before later routing decisions; use `0` or a negative value for burst submission of all requests.

## Notes

- Prefix-hit rates depend on online timing and cache placement. With `--submit-window 0`, all requests are submitted in a burst before workers have finished prefill, so control-plane routing has less opportunity to use newly reported page-table state.
- `multi-gpu` is an online round-robin baseline, not static offline sharding.
- `multi-gpu-kv-swapping` also uses round-robin dispatch, but enables the global pool with a smaller block budget to exercise rebalance behavior.
- Rebalance requests are based on the actual block shortage, not the full sequence block count. Move-style eviction plans filter out `ref_count > 0` blocks; those pinned blocks are candidates for future copy-style replication, but they are not released from the source GPU.
