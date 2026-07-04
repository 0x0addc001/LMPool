# Benchmarks

`shared_prefix_benchmark.py` runs a high-concurrency shared-prefix workload and reports:

- throughput in tokens/s
- goodput in requests/s under the configured e2e SLA
- mean and p95 TTFT
- mean and p95 TTPT
- mean and p95 end-to-end latency
- GPU utilization mean and p95
- GPU memory utilization mean and p95
- routing prefix-hit rate
- optional summary figure export

Scenario names:

- `single-gpu`
- `multi-gpu` (round-robin dispatch across multiple GPUs, without KV sharing)
- `multi-gpu-kv-routing`
- `multi-gpu-kv-swapping`
- `multi-gpu-lmpool`

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python benchmarks/shared_prefix_benchmark.py \
  --num-prompts 32 \
  --prompt-repeat 10 \
  --output-json /tmp/shared_prefix_benchmark.json \
  --output-figure /tmp/shared_prefix_benchmark.png
```

Set `--skip-pool` to skip the `multi-gpu-lmpool` comparison.
