# LMPool Paper Benchmark Runbook

本文档定义当前仓库可用于论文结果的完整实验矩阵。不要在不同配置之间改变模型、prompt
trace、每 rank KV block budget、随机种子或可见 GPU 集合。

## 1. Experiment Matrix

| Claim | Entry | Workload | Required comparison |
| --- | --- | --- | --- |
| NVLink KV data path | `benchmark_kv_transfer.py` | block-count sweep | latency, P95, GiB/s, validation |
| KV-aware routing | `benchmark_kv_routing.py` | `locality` | single GPU, round-robin, routing-only |
| Foreground transfer | `benchmark_e2e.py` | `memory-skew` | all five configurations, background disabled |
| Full composition | `benchmark_e2e.py` | `session-handoff` | all five configurations, warm-up/reuse separated |
| Load robustness (supplementary) | `benchmark_e2e.py` | `load-skew` | all five configurations, background disabled |

`benchmark_e2e.py --workload locality` 与独立 routing benchmark 重复，不作为额外主实验。
`load-skew` 只能支持负载鲁棒性结论，不能单独证明 transfer 收益。

## 2. Environment And Topology

以下命令针对当前机器物理 NVLink pairs `(0,1)`、`(3,4)`、`(5,6)`。经过
`CUDA_VISIBLE_DEVICES=0,1,3,4,5,6` 重映射后，脚本必须使用逻辑 pairs
`0,1;2,3;4,5`。

```bash
cd /home/jialiangli/LMPool
set -o pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR=/tmp/uvcache
export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6
export MODEL=/home/jialiangli/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export OUT="benchmarks/results/paper/${RUN_ID}"
mkdir -p "${OUT}"/{environment,kv_transfer,routing,memory_skew,session_handoff,load_skew}

nvidia-smi -L | tee "${OUT}/environment/gpus.txt"
nvidia-smi topo -m | tee "${OUT}/environment/topology.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,power.limit \
  --format=csv | tee "${OUT}/environment/gpu_inventory.csv"
git rev-parse HEAD | tee "${OUT}/environment/git_revision.txt"
git status --short | tee "${OUT}/environment/git_status.txt"

uv run python -c 'import torch; assert torch.cuda.device_count() == 6; print(torch.__version__, torch.cuda.device_count())'
test -f "${MODEL}/config.json"
```

先关闭机器上的其他 GPU workload，并保持 persistence mode、power limit 和 clocks 在所有
实验中一致。论文结果至少使用 `--repetitions 3`；下面统一使用 `5`。

## 3. Tests Before Benchmarking

CPU/模拟测试：

```bash
CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/uvcache \
  uv run pytest -q 2>&1 | tee "${OUT}/environment/pytest_cpu.log"
```

真实 NCCL round-trip 测试只需要一个物理 NVLink pair：

```bash
RUN_NCCL_INTEGRATION=1 CUDA_VISIBLE_DEVICES=0,1 UV_CACHE_DIR=/tmp/uvcache \
  uv run pytest -q tests/test_kv_transfer.py -s \
  2>&1 | tee "${OUT}/environment/pytest_nccl.log"
```

## 4. NVLink KV Transfer Microbenchmark

该实验不加载模型，只验证真实 Qwen3-0.6B KV shape 下的 P2P 数据通路。使用同一个 NVLink
pair 对 1/2/4/8 blocks 做 sweep：

```bash
CUDA_VISIBLE_DEVICES=0,1 UV_CACHE_DIR=/tmp/uvcache \
  uv run python benchmarks/benchmark_kv_transfer.py \
  --num-layers 28 \
  --block-size 256 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --block-counts 1,2,4,8 \
  --iterations 100 \
  --warmup 20 \
  --output-json "${OUT}/kv_transfer/summary.json" \
  --output-figure "${OUT}/kv_transfer/summary.png" \
  2>&1 | tee "${OUT}/kv_transfer/run.log"
```

所有 payload 的 `data_validation` 必须为 `passed`。E2E cost model 应采用与线上常见
transfer block 数最接近的一档有效带宽，不使用 NVLink 标称带宽。

## 5. KV-Aware Routing

该入口严格关闭 foreground/background transfer，只验证 cache locality 与 load-aware routing：

```bash
uv run python benchmarks/benchmark_kv_routing.py \
  --model-name-or-path "${MODEL}" \
  --world-size 6 \
  --num-prompts 192 \
  --prompt-repeat 16 \
  --max-tokens 64 \
  --temperature 0.6 \
  --ignore-eos \
  --seed 0 \
  --repetitions 5 \
  --locality-prefix-groups 16 \
  --nvlink-pairs "0,1;2,3;4,5" \
  --submit-window 16 \
  --kv-block-budget 64 \
  --gpu-memory-utilization 0.5 \
  --goodput-e2e-sla-ms 120000 \
  --output-json "${OUT}/routing/summary.json" \
  --output-figure "${OUT}/routing/summary.png" \
  2>&1 | tee "${OUT}/routing/run.log"
```

验收条件：routing-only 的 transfer counters 必须为 0；相对 `multi-gpu`，应提高 DP token
reuse，并改善 throughput、TTFT 或尾延迟，且 per-rank request share 不应重新集中到单卡。

## 6. Foreground Transfer Under Memory Skew

该实验关闭 background copy，只验证容量不足时的 foreground transfer。64-block budget 是
所有五个配置共用的受限容量，不允许单独缩小 transfer 场景预算。

```bash
uv run python benchmarks/benchmark_e2e.py \
  --model-name-or-path "${MODEL}" \
  --world-size 6 \
  --workload memory-skew \
  --memory-skew-prefix-groups 15 \
  --num-prompts 128 \
  --prompt-repeat 16 \
  --max-tokens 64 \
  --temperature 0.6 \
  --ignore-eos \
  --seed 0 \
  --repetitions 5 \
  --nvlink-pairs "0,1;2,3;4,5" \
  --submit-window 16 \
  --kv-block-budget 64 \
  --gpu-memory-utilization 0.5 \
  --goodput-e2e-sla-ms 120000 \
  --disable-background-copy \
  --foreground-transfer-min-benefit-ratio 1.1 \
  --foreground-transfer-bandwidth-gib-s 13.0 \
  --foreground-transfer-fixed-latency-ms 2.0 \
  --foreground-transfer-interference-multiplier 1.2 \
  --kv-transfer-prewarm-blocks 4 \
  --output-json "${OUT}/memory_skew/summary.json" \
  --output-figure "${OUT}/memory_skew/summary.png" \
  2>&1 | tee "${OUT}/memory_skew/run.log"
```

运行 transfer microbenchmark 后，应将 `13.0` 替换为 4-block 或实际线上 batch 对应的实测
GiB/s。验收时同时检查 `sent blocks`、`source freed`、`fg ok` 和 reuse-phase token ratio；
如果 transfer counters 为 0，该结果不能证明 foreground transfer。

## 7. Full LMPool Session Handoff

32 条 warm-up 请求在 source 建立 32 个会话前缀，其余 96 条请求在 NVLink partner 复用。
该 workload 用于端到端验证 proactive transfer、routing 与 pair 内并行执行的组合收益。

```bash
uv run python benchmarks/benchmark_e2e.py \
  --model-name-or-path "${MODEL}" \
  --world-size 6 \
  --workload session-handoff \
  --handoff-prefix-groups 32 \
  --handoff-warmup-prompts 32 \
  --num-prompts 128 \
  --prompt-repeat 16 \
  --max-tokens 64 \
  --temperature 0.6 \
  --ignore-eos \
  --seed 0 \
  --repetitions 5 \
  --nvlink-pairs "0,1;2,3;4,5" \
  --submit-window 64 \
  --kv-block-budget 128 \
  --gpu-memory-utilization 0.5 \
  --goodput-e2e-sla-ms 120000 \
  --background-copy-max-blocks 8 \
  --background-copy-hot-threshold 1 \
  --background-copy-cooldown-s 0.1 \
  --background-copy-expected-reuses 4 \
  --foreground-transfer-min-benefit-ratio 1.1 \
  --foreground-transfer-bandwidth-gib-s 13.0 \
  --foreground-transfer-fixed-latency-ms 2.0 \
  --foreground-transfer-interference-multiplier 1.2 \
  --kv-transfer-prewarm-blocks 4 \
  --output-json "${OUT}/session_handoff/summary.json" \
  --output-figure "${OUT}/session_handoff/summary.png" \
  2>&1 | tee "${OUT}/session_handoff/run.log"
```

主结论优先使用 reuse-phase 图和 JSON，同时报告 aggregate 指标。LMPool 必须与
`multi-gpu`、routing-only、transfer-only 三个消融比较；只优于 single GPU 不足以支持组合贡献。

## 8. Supplementary Load-Skew Robustness

该实验观察 routing 在热点 owner 负载倾斜下是否保持并行度。它不是 transfer 主实验：

```bash
uv run python benchmarks/benchmark_e2e.py \
  --model-name-or-path "${MODEL}" \
  --world-size 6 \
  --workload load-skew \
  --num-prompts 192 \
  --prompt-repeat 16 \
  --max-tokens 64 \
  --temperature 0.6 \
  --ignore-eos \
  --seed 0 \
  --repetitions 5 \
  --nvlink-pairs "0,1;2,3;4,5" \
  --submit-window 16 \
  --kv-block-budget 64 \
  --gpu-memory-utilization 0.5 \
  --goodput-e2e-sla-ms 120000 \
  --disable-background-copy \
  --output-json "${OUT}/load_skew/summary.json" \
  --output-figure "${OUT}/load_skew/summary.png" \
  2>&1 | tee "${OUT}/load_skew/run.log"
```

## 9. Result Acceptance

每个 JSON/日志至少检查：所有请求完成、每个场景 repetitions 数正确、无 worker/control
timeout、无 NCCL watchdog 错误、实际每-rank KV capacity 等于请求 budget。论文表格报告
mean 和标准差；延迟同时报告 mean、P90、P95。出现 transfer/rebalance failure 时必须结合
failure reason 解释，不能只比较总吞吐。
