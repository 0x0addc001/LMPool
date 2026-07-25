# LMPool Paper Benchmark Runbook

本文档定义当前仓库可用于论文结果的完整实验矩阵。每项主实验必须分别使用
Qwen3-0.6B 和 Qwen3-1.7B；同一模型内不要在不同配置之间改变 prompt trace、每 rank
KV block budget、随机种子或可见 GPU 集合。跨模型比较保持 workload 参数一致，模型结构、
KV geometry 和 dtype 则由各自 `config.json` 自动解析。

## 1. Experiment Matrix

| Claim | Entry | Workload | Required comparison |
| --- | --- | --- | --- |
| NVLink KV data path | `benchmark_kv_transfer.py` | block-count sweep | latency, P95, GiB/s, validation |
| KV-aware routing | `benchmark_kv_routing.py` | `locality` | single GPU, round-robin, routing-only |
| Foreground transfer | `benchmark_e2e.py` | `memory-skew` | all five configurations, background disabled |
| Full composition | `benchmark_e2e.py` | `session-handoff` | all five configurations, warm-up/reuse separated |
| Load robustness (supplementary) | `benchmark_e2e.py` | `load-skew` | all five configurations, background disabled |
| Model-scale robustness | all entries | Qwen3-0.6B / Qwen3-1.7B | same trace and policy parameters, model-specific runtime config |

`benchmark_e2e.py --workload locality` 与独立 routing benchmark 重复，不作为额外主实验。
`load-skew` 只能支持负载鲁棒性结论，不能单独证明 transfer 收益。

## 2. Common Fixed Inputs

以下命令针对当前机器物理 NVLink pairs `(0,1)`、`(3,4)`、`(5,6)`。经过
`CUDA_VISIBLE_DEVICES=0,1,3,4,5,6` 重映射后，脚本必须使用逻辑 pairs
`0,1;2,3;4,5`。

本节只定义自动和手工模式共用的固定输入，不启动实验，也不定义 `MODEL`、`RUN_ID` 或
`OUT`。每次打开新 shell 后先完整执行一次：

两种运行方式互斥，不要交叉执行：

| 目标 | 执行路径 | 不需要执行 |
| --- | --- | --- |
| 完整采集双模型论文数据（推荐） | 第 2 节 -> 第 5 节预检（首次一次）-> 第 3 节 | 第 4 节和第 6--10 节手工命令 |
| 调试或只重跑一个模型/工作负载 | 第 2 节 -> 第 5 节预检（首次一次）-> 第 4 节 -> 第 6--10 节中所需命令 | 第 3 节 runner |

```bash
cd /home/jialiangli/LMPool
set -o pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR=/tmp/uvcache
export MODEL_06B=/home/jialiangli/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
export MODEL_17B=/home/jialiangli/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
export GPU_SET=0,1,3,4,5,6
export WORLD_SIZE=6
export NVLINK_PAIRS="0,1;2,3;4,5"
export TRANSFER_PAIRS="0,1;3,4;5,6"

CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python -c \
  'import torch; assert torch.cuda.device_count() == 6; print(torch.__version__, torch.cuda.device_count())'
test -f "${MODEL_06B}/config.json"
test -f "${MODEL_17B}/config.json"
```

当前已验证的本地模型如下。Qwen3-1.7B 包含两个完整 safetensors 分片，分片文件合计
`4,063,515,592` bytes，本地 cache 占用约 `3.8 GiB`；其配置为 28 layers、hidden size
2048、16 attention heads、8 KV heads、head dimension 128 和 BF16。Qwen3-0.6B cache
占用约 `1.5 GiB`。两个 snapshot 均可在 `HF_HUB_OFFLINE=1` 和
`TRANSFORMERS_OFFLINE=1` 下加载。

正式实验前执行一次纯离线预检，避免把缺失或不完整权重误判为 benchmark 故障：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python - <<'PY'
import os
import hashlib
from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

for variable in ("MODEL_06B", "MODEL_17B"):
    model = Path(os.environ[variable])
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"{variable} has no safetensors shards: {model}")
    tensor_count = 0
    for shard in shards:
        digest = hashlib.sha256()
        with shard.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        expected_digest = shard.resolve().name
        if len(expected_digest) == 64 and digest.hexdigest() != expected_digest:
            raise SystemExit(f"checksum mismatch: {shard}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            tensor_count += len(handle.keys())
    print(
        variable,
        model,
        f"layers={config.num_hidden_layers}",
        f"hidden={config.hidden_size}",
        f"kv_heads={config.num_key_value_heads}",
        f"dtype={config.dtype}",
        f"vocab={len(tokenizer)}",
        f"shards={len(shards)}",
        f"tensors={tensor_count}",
    )
PY

test -z "$(find \
  /home/jialiangli/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B \
  /home/jialiangli/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B \
  -name '*.incomplete' -print -quit)"
df -h /home/jialiangli/.cache/huggingface/hub
```

截至 2026-07-19，当前主文件系统使用率为 `99%`、剩余约 `12 GiB`。全量双模型实验前应
以 `df` 的实时结果确认结果目录、日志和临时文件不会耗尽空间；不要删除上述 snapshot
中的 symlink 或 `blobs/` 文件。

先关闭机器上的其他 GPU workload，并保持 persistence mode、power limit 和 clocks 在所有
实验中一致。论文结果至少使用 `--repetitions 3`；下面统一使用 `5`。

## 3. Mode A: Automated Dual-Model Suite

这是论文结果的推荐模式。先执行第 2 节，再执行下面这一整块。统一运行器会依次对 0.6B 和
1.7B 执行三个物理 NVLink pair 的 transfer sweep、routing、memory-skew、session-handoff
和补充 load-skew。它会读取每个模型的 KV geometry/dtype，在三个物理 pair 上测量
1/2/4/8/16/32/64-block 延迟，将其映射为逻辑 pair 的分段 P95 latency profile，并自动写入
该模型的 E2E transfer 成本参数。运行时按实际 plan 字节数插值，超出测量范围时用末段斜率
外推；完成的 transfer 再按 pair 和 size bucket 更新 EWMA：

空载 microbenchmark 只给出随 payload 变化的数据通路项，不能直接代表完整 serving
事务。当前论文配置使用 `T_static = 40 ms + 1.2 × profile_P95(plan_bytes)`：40 ms 是手工
选择的保守 cold-start prior，用于覆盖调度、协调、NCCL 排队、目标端发布与 block 注册；
它不是由独立 pilot 或 held-out prediction study 拟合得到的参数。1.2 只修正 payload
相关的加载态干扰，运行时观测通过 pair × size-bucket EWMA 保守提高估计。

首次正式采集前先完成第 5 节测试；测试不需要在每个 trial 前重复执行。

```bash
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export OUT="benchmarks/results/paper/${RUN_ID}"
export REPETITIONS=5
export TRANSFER_FIXED_LATENCY_MS=40.0
export TRANSFER_INTERFERENCE_MULTIPLIER=1.2

bash benchmarks/run_paper_suite.sh
```

结果按 `${OUT}/qwen3-0.6b/` 和 `${OUT}/qwen3-1.7b/` 分开保存。运行器默认从标准
Hugging Face cache 自动发现 snapshot；第 2 节导出的 `MODEL_06B` 和 `MODEL_17B` 会将其
固定为本文验证过的版本。任一模型不存在或不含 safetensors 权重时，运行器会在开始任何
GPU 实验前失败，不会联网下载，也不会用另一模型代替。开发阶段可设 `REPETITIONS=1`，
论文结果至少使用 `3`，推荐 `5`。

使用本模式时，runner 已经执行第 6--10 节列出的全部 benchmark。不要随后再手工执行这些
命令，否则只是产生一套重复且目录结构不同的结果。

如果运行中断，保留原来的 `OUT` 并设置 `RESUME=1`。runner 会校验每个 JSON 的结果结构、
模型、重复次数、workload 和 E2E transfer 校准参数，并检查 transfer 的 7 个 payload、
routing 的 3 个场景或 E2E 的 5 个场景是否齐全。它只跳过结构和关键配置都一致的
artifact，只重跑缺失、不完整或配置不一致的实验。例如恢复一个未改变代码和参数的 run：

```bash
export RUN_ID=20260725T031840Z
export OUT="benchmarks/results/paper/${RUN_ID}"
export REPETITIONS=5
export TRANSFER_FIXED_LATENCY_MS=40.0
export TRANSFER_INTERFERENCE_MULTIPLIER=1.2
export RESUME=1

bash benchmarks/run_paper_suite.sh
```

`20260725T031840Z` 当前保留的 transfer workload 结果使用 `40 ms` 固定事务先验。不要在
修改成本参数或代码后继续向该目录拼接结果；应使用新的 `RUN_ID` 对两个模型重跑，保证同一
结果目录内配置一致。早期使用 `2 ms` 的中间 artifact 不属于最终论文结果。

KV transfer microbenchmark 为每个 payload case 使用唯一 FileStore rendezvous，不依赖临时
TCP 端口。若某个 worker 仍异常退出，父进程会立即报告 exit code，而不会等待完整超时。

## 4. Mode B: Manual Single-Model Runs

仅在调试某个 workload 或只重跑一个模型时使用本模式。先执行第 2 节，然后从下面两组
`MODEL`/`MODEL_LABEL` 中只选择一组；不要同时设置两组：

```bash
# Qwen3-0.6B：选择这一组
export MODEL="${MODEL_06B}"
export MODEL_LABEL=qwen3-0.6b

# Qwen3-1.7B：测试 1.7B 时改为下面两行
# export MODEL="${MODEL_17B}"
# export MODEL_LABEL=qwen3-1.7b

export CUDA_VISIBLE_DEVICES="${GPU_SET}"
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export OUT="benchmarks/results/paper/${RUN_ID}/${MODEL_LABEL}"
export REPETITIONS=5
export TRANSFER_PROFILE="${OUT}/kv_transfer/latency_profile.json"
mkdir -p "${OUT}"/{environment,kv_transfer,routing,memory_skew,session_handoff,load_skew}

nvidia-smi -L | tee "${OUT}/environment/gpus.txt"
nvidia-smi topo -m | tee "${OUT}/environment/topology.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,power.limit \
  --format=csv | tee "${OUT}/environment/gpu_inventory.csv"
git rev-parse HEAD | tee "${OUT}/environment/git_revision.txt"
git status --short | tee "${OUT}/environment/git_status.txt"
```

完成设置后，只执行第 6--10 节中需要的手工命令。若要切换模型，重新执行本节并生成新的
`RUN_ID`，不要让 0.6B 和 1.7B 共用同一个 `${OUT}`。

## 5. Tests Before Benchmarking

测试独立于 Mode A/Mode B。在正式实验前运行一次，并把日志写入单独的 preflight 目录：

CPU/模拟测试：

```bash
export PRECHECK_OUT="benchmarks/results/paper/preflight_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${PRECHECK_OUT}"

CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/uvcache \
  uv run pytest -q 2>&1 | tee "${PRECHECK_OUT}/pytest_cpu.log"
```

真实 NCCL round-trip 测试只需要一个物理 NVLink pair：

```bash
RUN_NCCL_INTEGRATION=1 CUDA_VISIBLE_DEVICES=0,1 UV_CACHE_DIR=/tmp/uvcache \
  uv run pytest -q tests/test_kv_transfer.py -s \
  2>&1 | tee "${PRECHECK_OUT}/pytest_nccl.log"
```

## 6. NVLink KV Transfer Microbenchmark

该实验不加载权重，但会从指定模型解析真实 KV shape 和 dtype。对每个模型、每个物理
NVLink pair 分别执行 1/2/4/8/16/32/64 blocks sweep，然后按顺序映射到 E2E 的逻辑 pair：

```bash
transfer_inputs=()
IFS=';' read -r -a physical_pairs <<< "${TRANSFER_PAIRS}"
for pair in "${physical_pairs[@]}"; do
  pair_label="${pair/,/-}"
  pair_json="${OUT}/kv_transfer/pair_${pair_label}.json"
  CUDA_VISIBLE_DEVICES="${pair}" UV_CACHE_DIR=/tmp/uvcache \
    uv run python benchmarks/benchmark_kv_transfer.py \
    --model-name-or-path "${MODEL}" \
    --dtype auto \
    --block-size 256 \
    --block-counts 1,2,4,8,16,32,64 \
    --iterations 100 \
    --warmup 20 \
    --output-json "${pair_json}" \
    --output-figure "${OUT}/kv_transfer/pair_${pair_label}.png" \
    2>&1 | tee "${OUT}/kv_transfer/pair_${pair_label}.log"
  transfer_inputs+=("${pair_json}")
done

uv run python benchmarks/build_transfer_profile.py \
  --inputs "${transfer_inputs[@]}" \
  --logical-pairs "${NVLINK_PAIRS}" \
  --latency-metric p95_latency_ms \
  --output "${TRANSFER_PROFILE}"
```

所有 payload 的 `data_validation` 必须为 `passed`。`latency_profile.json` 保留每个物理
pair 的来源、模型 KV geometry、原始延迟和单调化后的决策延迟。E2E loader 会校验逻辑 pair
以及每 block 字节数，防止 0.6B/1.7B、FP16/BF16 或不同 block size 的 profile 混用。
线上 batch 并非固定档位：foreground plan 等于实际 shortage，background plan 可合并多条
候选；成本模型在相邻测量点间线性插值，并由运行时 pair x size-bucket EWMA 修正。

## 7. KV-Aware Routing

该入口严格关闭 foreground/background transfer，只验证 cache locality 与 load-aware routing：

```bash
uv run python benchmarks/benchmark_kv_routing.py \
  --model-name-or-path "${MODEL}" \
  --dtype auto \
  --world-size "${WORLD_SIZE}" \
  --num-prompts 192 \
  --prompt-repeat 16 \
  --max-tokens 64 \
  --temperature 0.6 \
  --ignore-eos \
  --seed 0 \
  --repetitions "${REPETITIONS}" \
  --locality-prefix-groups 16 \
  --nvlink-pairs "${NVLINK_PAIRS}" \
  --submit-window 16 \
  --kv-block-budget 64 \
  --gpu-memory-utilization 0.5 \
  --goodput-e2e-sla-ms 10000 \
  --output-json "${OUT}/routing/summary.json" \
  --output-figure "${OUT}/routing/summary.png" \
  2>&1 | tee "${OUT}/routing/run.log"
```

验收条件：routing-only 的 transfer counters 必须为 0；相对 `multi-gpu`，应提高 DP token
reuse，并改善 throughput、TTFT 或尾延迟，且 per-rank request share 不应重新集中到单卡。

## 8. Foreground Transfer Under Memory Skew

该实验关闭 background copy，只验证容量不足时的 foreground transfer。64-block budget 是
所有五个配置共用的受限容量，不允许单独缩小 transfer 场景预算。

```bash
uv run python benchmarks/benchmark_e2e.py \
  --model-name-or-path "${MODEL}" \
  --dtype auto \
  --world-size "${WORLD_SIZE}" \
  --workload memory-skew \
  --memory-skew-prefix-groups 15 \
  --num-prompts 128 \
  --prompt-repeat 16 \
  --max-tokens 64 \
  --temperature 0.6 \
  --ignore-eos \
  --seed 0 \
  --repetitions "${REPETITIONS}" \
  --nvlink-pairs "${NVLINK_PAIRS}" \
  --submit-window 16 \
  --kv-block-budget 64 \
  --gpu-memory-utilization 0.5 \
  --goodput-e2e-sla-ms 10000 \
  --disable-background-copy \
  --foreground-transfer-min-benefit-ratio 1.1 \
  --foreground-transfer-profile-json "${TRANSFER_PROFILE}" \
  --foreground-transfer-fixed-latency-ms 40.0 \
  --foreground-transfer-interference-multiplier 1.2 \
  --kv-transfer-prewarm-blocks 4 \
  --output-json "${OUT}/memory_skew/summary.json" \
  --output-figure "${OUT}/memory_skew/summary.png" \
  2>&1 | tee "${OUT}/memory_skew/run.log"
```

运行前必须先由第 6 节生成 `TRANSFER_PROFILE`。验收时同时检查 `sent blocks`、
`source freed`、`fg ok` 和 reuse-phase token ratio；如果 transfer counters 为 0，该结果
不能证明 foreground transfer。

## 9. Full LMPool Session Handoff

32 条 warm-up 请求在 source 建立 32 个会话前缀，其余 96 条请求在 NVLink partner 复用。
该 workload 用于端到端验证 proactive transfer、routing 与 pair 内并行执行的组合收益。

```bash
uv run python benchmarks/benchmark_e2e.py \
  --model-name-or-path "${MODEL}" \
  --dtype auto \
  --world-size "${WORLD_SIZE}" \
  --workload session-handoff \
  --handoff-prefix-groups 32 \
  --handoff-warmup-prompts 32 \
  --num-prompts 128 \
  --prompt-repeat 16 \
  --max-tokens 64 \
  --temperature 0.6 \
  --ignore-eos \
  --seed 0 \
  --repetitions "${REPETITIONS}" \
  --nvlink-pairs "${NVLINK_PAIRS}" \
  --submit-window 64 \
  --kv-block-budget 128 \
  --gpu-memory-utilization 0.5 \
  --goodput-e2e-sla-ms 10000 \
  --background-copy-max-blocks 8 \
  --background-copy-hot-threshold 1 \
  --background-copy-cooldown-s 0.1 \
  --background-copy-expected-reuses 4 \
  --foreground-transfer-min-benefit-ratio 1.1 \
  --foreground-transfer-profile-json "${TRANSFER_PROFILE}" \
  --foreground-transfer-fixed-latency-ms 40.0 \
  --foreground-transfer-interference-multiplier 1.2 \
  --kv-transfer-prewarm-blocks 4 \
  --output-json "${OUT}/session_handoff/summary.json" \
  --output-figure "${OUT}/session_handoff/summary.png" \
  2>&1 | tee "${OUT}/session_handoff/run.log"
```

主结论优先使用 reuse-phase 图和 JSON，同时报告 aggregate 指标。LMPool 必须与
`multi-gpu`、routing-only、transfer-only 三个消融比较；只优于 single GPU 不足以支持组合贡献。

## 10. Supplementary Load-Skew Robustness

该实验观察 routing 在热点 owner 负载倾斜下是否保持并行度。它不是 transfer 主实验：

```bash
uv run python benchmarks/benchmark_e2e.py \
  --model-name-or-path "${MODEL}" \
  --dtype auto \
  --world-size "${WORLD_SIZE}" \
  --workload load-skew \
  --num-prompts 192 \
  --prompt-repeat 16 \
  --max-tokens 64 \
  --temperature 0.6 \
  --ignore-eos \
  --seed 0 \
  --repetitions "${REPETITIONS}" \
  --nvlink-pairs "${NVLINK_PAIRS}" \
  --submit-window 16 \
  --kv-block-budget 64 \
  --gpu-memory-utilization 0.5 \
  --goodput-e2e-sla-ms 10000 \
  --disable-background-copy \
  --foreground-transfer-profile-json "${TRANSFER_PROFILE}" \
  --output-json "${OUT}/load_skew/summary.json" \
  --output-figure "${OUT}/load_skew/summary.png" \
  2>&1 | tee "${OUT}/load_skew/run.log"
```

## 11. Result Acceptance

每个 JSON/日志至少检查：所有请求完成、每个场景 repetitions 数正确、无 worker/control
timeout、无 NCCL watchdog 错误、实际每-rank KV capacity 等于请求 budget。JSON 顶层为
`metadata` 和 `results`：前者保存精确命令、Git revision、模型结构/dtype 和解析后的配置；
后者保存聚合结果及每次 `trial_results`。论文表格报告 mean 和 95% CI，并在附录保留 sample
standard deviation；延迟同时报告 mean、P90、P95。出现 transfer/rebalance failure 时必须
结合 failure reason 解释，不能只比较总吞吐。
