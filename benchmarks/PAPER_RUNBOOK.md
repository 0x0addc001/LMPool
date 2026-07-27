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
| Load relief by proactive background transfer | `benchmark_e2e.py` | `load-skew` | all five configurations, source warm-up then reuse burst |
| Capacity offload by foreground transfer | `benchmark_e2e.py` | `memory-skew` | all five configurations, background disabled |
| Model-scale robustness | all entries | Qwen3-0.6B / Qwen3-1.7B | same trace and policy parameters, model-specific runtime config |

`capacity-offload` 仅保留为 `memory-skew` 的命令行兼容别名，不再产生独立论文目录。
`benchmark_e2e.py --workload locality` 与独立 routing benchmark 重复，不作为额外主实验。
`load-skew` 现在是 background transfer 的负载主实验：warm-up 把 24 条约 5.7K-token
长前缀放在三个 source rank，reuse burst 让 background copy 根据明确 forecast 预先复制。
Prefix group 以两个一组的方式在三个
NVLink source 间交错，使普通 round-robin 在 reuse 时将一半 group 留在 owner，另一半送到
该 owner 的 direct NVLink partner。每个 partner 因而面对 12 次真实冷 prefill，而不是旧版
6-group trace 中仅有的 3 次；routing-only 则会把请求拉回 owner，形成受控负载倾斜。
`memory-skew` 只认 source block release，专门证明 offloading。完整 transaction
calibration 使用独立的内部 `transfer-calibration` trace，只生成成本 profile，不作为论文
workload 或 serving 性能证据。

## 2. Common Fixed Inputs

以下命令针对当前机器物理 NVLink pairs `(0,1)`、`(3,4)`、`(5,6)`。经过
`CUDA_VISIBLE_DEVICES=0,1,3,4,5,6` 重映射后，脚本必须使用逻辑 pairs
`0,1;2,3;4,5`。

本节只定义自动和手工模式共用的固定输入，不启动实验，也不定义 `MODEL`、`RUN_ID` 或
`OUT`。每次打开新 shell 后先完整执行一次：

两种运行方式互斥，不要交叉执行：

| 目标 | 执行路径 | 不需要执行 |
| --- | --- | --- |
| 完整采集双模型论文数据（推荐） | 第 2 节 -> 第 5 节预检（首次一次）-> 第 3 节 | 第 4 节和第 6--9 节手工命令 |
| 调试或只重跑一个模型/工作负载 | 第 2 节 -> 第 5 节预检（首次一次）-> 第 4 节 -> 第 6--9 节中所需命令 | 第 3 节 runner |

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

### 2.1 Workload Preflight

正式双模型采集前，先只用 Qwen3-0.6B 和一次 repetition 验证新 trace。预验可以复用
`20260726T165849Z` 中同一模型、同一拓扑生成的 cost profile，因为本次修改只重构
calibration trace 的名称和 serving workload，不改变 profile schema、KV geometry 或物理
NVLink pair。预验结果必须写入独立目录，不能覆盖或拼接论文目录。推荐直接运行统一脚本：

```bash
export PREFLIGHT_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export TRANSFER_PROFILE="benchmarks/results/paper/20260726T165849Z/qwen3-0.6b/kv_transfer/latency_profile.json"

MODEL="${MODEL_06B}" \
PREFLIGHT_ID="${PREFLIGHT_ID}" \
TRANSFER_PROFILE="${TRANSFER_PROFILE}" \
bash benchmarks/run_preflight_suite.sh
```

脚本固定执行第 7 节的 5x routing、第 8 节的 memory-skew 和第 9 节的 load-skew，并在
全部 JSON 写完后执行以下验收。Qwen3-0.6B 主 SLA 固定为 3 秒；memory-skew 使用
24 warm-up、64 pressure 和 168 reuse requests。预验不用于论文数值，只检查：

- routing-only 的 transfer counters 为 0，5x 的 uncached prefill 明显低于 multi-GPU，
  且 per-rank request share 没有集中到单卡；
- memory-skew 至少一个 transfer-enabled 场景满足 `fg ok>0`、
  `source freed>0` 和 `offload_verified=true`，reuse-phase TTFT/throughput 相对无
  transfer 基线方向正确；
- load-skew 满足 `bg ok>0`、`source kept>0`，完整 LMPool 出现 replica/lease route，
  reuse request hit 高于实际 round-robin 结果，且 reuse TTFT 方向正确。Foreground
  offload 不在此项强制触发，因为复用阶段正在引用的 prefix blocks 不能由 move-style
  transfer 释放。

预验中断后，保留原 `PREFLIGHT_ID` 和参数并设置 `RESUME=1` 再运行同一脚本。它会跳过
已经成功写入且元数据一致的 workload；中断时尚未写完的单个 workload 从第一轮重新执行：

```bash
MODEL="${MODEL_06B}" \
PREFLIGHT_ID="replace-with-the-original-preflight-id" \
TRANSFER_PROFILE="${TRANSFER_PROFILE}" \
RESUME=1 \
bash benchmarks/run_preflight_suite.sh
```

## 3. Mode A: Automated Dual-Model Suite

这是论文结果的推荐模式。先执行第 2 节，再执行下面这一整块。统一运行器会依次对 0.6B 和
1.7B 执行三个物理 NVLink pair 的 transfer sweep、1x/3x/5x 长前缀 routing、
load-skew transfer relief 和 memory-skew capacity offload。
它会读取每个模型的 KV geometry/dtype，在三个物理 pair 上测量
1/2/4/8/16/32/64-block 延迟，将其映射为逻辑 pair 的分段 P95 latency profile，并自动写入
该模型的数据路径成本。随后 runner 使用独立 seed，在 1/2/4/8/16/32/64-block 事务上运行独立的
`multi-gpu-lmpool` calibration matrix，采集完整 dispatch-to-publish 事务并计算
pair × size-bucket P95 residual。正式评测只加载校准结果，不把正式结果反向用于 prior。
运行时按实际 plan 字节数插值，超出测量范围时用末段斜率外推；完成的 transfer 再按
pair 和 size bucket 更新 EWMA：

空载 microbenchmark 只给出随 payload 变化的数据通路项，不能直接代表完整 serving
事务。新配置使用
`T_static = residual_P95(pair,size) + 1.2 × data_path_P95(pair,bytes)`。
Residual 定义为完整 dispatch-to-publish 耗时减去干扰修正后的数据路径耗时并截断到零。
`--foreground-transfer-fixed-latency-ms` 仅作为没有 residual profile 和完整事务观测时的
fallback，默认 0；取得完整观测后可以向上或向下纠正，不再把手工 40 ms 设为永久下界。

首次正式采集前先完成第 5 节测试；测试不需要在每个 trial 前重复执行。

```bash
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export OUT="benchmarks/results/paper/${RUN_ID}"
export REPETITIONS=5
export COST_CALIBRATION_REPETITIONS=3
export COST_CALIBRATION_BATCH_BLOCKS="1 2 4 8 16 32 64"
export ROUTING_PREFIX_MULTIPLIERS="1 3 5"
export GOODPUT_SLA_MS_06B=3000
export GOODPUT_SLA_MS_17B=5000
export GOODPUT_SLA_SWEEP_MS="2000,3000,5000,10000"
export TRANSFER_FIXED_LATENCY_MS=0.0
export TRANSFER_INTERFERENCE_MULTIPLIER=1.2

bash benchmarks/run_paper_suite.sh
```

结果按 `${OUT}/qwen3-0.6b/` 和 `${OUT}/qwen3-1.7b/` 分开保存。每个模型完整结束后会写入
`SUITE_COMPLETE`，控制台也会打印 `[suite] completed qwen3-...`；两个 marker 都存在才表示
双模型矩阵完成。运行器默认从标准
Hugging Face cache 自动发现 snapshot；第 2 节导出的 `MODEL_06B` 和 `MODEL_17B` 会将其
固定为本文验证过的版本。任一模型不存在或不含 safetensors 权重时，运行器会在开始任何
GPU 实验前失败，不会联网下载，也不会用另一模型代替。开发阶段可设 `REPETITIONS=1`，
论文结果至少使用 `3`，推荐 `5`。

主 goodput SLA 在正式运行前固定：Qwen3-0.6B 使用 3 秒，Qwen3-1.7B 使用 5 秒。
二者对应不同模型规模的服务目标，不用于模型间绝对 goodput 比较。每次运行还从同一批
request completion samples 计算 2/3/5/10 秒 SLA sensitivity；该统计不会重跑推理，也不
影响 routing 或 transfer 决策。

使用本模式时，runner 已经执行第 6--9 节列出的全部 benchmark。不要随后再手工执行这些
命令，否则只是产生一套重复且目录结构不同的结果。

如果运行中断，保留原来的 `OUT` 并设置 `RESUME=1`。runner 会校验每个 JSON 的结果结构、
模型、重复次数、workload 和 E2E transfer 校准参数，并检查 transfer 的 7 个 payload、
routing 的 3 个系统配置或 E2E 的 5 个系统配置是否齐全。每个 routing 长度点都写入
`routing/prefix_Nx.{json,png,log}`，不再把某个长度点含糊地命名为 `summary`。它只跳过
结构和关键配置都一致的
artifact，只重跑缺失、不完整或配置不一致的实验。例如恢复一个未改变代码和参数的 run：

```bash
export RUN_ID="replace-with-the-original-run-id"
export OUT="benchmarks/results/paper/${RUN_ID}"
export REPETITIONS=5
export COST_CALIBRATION_REPETITIONS=3
export COST_CALIBRATION_BATCH_BLOCKS="1 2 4 8 16 32 64"
export ROUTING_PREFIX_MULTIPLIERS="1 3 5"
export GOODPUT_SLA_MS_06B=3000
export GOODPUT_SLA_MS_17B=5000
export GOODPUT_SLA_SWEEP_MS="2000,3000,5000,10000"
export TRANSFER_FIXED_LATENCY_MS=0.0
export TRANSFER_INTERFERENCE_MULTIPLIER=1.2
export RESUME=1

bash benchmarks/run_paper_suite.sh
```

不要在修改 workload、成本参数或代码后向旧目录拼接结果；应使用新的 `RUN_ID` 对两个模型
重跑，保证同一结果目录内配置一致。`RESUME=1` 只用于恢复同一代码版本、同一参数和同一
实验批次的中断运行。它按 artifact 粒度恢复：已经完整写出的 microbenchmark pair、
transaction-calibration bucket、routing multiplier 或 E2E workload 会被跳过；正在运行但
尚未写出完整 JSON 的单个 artifact 会从该 artifact 的第一轮重新开始。恢复时必须保留原
`RUN_ID`、`OUT`、`REPETITIONS`、`COST_CALIBRATION_REPETITIONS`、
`COST_CALIBRATION_BATCH_BLOCKS`、`ROUTING_PREFIX_MULTIPLIERS`、模型路径和拓扑变量。
还必须保留 `GOODPUT_SLA_MS_06B`、`GOODPUT_SLA_MS_17B` 和
`GOODPUT_SLA_SWEEP_MS`。

KV transfer microbenchmark 为每个 payload case 使用唯一 FileStore rendezvous，不依赖临时
TCP 端口。若某个 worker 仍异常退出，父进程会立即报告 exit code，而不会等待完整超时。

## 4. Mode B: Manual Single-Model Runs

仅在调试某个 workload 或只重跑一个模型时使用本模式。先执行第 2 节，然后从下面两组
`MODEL`/`MODEL_LABEL` 中只选择一组；不要同时设置两组：

```bash
# Qwen3-0.6B：选择这一组
export MODEL="${MODEL_06B}"
export MODEL_LABEL=qwen3-0.6b
export GOODPUT_SLA_MS=3000

# Qwen3-1.7B：测试 1.7B 时改为下面三行
# export MODEL="${MODEL_17B}"
# export MODEL_LABEL=qwen3-1.7b
# export GOODPUT_SLA_MS=5000

export CUDA_VISIBLE_DEVICES="${GPU_SET}"
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export OUT="benchmarks/results/paper/${RUN_ID}/${MODEL_LABEL}"
export REPETITIONS=5
export GOODPUT_SLA_SWEEP_MS="2000,3000,5000,10000"
export TRANSFER_PROFILE="${OUT}/kv_transfer/latency_profile.json"
export DATA_PATH_PROFILE="${OUT}/kv_transfer/data_path_profile.json"
mkdir -p "${OUT}"/{environment,kv_transfer,routing,memory_skew,load_skew}

nvidia-smi -L | tee "${OUT}/environment/gpus.txt"
nvidia-smi topo -m | tee "${OUT}/environment/topology.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,power.limit \
  --format=csv | tee "${OUT}/environment/gpu_inventory.csv"
git rev-parse HEAD | tee "${OUT}/environment/git_revision.txt"
git status --short | tee "${OUT}/environment/git_status.txt"
```

完成设置后，只执行第 6--9 节中需要的手工命令。若要切换模型，重新执行本节并生成新的
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
  --output "${DATA_PATH_PROFILE}"

# 独立 transaction calibration 只运行完整 LMPool，不作为论文性能结果。分别限制事务为
# 1/2/4/8/16/32/64 blocks，避免用大事务估计小 foreground/background plan。
transaction_calibrations=()
for calibration_blocks in 1 2 4 8 16 32 64; do
  calibration_candidate_blocks=4
  if (( calibration_blocks < calibration_candidate_blocks )); then
    calibration_candidate_blocks="${calibration_blocks}"
  fi
  calibration_json="${OUT}/kv_transfer/transaction_calibration_${calibration_blocks}.json"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_e2e.py \
    --model-name-or-path "${MODEL}" \
    --dtype auto \
    --world-size "${WORLD_SIZE}" \
    --workload transfer-calibration \
    --scenarios multi-gpu-lmpool \
    --calibration-prefix-groups 66 \
    --calibration-warmup-prompts 66 \
    --num-prompts 132 \
    --prompt-repeat 8 \
    --max-tokens 8 \
    --ignore-eos \
    --seed "$((1000 + calibration_blocks))" \
    --repetitions 3 \
    --nvlink-pairs "${NVLINK_PAIRS}" \
    --submit-window 66 \
    --kv-block-budget 128 \
    --gpu-memory-utilization 0.5 \
    --goodput-e2e-sla-ms 10000 \
    --background-copy-max-blocks "${calibration_candidate_blocks}" \
    --background-copy-batch-max-blocks "${calibration_blocks}" \
    --background-copy-batch-max-candidates 32 \
    --background-copy-hot-threshold 1 \
    --background-copy-cooldown-s 0.1 \
    --background-copy-expected-reuses 4 \
    --foreground-transfer-min-benefit-ratio 0.0 \
    --foreground-transfer-profile-json "${DATA_PATH_PROFILE}" \
    --foreground-transfer-fixed-latency-ms 0.0 \
    --foreground-transfer-interference-multiplier 1.2 \
    --kv-transfer-prewarm-blocks 4 \
    --output-json "${calibration_json}" \
    --output-figure "${OUT}/kv_transfer/transaction_calibration_${calibration_blocks}.png"
  transaction_calibrations+=("${calibration_json}")
done

uv run python benchmarks/build_transfer_profile.py \
  --inputs "${transfer_inputs[@]}" \
  --logical-pairs "${NVLINK_PAIRS}" \
  --latency-metric p95_latency_ms \
  --transaction-inputs "${transaction_calibrations[@]}" \
  --transaction-scenario multi-gpu-lmpool \
  --transaction-percentile 0.95 \
  --output "${TRANSFER_PROFILE}"
```

所有 payload 的 `data_validation` 必须为 `passed`。`data_path_profile.json` 保留每个物理
pair 的来源、模型 KV geometry、原始延迟和单调化后的决策延迟。E2E loader 会校验逻辑 pair
以及每 block 字节数，防止 0.6B/1.7B、FP16/BF16 或不同 block size 的 profile 混用。
线上 batch 并非固定档位：foreground plan 等于实际 shortage，background plan 可合并多条
候选；最终 `latency_profile.json` 同时包含数据路径曲线和独立事务 residual P95，运行时再由
pair x size-bucket EWMA 修正。Calibration 与正式评测必须使用不同 seed。

## 7. KV-Aware Routing

该入口严格关闭 foreground/background transfer，只验证 cache locality 与 load-aware routing。
主结果是 1x/3x/5x 前缀长度 sweep；`--max-tokens 8` 只减少 decode 干扰，真正的自变量是
`--prompt-repeat`。下面给出 5x 主档，1x/3x 分别把 repeat/model length 改为
`16/2048` 和 `48/6144`。需要验证极长上下文趋势时，可额外运行 10x，即
`prompt-repeat=160`、`max-model-length=20480` 和
`max-num-batched-tokens=20480`；两个 Qwen3 模型的 40960-token position limit 可以容纳
该点，但它不默认进入整套实验，以免显著增加两模型五次重复的总耗时。

```bash
CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_kv_routing.py \
  --model-name-or-path "${MODEL}" \
  --dtype auto \
  --world-size "${WORLD_SIZE}" \
  --num-prompts 192 \
  --prompt-repeat 80 \
  --max-tokens 8 \
  --max-model-length 10240 \
  --max-num-batched-tokens 10240 \
  --temperature 0.6 \
  --ignore-eos \
  --seed 0 \
  --repetitions "${REPETITIONS}" \
  --locality-prefix-groups 16 \
  --nvlink-pairs "${NVLINK_PAIRS}" \
  --submit-window 16 \
  --kv-block-budget 384 \
  --gpu-memory-utilization 0.7 \
  --goodput-e2e-sla-ms "${GOODPUT_SLA_MS}" \
  --goodput-e2e-sla-sweep-ms "${GOODPUT_SLA_SWEEP_MS}" \
  --output-json "${OUT}/routing/prefix_5x.json" \
  --output-figure "${OUT}/routing/prefix_5x.png" \
  2>&1 | tee "${OUT}/routing/prefix_5x.log"
```

验收条件：routing-only 的 transfer counters 必须为 0；相对 `multi-gpu`，应提高 DP token
reuse，并改善 throughput、TTFT 或尾延迟，且 per-rank request share 不应重新集中到单卡。

## 8. Memory Skew: Capacity Relief by Foreground Transfer

该实验关闭 background copy，只验证容量不足时的 foreground move-style transfer。Warm-up
结束后，ingress 会在 pressure phase 前同步发布剩余 reuse 的精确 prefix-demand snapshot，
使 foreground admission 按已知未来需求估值，而不是只按 warm-up 历史访问次数猜测。64-block
budget 是所有五个配置共用的受限容量，不允许单独缩小 transfer 场景预算。12 个热点各预热
2 次，使每个 source 持有 4 条约 3.8K-token 长前缀；64 条互不共享的 pressure 请求使 source
工作集超过 budget 并触发容量释放。余下 168 条请求专门测量 offload 后的 prefix reuse，
在不增加冷 pressure 的情况下摊薄 transfer 固定开销。

```bash
CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_e2e.py \
  --model-name-or-path "${MODEL}" \
  --dtype auto \
  --world-size "${WORLD_SIZE}" \
  --workload memory-skew \
  --memory-skew-prefix-groups 12 \
  --memory-skew-warmup-prompts 24 \
  --memory-skew-pressure-prompts 64 \
  --num-prompts 256 \
  --prompt-repeat 32 \
  --max-tokens 16 \
  --temperature 0.6 \
  --ignore-eos \
  --seed 0 \
  --repetitions "${REPETITIONS}" \
  --nvlink-pairs "${NVLINK_PAIRS}" \
  --submit-window 32 \
  --kv-block-budget 64 \
  --gpu-memory-utilization 0.5 \
  --goodput-e2e-sla-ms "${GOODPUT_SLA_MS}" \
  --goodput-e2e-sla-sweep-ms "${GOODPUT_SLA_SWEEP_MS}" \
  --disable-background-copy \
  --foreground-transfer-min-benefit-ratio 1.1 \
  --foreground-transfer-profile-json "${TRANSFER_PROFILE}" \
  --foreground-transfer-fixed-latency-ms 0.0 \
  --foreground-transfer-interference-multiplier 1.2 \
  --kv-transfer-prewarm-blocks 4 \
  --output-json "${OUT}/memory_skew/summary.json" \
  --output-figure "${OUT}/memory_skew/summary.png" \
  2>&1 | tee "${OUT}/memory_skew/run.log"
```

运行前必须先由第 6 节生成 `TRANSFER_PROFILE`。验收时要求
`multi-gpu-kv-transfer` 或 `multi-gpu-lmpool` 的 `offload_verified=true`、
`transfer_release_count>0` 且 `fg ok>0`。随后比较 preemption、throughput、TTFT 和 P90 E2E。
只有发送 block 而没有释放源端 block 是复制，不构成 offloading 证据。

## 9. Load Skew: Background Transfer for Load Relief

该实验先用 48 个请求把 24 条约 5.7K-token 长前缀固定在三个 source rank，每组预热两次，
再一次最多提交 64 个请求，形成 144-request 热点 burst。普通 round-robin 对每组重复 6 次，
其中 12 个组落在 owner、12 个组落在 direct NVLink partner；未 transfer 的 partner 因而
必须完成 12 次长前缀冷 prefill。Reuse 前的 phase boundary 将未来前缀需求交给控制面，
background path 可按 pair 批量复制；192-block budget 可容纳每个 source 或 partner 的
8 条长前缀。复用请求会 pin 已放置的 prefix，因而本实验不要求 foreground move 释放这些
活跃 blocks；foreground capacity offload 由第 8 节单独验证。Routing-only 始终关闭
transfer，因而是完整 LMPool 的直接消融基线。输出限制为 8 token，避免 decode
掩盖 transfer 对 TTFT 和 serving throughput 的影响。

```bash
CUDA_VISIBLE_DEVICES="${GPU_SET}" uv run python benchmarks/benchmark_e2e.py \
  --model-name-or-path "${MODEL}" \
  --dtype auto \
  --world-size "${WORLD_SIZE}" \
  --workload load-skew \
  --load-skew-prefix-groups 24 \
  --load-skew-warmup-prompts 48 \
  --num-prompts 192 \
  --prompt-repeat 48 \
  --max-tokens 8 \
  --temperature 0.6 \
  --ignore-eos \
  --seed 0 \
  --repetitions "${REPETITIONS}" \
  --nvlink-pairs "${NVLINK_PAIRS}" \
  --submit-window 64 \
  --kv-block-budget 192 \
  --gpu-memory-utilization 0.7 \
  --goodput-e2e-sla-ms "${GOODPUT_SLA_MS}" \
  --goodput-e2e-sla-sweep-ms "${GOODPUT_SLA_SWEEP_MS}" \
  --background-copy-max-blocks 24 \
  --background-copy-batch-max-blocks 48 \
  --background-copy-batch-max-candidates 4 \
  --background-copy-hot-threshold 2 \
  --background-copy-min-load-skew 2 \
  --background-copy-expected-reuses 8 \
  --background-copy-cooldown-s 0.1 \
  --foreground-transfer-min-benefit-ratio 1.1 \
  --foreground-transfer-profile-json "${TRANSFER_PROFILE}" \
  --foreground-transfer-fixed-latency-ms 0.0 \
  --foreground-transfer-interference-multiplier 1.2 \
  --kv-transfer-prewarm-blocks 4 \
  --output-json "${OUT}/load_skew/summary.json" \
  --output-figure "${OUT}/load_skew/summary.png" \
  2>&1 | tee "${OUT}/load_skew/run.log"
```

## 10. Result Acceptance

每个 JSON/日志至少检查：所有请求完成、每个场景 repetitions 数正确、无 worker/control
timeout、无 NCCL watchdog 错误、实际每-rank KV capacity 等于请求 budget。JSON 顶层为
`metadata` 和 `results`：前者保存精确命令、Git revision、模型结构/dtype 和解析后的配置；
后者保存聚合结果及每次 `trial_results`。论文表格报告 mean 和 95% CI，并在附录保留 sample
standard deviation；延迟同时报告 mean、P90、P95。TPOT 是纯 decode 指标，按每个请求
`(completion timestamp - first-token timestamp) / (output tokens - 1)` 计算，单 token
输出不进入 TPOT 分布；不要把旧版 `E2E / output tokens` 字段解释为 TPOT。95% CI 的样本是
完整场景的独立 repetition，半宽为
`t(0.975, repetitions - 1) * sample_std / sqrt(repetitions)`。P90 E2E 先在每次 trial 内
计算，再对 trial-level P90 计算 CI。出现 transfer/rebalance failure 时必须结合 failure
reason 解释，不能只比较总吞吐。

Routing 结果必须同时画出 1x/3x/5x 三档，不能只选择最有利的 5x 点。三档保持请求数、前缀
组、提交窗口、输出长度和 KV budget 不变；验收重点是 routing 相对 round-robin 的 uncached
prefill token 减少量能否随可复用前缀增长，并在 3x/5x 转化为 TTFT、E2E 或 throughput 收益。
若额外测 10x，也必须作为同一 sweep 的敏感性点完整报告，不能替代默认三档。

Memory-skew 首先检查 workload 是否成立：每个 source rank 必须收到 warm-up 和 pressure
请求，partner 必须有目标空间。然后检查 `fg ok>0`、`source freed>0` 和
`offload_verified=true`。只有满足这些机制条件后，才比较 transfer-only 对 multi-gpu、完整
LMPool 对 routing-only 的 throughput 与 latency；若机制未触发，该批次只能诊断成本门限或
容量构造，不能用于宣称 transfer 无效或有效。

Load-skew 首先检查 48/144 的 phase 计数和 source/partner 的 per-rank 分布。机制验收要求
`multi-gpu-kv-transfer` 或 `multi-gpu-lmpool` 的 `background_copy_success>0`、
`transfer_copy_count>0`；完整 LMPool 还应出现 `placement_lease_route_count>0` 或
replica-copy route，reuse request hit 应高于同批次 round-robin。只有这些条件
成立后，才能把相对 routing-only 的 throughput、TTFT 和 P90 E2E 差异归因于 transfer。
