"""
运行方式示例：

1. 使用默认参数：
   CUDA_VISIBLE_DEVICES=0,2 UV_CACHE_DIR=/tmp/uvcache uv run python benchmarks/benchmark_e2e.py

2. 显式指定参数：
    CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 UV_CACHE_DIR=/tmp/uvcache \
    uv run python benchmarks/benchmark_e2e.py \
    --num-prompts 192 \
    --prompt-repeat 48 \
    --max-tokens 8 \
    --temperature 0.6 \
    --ignore-eos \
    --seed 0 \
    --repetitions 3 \
    --workload load-skew \
    --locality-prefix-groups 16 \
    --load-skew-prefix-groups 24 \
    --load-skew-warmup-prompts 48 \
    --load-skew-hot-groups 24 \
    --load-skew-hot-share 0.8 \
    --world-size 6 \
    --nvlink-pairs "0,1;2,3;4,5" \
    --kv-block-budget 192 \
    --gpu-memory-utilization 0.70 \
    --submit-window 336 \
    --background-copy-max-blocks 24 \
    --background-copy-batch-max-blocks 48 \
    --background-copy-batch-max-candidates 4 \
    --background-copy-cooldown-s 2.0 \
    --background-copy-hot-threshold 3 \
    --route-load-weight 0.03 \
    --route-load-bypass-threshold 256 \
    --route-prefill-cost-weight 1.0 \
    --route-reclaim-cost-weight 0.5 \
    --foreground-transfer-cost-weight 1.0 \
    --foreground-transfer-min-benefit-ratio 1.5 \
    --foreground-transfer-bandwidth-gib-s 3.5 \
    --foreground-transfer-profile-json ./benchmarks/results/kv_transfer/latency_profile.json \
    --foreground-transfer-fixed-latency-ms 0.0 \
    --foreground-transfer-interference-multiplier 1.2 \
    --scenarios single-gpu,multi-gpu,multi-gpu-kv-routing,multi-gpu-kv-transfer,multi-gpu-lmpool \
    --foreground-prefill-token-time-ms 0.02 \
    --foreground-future-reuse-discount 0.5 \
    --route-cache-queue-slack 256 \
    --goodput-e2e-sla-ms 3000 \
    --goodput-e2e-sla-sweep-ms 2000,3000,5000,10000 \
    --output-json ./benchmarks/results/e2e.json \
    --output-figure ./benchmarks/results/e2e.png

参数说明：
1. `--num-prompts`：
  本次压测总共生成多少条请求。脚本会按 workload 构造共享前缀组和不同后缀；
  值越大，并发压力越高，统计结果也更稳定。

2. `--prompt-repeat`：
  共享前缀重复多少次。值越大，公共前缀越长，越容易观察 prefix cache / 路由收益。

3. `--max-tokens`：
  每条请求最多生成多少个输出 token。值越大，decode 阶段占比越高，也更容易触发 transfer 压力。

4. `--temperature`：
  采样温度。benchmark 默认主要看系统性能，通常保持固定值即可，不建议在不同实验间频繁改动。

5. `--ignore-eos` / `--no-ignore-eos`：
  默认启用 `--ignore-eos`，每条请求固定生成 `--max-tokens` 个 token，保证不同场景执行相同
  decode 工作量。只有需要模拟真实 EOS 提前结束时才使用 `--no-ignore-eos`。

6. `--seed`：
  data-plane 随机种子基值；rank `r` 使用 `seed + r`，用于复现实验。

7. `--repetitions`：
  每个场景完整重复运行次数，默认 1。论文实验建议至少设为 3；多次运行时输出 mean/std，
  JSON 额外保存 throughput、goodput、TTFT 和 E2E 的标准差。

8. `--workload`：
  workload 类型。`locality` 用多组长共享前缀验证 KVCache-aware routing；
  `load-skew` 先在每个 NVLink pair 的 source 预热多条长前缀，再以高 submit window
  提交热点复用 burst，用于观察 forecast-driven background copy 和 replica-aware routing；
  `memory-skew` 依次执行热点前缀预热、源端一次性前缀施压、热点前缀复用三个阶段，
  用于验证 foreground transfer 是否既释放源端容量，又在 NVLink 伙伴保留可复用前缀。
  `capacity-offload` 是 `memory-skew` 的兼容别名，不再作为独立论文 workload；两者都只有在
  `offload_verified=true` 且 `transfer_release_count>0` 时才能证明 transfer 产生容量缓解。
  `transfer-calibration` 是 runner 内部使用的两阶段 transaction calibration trace，不是
  论文 workload，也不用于报告 serving 性能。
  对 topology-blind baseline 和 transfer-only，复用阶段会确定性地跨到对应 NVLink partner：
  baseline 必须在 partner 重算，只有已经完成 transfer 的前缀才能直接命中。

9. `--locality-prefix-groups`：
  `locality` workload 中不同长共享前缀的组数，默认 16。每组请求数保持均衡，并按 `--seed`
  打乱提交顺序，避免前缀组编号与 round-robin rank 周期重合。组数必须在 1 和
  `--num-prompts` 之间；组数越多，未启用 routing 时跨 GPU 重复缓存和 cache churn 越明显。

10. `--load-skew-prefix-groups`：
  `load-skew` 中固定在 NVLink source rank 上预热的长热点前缀组数量。论文配置使用 24 组，
  使三个 NVLink pair 的每个 source 持有 8 组前缀，而 partner 在 reuse burst 前保持空闲。
  warm-up 映射按 NVLink pair 周期交错以均衡初始 owner；reuse 顺序则按 `--seed` 打乱，
  刻意消除 prefix group 与 round-robin rank 周期的对齐，使 baseline 必须面对真实的
  跨 rank 重复 prefill，而 LMPool 可以利用 source/replica placement。

11. `--load-skew-hot-groups` / `--load-skew-hot-share`：
  reuse burst 中的高频前缀组数量和请求比例。`0` 表示所有
  `--load-skew-prefix-groups` 都是热点；其余请求使用每次都不同的一次性 cold prefix。
  论文配置使用 24 个热点和 0.8，使 background copy 面对明确的高收益对象，同时保留
  不可复用的真实长尾。

12. `--load-skew-warmup-prompts`：
  `load-skew` 的预热请求数。默认 0 使用总请求数的四分之一且至少覆盖每个热点一次；
  论文配置使用 48 个 warm-up 和 336 个 reuse 请求；每个热点先获得 2 次真实访问，
  reuse 中 80% 的请求访问热点，其余请求使用一次性 cold prefix。

13. `--memory-skew-prefix-groups`：
  `memory-skew` workload 中需要跨阶段保留的长会话前缀数量。每组在 warm-up 中建立一条
  可迁移 session chain 和一条较短的 routing anchor。默认 0 自动选择不超过 15 的最大奇数，
  并保证两条 warm-up 请求、一次 trigger 和一次精确 reuse 都能覆盖每组。

14. `--memory-skew-warmup-prompts`：
  `memory-skew` 中建立长 session KV 的请求数。每个 session 内含短 anchor，因此每组至少一条
  warm-up 请求；worker rank 仍由当前场景的普通分发策略决定。

15. `--memory-skew-pressure-prompts`：
  `memory-skew` 中只复用短 anchor、再追加唯一短尾的压力请求数。它们占满 source KV，
  但不引用或 pin 住长 session chain。pressure 会完整 drain，之后才进入 trigger。

16. `--memory-skew-trigger-prompts`：
  pressure 完成后提交的长 anchor continuation 数。默认每个 prefix group 一条。它们通过普通
  路由在热门 anchor owner 处请求额外 KV blocks，触发 foreground transfer。

17. `--memory-skew-pressure-hot-groups` / `--memory-skew-pressure-hot-share`：
  接收大部分 pressure 请求的 anchor group 数量及其请求比例。默认两个 group 承担 80% pressure，
  以生成由 KV-aware routing 自然导致的容量倾斜，不指定任何 worker rank。

18. `--memory-skew-anchor-share`：
  每个 session 中共享 routing anchor 所占的文本比例，默认 0.375。较长 anchor 让绕过 owner
  需要重算足够多的 prefix；其余 session suffix 仍可作为 foreground transfer 候选。

17. `--calibration-prefix-groups`：
  仅供 `transfer-calibration` 使用的不同前缀组数。每个组必须同时出现在 source-build 和
  partner-reuse 阶段，用于产生完整 transfer transaction observation。

18. `--calibration-warmup-prompts`：
  `transfer-calibration` 中固定在 NVLink source 侧建立 KV 的请求数。其余请求固定到对应
  partner；该参数只用于生成成本 profile，不进入论文 serving 性能对比。

18. `--output-json`：
  将各场景统计结果导出到指定 JSON 文件。脚本会自动创建父目录，并在成功后打印
  `saved json: ...`。

19. `--model-name-or-path`：
  指定要测试的模型名称或本地路径。默认使用脚本里的 Qwen 配置，对应模型结构也基于这份配置。

20. `--dtype`：
  模型权重与 KV cache 的运行 dtype。`auto` 从模型 `config.json` 读取；也可显式指定
  `float16`、`bfloat16` 或 `float32`。构建和加载 transfer profile 时 dtype 必须一致。

21. `--nvlink-pairs`：
  手动指定 NVLink 拓扑，格式如 `0,1` 或 `0,1;2,3`。这里使用的是
  `CUDA_VISIBLE_DEVICES` 之后的逻辑 GPU 编号，不是物理 GPU 编号。如果不想手动写，
  可以传空字符串，让底层逻辑尝试解析 `nvidia-smi topo -m`。命令行里包含分号时必须加引号，
  例如 `--nvlink-pairs "0,2;1,3;4,5;6,7"`。

22. `--world-size`：
  多卡场景启动多少个 data-plane worker。默认 2；八卡实验需要显式传 `--world-size 8`。
  该值不能超过 `CUDA_VISIBLE_DEVICES` 暴露出的 GPU 数。

23. `--kv-block-budget`：
  每个 rank 请求使用的 KV block 数。五个场景必须使用同一个值，避免把容量差异误判成
  routing / transfer 收益。显式设置后采用严格语义：worker 实际容量不足时会在提交请求前报错，
  不再静默缩小预算。

24. `--gpu-memory-utilization`：
  ModelRunner 推导可用 KV cache 容量时可使用的空闲显存比例，范围 `(0, 1]`，benchmark
  默认 0.20。最终分配仍受 `--kv-block-budget` 限制；提高该值只是让显式 block budget
  能够实现，不会越过该上限。

25. `--goodput-e2e-sla-ms`：
  goodput 的端到端延迟门槛，单位毫秒。只有在这个 SLA 内完成的请求，其输出 token
  才计入 goodput。因此表里的 goodput 单位是 tokens/s，不是 requests/s。

26. `--goodput-e2e-sla-sweep-ms`：
  用逗号分隔的多个端到端 SLA 门槛。脚本使用同一批请求完成时间计算每个门槛下的
  goodput，不会重新执行推理；JSON 同时保存各门槛的均值和 95% 置信区间。

27. `--skip-pool`：
  跳过 `multi-gpu-lmpool` 场景，只跑基线、routing 和 transfer。

28. `--output-figure`：
  将五种场景的核心指标画成一张图表图片并保存到指定路径。脚本会自动创建父目录，
  使用无显示环境可用的 Matplotlib Agg 后端，并在成功后打印 `saved figure: ...`。

29. `--submit-window`：
  benchmark 中允许同时在途的请求数。值越大越接近一次性高并发提交；值越小越容易让前面请求先完成
  prefill 并上报全局页表，从而观察在线 prefix reuse。设为 0 或负数表示一次性提交全部请求。
  如果要验证 prefix hit 是否生效，建议先用 4 ~ 8；如果要模拟 burst 流量，可以设为 0 或 -1。

30. `--disable-background-copy`：
  关闭后台 speculative copy-style transfer。默认开启，用于把热点 prefix block 异步复制到 NVLink
  伙伴，服务后续请求；关闭后只保留因当前请求容量不足而同步触发的 foreground transfer。

31. `--background-copy-max-blocks`：
  每条后台候选前缀链最多贡献多少个 prefix block。相同方向、相同 NVLink pair 的多条候选
  会在内部合并成一个有界 plan，以摊薄控制协议和 payload 启动开销。

32. `--background-copy-batch-max-blocks`：
  同一 `src -> dst` 事务合并后的总 block 上限。它限制一次 packed transfer 的 payload，
  不改变单条候选链的 `--background-copy-max-blocks` 上限。

33. `--background-copy-batch-max-candidates`：
  一个 pair 在构造一次 packed transfer 时最多检查的候选链数。该参数与总 block 上限共同
  约束事务大小，主要用于成本校准和控制面开销保护。

34. `--background-copy-cooldown-s`：
  同一个 prefix 在同一组 `src -> dst` GPU 之间再次触发后台 copy 的最短间隔，单位秒。
  值越大越保守，值越小越容易在高并发下产生更多 transfer。验证后台 copy 收益时可尝试 0.5。

35. `--background-copy-hot-threshold`：
  最大热点前缀链中每个 block 至少需要达到的 worker 上报访问次数。值越大越保守，能减少
  无效 copy；该统计来自真实数据面访问，不再使用路由请求次数代替。

36. `--background-copy-min-load-skew`：
  route-originated 候选发现要求 prefix owner 与 NVLink partner 至少相差多少队列压力，默认 2；
  phase 边界的 ingress forecast 使用已观测的数据放置偏斜，不受该瞬时负载差限制。

37. `--background-copy-expected-reuses`：
  预测未来复用次数的保守上限，默认 4。实际预测来自 ingress 尚未提交请求的逐前缀计数；
  没有 forecast 时才使用折扣后的 worker 历史访问次数，不再固定假设一定有 4 次复用。

38. `--route-load-weight`：
  旧 prefix score 中 token-aware load 的 tie-break 权重。主路由决策现在使用统一预计完成成本；
  该参数只在成本相同时参与稳定排序，通常保持默认值。

39. `--route-decode-token-weight`：
  一个预计 decode token 在路由负载快照中的权重，默认 8，用于避免长输出请求集中到单一 owner。

40. `--route-owner-spill-sequence-skew`：
  prefix owner 比 NVLink partner 多出的序列压力达到该值时允许 pair 内 spill，默认 2。

41. `--route-owner-spill-max-extra-cost`：
  pair spill 相比留在 owner 最多允许增加的 token-equivalent 重算成本，默认 2048。

42. `--route-load-bypass-threshold`：
  冷目标的预计总成本必须比 prefix owner 至少低多少 token-equivalent cost，才允许绕过
  owner。值越小越激进，越容易牺牲 locality 换并行度。

43. `--route-prefill-cost-weight`：
  缺失 prefix token 的重算成本权重。默认 1.0，使一个缺失 token 与一个 waiting token
  使用相同成本单位；增大后路由更偏向已有 prefix 的 owner。

44. `--route-reclaim-cost-weight`：
  使用 reclaimable capacity 时，每个待回收 block 按 `block_size * weight` 计入的附加成本。
  默认 0.5，用于反映回收元数据操作及未来 cache miss 风险。

45. `--foreground-transfer-cost-weight`：
  对时间模型算出的 transfer 成本施加的整体倍率，默认 1.0。大于 1 会更保守；通常保持
  1.0，优先校准下面的带宽、固定延迟和干扰系数。

46. `--foreground-transfer-min-benefit-ratio`：
  foreground transfer 预计节省的 prefill 毫秒数与预计 transfer 毫秒数的最小比值。
  默认 1.5；未达到门槛时跳过 transfer，直接使用本地回收。

47. `--foreground-transfer-bandwidth-gib-s`：
  未提供 size-aware profile 时的兼容回退带宽，单位 GiB/s，默认 3.5。正式实验应优先使用
  下一项 profile，不再用单一带宽代表所有 plan 大小。

48. `--foreground-transfer-profile-json`：
  `build_transfer_profile.py` 生成的分段延迟 profile。它按逻辑 NVLink pair 保存
  1/2/4/8/16/32/64-block 的实测延迟；运行时按实际 payload 插值，并严格校验模型、dtype
  对应的每 block 字节数。若构建时提供独立 E2E calibration JSON，同一文件还包含
  pair × size-bucket 的 dispatch-to-publish P95 transaction residual。未提供数据路径
  profile 时才回退到上一项标量带宽。

49. `--foreground-transfer-fixed-latency-ms`：
  没有 transaction residual profile 时的冷启动 fallback，单位毫秒，默认 0.0。它不再是
  永久下界：取得一次完整 dispatch-to-publish 观测后，在线 EWMA 可以向上或向下纠正该值。
  正式实验应通过独立 calibration run 生成 P95 residual profile，而不是手工填写 40 ms。

50. `--foreground-transfer-interference-multiplier`：
  对空载 transfer microbenchmark 的分段数据通路 P95 施加的加载态干扰倍率，
  默认 1.2，且不能小于 1。它只表示随 payload 变化的模型执行竞争、打包和解包干扰；
  完整 serving 事务中不随 payload 线性变化的部分由 residual profile 表示。

51. `--scenarios`：
  逗号分隔的场景子集。默认运行五个场景；成本校准时可设置
  `--scenarios multi-gpu-lmpool`，只运行完整系统，避免把基线重复执行。

52. `--foreground-prefill-token-time-ms`：
  重算一个未缓存 prompt token 的预计耗时，单位毫秒，默认 0.02。该值应由目标模型的
  prefill 统计校准。

53. `--foreground-future-reuse-discount`：
  将历史叶前缀访问次数折算成未来复用次数的折扣，范围 `[0, 1]`，默认 0.5。成本模型不会
  再把前缀链上每个 block 的访问次数相加，从而避免将一条请求重复计算多次。Ingress 已经
  发布尚未提交请求的精确 prefix-demand snapshot 时，foreground admission 直接使用该计数，
  不再对它应用历史折扣。

54. `--kv-transfer-prewarm-blocks`：
  serving 开始前，每个 NVLink pair 使用真实 KV 形状预热的 block 数，默认 2。预热会循环
  使用与线上一致的单个 all-layer 连续 payload，并把测得的 pair-specific 成本送入控制面，
  但不计入 throughput 或 latency。

55. `--route-cache-queue-slack`：
  route cache 命中时允许 cached owner 相比最低成本候选多出的 token-equivalent cost。
  值越小，缓存路由越容易被负载不均打破。

说明：
1. 建议显式设置 CUDA_VISIBLE_DEVICES，避免在共享机器上误用其他 GPU。
2. 如果物理 GPU 0 和 2 之间有 NVLink，可以使用 `CUDA_VISIBLE_DEVICES=0,2`。
   但脚本内部看到的是重映射后的逻辑 GPU `0,1`，因此 `--nvlink-pairs` 应写成 `0,1`，而不是 `0,2`。
3. `multi-gpu`，`multi-gpu-kv-transfer` 场景当前采用 round-robin 分发。
4. 表里的 prefix hit 是 worker 在 prefill 时实际观察到的本地 prefix cache 命中率，
   round-robin 基线也会统计，因此可横向对比。它不是控制面路由命中率。
5. `multi-gpu-lmpool` 需要至少 2 张可见 CUDA GPU。
6. 所有场景仍应使用相同的 `--kv-block-budget`；使用 `memory-skew` 验证源端容量释放。
   `capacity-offload` 只是同一 trace 的兼容别名。不要通过缩小某一个场景的容量制造
   不公平结果。`transfer-calibration` 仅生成成本 profile，不属于上述公平性比较。
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lmpool.engine.block_manager import BlockManager
from lmpool.engine.llm_engine import LLMEngine
from lmpool.engine.sequence import Sequence
from lmpool.sampling_parameters import SamplingParams

try:
    from .benchmark_utils import build_run_metadata, resolve_model_runtime_config
    from .transfer_profile import load_transfer_latency_profile
except ImportError:
    from benchmark_utils import build_run_metadata, resolve_model_runtime_config
    from transfer_profile import load_transfer_latency_profile


def prepare_benchmark_rendezvous(config: dict) -> tuple[dict, Path | None]:
    """Give each local benchmark trial an independent rendezvous store."""
    trial_config = dict(config)
    if trial_config.get("distributed_init_method"):
        return trial_config, None
    rendezvous_path = Path("/tmp") / (
        f"lmpool-rendezvous-{os.getpid()}-{uuid.uuid4().hex}"
    )
    trial_config["distributed_init_method"] = rendezvous_path.resolve().as_uri()
    return trial_config, rendezvous_path


MODEL_CONFIG = {
    # 这份配置尽量贴近仓库当前默认模型，方便 benchmark 与主流程复用同一套模型结构
    "max_num_sequences": 64,
    "max_num_batched_tokens": 4096,
    "max_cached_blocks": 1024,
    "block_size": 256,
    "model_name_or_path": "Qwen/Qwen3-0.6B",
    "enforce_eager": True,
    "vocab_size": 151936,
    "hidden_size": 1024,
    "num_heads": 16,
    "head_dim": 128,
    "num_kv_heads": 8,
    "intermediate_size": 3072,
    "num_layers": 28,
    "tie_word_embeddings": True,
    "base": 1000000,
    "rms_norm_epsilon": 1e-6,
    "qkv_bias": False,
    "scale": 1,
    "max_position": 32768,
    "ffn_bias": False,
    "max_num_batch_tokens": 4096,
    "max_model_length": 2048,
    # This fraction is used only to derive KV capacity; max_cached_blocks still
    # caps the actual allocation. Five percent yielded only 14 Qwen3 blocks on
    # the evaluation GPUs, silently invalidating explicit 64-block experiments.
    "gpu_memory_utilization": 0.20,
    "eos": 151645,
    "log_level": "ERROR",
    "log_timing": False,
    "log_decode_every_n": 16,
    # benchmark 启动阶段会经历权重加载、warmup、KV cache 分配，默认 3 秒 heartbeat 超时过于激进
    "heartbeat_interval": 1.0,
    "heartbeat_timeout": 3600.0,
    "distributed_timeout_s": 1800.0,
    "worker_join_timeout": 30.0,
    "route_prefix_hit_weight": 8.0,
    "route_queue_pressure_weight": 1.0,
    "route_free_block_weight": 0.05,
    "route_load_weight": 0.03,
    "route_waiting_token_weight": 1.0,
    "route_running_token_weight": 0.25,
    "route_running_sequence_weight": 32.0,
    "route_load_bypass_threshold": 256.0,
    "route_decode_token_weight": 8.0,
    "route_owner_spill_sequence_skew": 2.0,
    "route_owner_spill_max_extra_cost": 2048.0,
    "route_prefill_cost_weight": 1.0,
    "route_reclaim_cost_weight": 0.5,
    "route_cache_queue_slack": 256.0,
    "enable_foreground_rebalance": True,
    "enable_transfer_aware_owner_routing": True,
    "foreground_transfer_min_blocks": 2,
    "foreground_transfer_cost_weight": 1.0,
    "foreground_transfer_min_benefit_ratio": 1.5,
    "foreground_transfer_require_idle_target": True,
    # The profile measures an otherwise-idle packed data path. Add a
    # loaded-serving transaction residual, then scale only the payload-varying
    # part. Pair/size EWMAs can raise the estimate after complete observations.
    "foreground_transfer_bandwidth_gib_s": 3.5,
    "foreground_transfer_latency_profile": None,
    "foreground_transfer_fixed_latency_ms": 0.0,
    "foreground_transfer_interference_multiplier": 1.2,
    "foreground_prefill_token_time_ms": 0.02,
    "foreground_future_reuse_discount": 0.5,
    "foreground_transfer_ewma_alpha": 0.25,
    "enable_kv_transfer_prewarm": True,
    "kv_transfer_prewarm_blocks": 2,
    "foreground_transfer_fail_cooldown_s": 2.0,
    "foreground_transfer_fail_cooldown_max_s": 30.0,
    # 后台 proactive copy-style transfer：worker access snapshot 发现热点，
    # ingress 未提交需求估计剩余复用，并按 NVLink pair 在低负载时串行放置。
    "enable_background_copy": True,
    "background_transfer_mode": "copy",
    "background_move_source_free_block_threshold": 0,
    "background_copy_max_blocks": 8,
    # Coalesce independent prefix candidates on one directed NVLink pair into
    # one control transaction and one contiguous KV payload.
    "background_copy_batch_max_candidates": 16,
    "background_copy_batch_max_blocks": 128,
    "background_copy_cooldown_s": 2.0,
    "background_copy_hot_threshold": 3,
    "background_copy_min_load_skew": 2.0,
    "background_copy_expected_reuses": 4.0,
    "background_copy_idle_pressure_threshold": 2.0,
    "background_copy_flush_timeout_s": 600.0,
}


WORKLOAD_SUMMARY_TITLES = {
    "locality": "KV Locality End-to-End Benchmark Summary",
    "load-skew": "Load-Skew Transfer-Relief Benchmark Summary",
    "memory-skew": "Memory-Skew Capacity-Offload Benchmark Summary",
    "capacity-offload": "Memory-Skew Capacity-Offload Benchmark Summary",
    "transfer-calibration": "KV Transfer Transaction Calibration Summary",
}

SCENARIO_NAMES = (
    "single-gpu",
    "multi-gpu",
    "multi-gpu-kv-routing",
    "multi-gpu-kv-transfer",
    "multi-gpu-lmpool",
)


def workload_summary_title(workload: str) -> str:
    """Return the publication-facing title for one end-to-end workload."""
    try:
        return WORKLOAD_SUMMARY_TITLES[workload]
    except KeyError as exc:
        raise ValueError(f"unknown workload: {workload}") from exc


SUFFIXES = [
    # 共享前缀固定，后缀变化，用来模拟真实业务里“前半段高度重复、后半段各不相同”的请求分布
    "introduce yourself",
    "list all prime numbers within 100",
    "give me your opinion on the impact of artificial intelligence on society",
    "what is the capital of France?",
    "explain quantum computing in simple terms",
    "write a haiku about programming",
    "what is the difference between DNA and RNA?",
    "how does a blockchain work?",
    "explain the theory of relativity briefly",
    "what are the benefits of renewable energy?",
    "describe the water cycle",
    "what is machine learning?",
    "how do airplanes fly?",
    "explain the Pythagorean theorem",
    "what is the speed of light?",
    "write a short poem about the ocean",
]


@dataclass
class ScenarioResult:
    # 每个 benchmark 场景统一产出同一份统计结构，方便最后横向对比和导出 JSON
    name: str
    total_requests: int
    total_tokens: int
    elapsed_s: float
    throughput_tok_s: float
    goodput_tok_s: float
    mean_ttft_s: float
    p50_ttft_s: float
    p90_ttft_s: float
    p95_ttft_s: float
    mean_tpot_s: float
    p50_tpot_s: float
    p90_tpot_s: float
    p95_tpot_s: float
    mean_e2e_s: float
    p50_e2e_s: float
    p90_e2e_s: float
    p95_e2e_s: float
    route_hit_rate: float
    routed_to_prefix_owner_rate: float
    prefix_hit_rate: float
    initial_cached_token_ratio: float
    prefill_attempts: int
    preemption_count: int
    redundant_prefill_tokens: int
    transfer_count: int
    transfer_bytes: int
    transfer_time_s: float
    transfer_source_time_s: float
    transfer_target_time_s: float
    transfer_bandwidth_gib_s: float
    estimated_transfer_cost_ms: float
    estimated_saved_prefill_ms: float
    transfer_copy_count: int
    transfer_release_count: int
    chain_transfer_count: int
    hot_transfer_block_count: int
    hot_transfer_block_ratio: float
    rebalance_success: int
    rebalance_fail: int
    rebalance_fail_reasons: dict[str, int]
    background_copy_success: int
    background_copy_fail: int
    background_copy_fail_reasons: dict[str, int]
    gpu_util_mean: float | None
    gpu_util_p95: float | None
    gpu_mem_util_mean: float | None
    gpu_mem_util_p95: float | None
    rank_stats: dict[int, dict]
    trace_request_share_rate: float = 0.0
    trace_token_share_ratio: float = 0.0
    theoretical_prefix_hit_rate: float = 0.0
    route_matched_block_ratio: float = 0.0
    reclaimable_capacity_route_rate: float = 0.0
    stale_route_hit_rate: float = 0.0
    reuse_phase_request_hit_rate: float = 0.0
    reuse_phase_token_ratio: float = 0.0
    pressure_reuse_overlap_s: float = 0.0
    repetitions: int = 1
    throughput_tok_s_std: float = 0.0
    goodput_tok_s_std: float = 0.0
    mean_ttft_s_std: float = 0.0
    mean_tpot_s_std: float = 0.0
    mean_e2e_s_std: float = 0.0
    p90_e2e_s_std: float = 0.0
    throughput_tok_s_ci95: float = 0.0
    goodput_tok_s_ci95: float = 0.0
    mean_ttft_s_ci95: float = 0.0
    mean_tpot_s_ci95: float = 0.0
    mean_e2e_s_ci95: float = 0.0
    p90_e2e_s_ci95: float = 0.0
    trial_results: list[dict] | None = None
    phase_latency_stats: dict[str, dict[str, float]] | None = None
    pair_spill_count: int = 0
    replica_copy_route_count: int = 0
    placement_lease_route_count: int = 0
    background_placement_stats: dict[str, int] | None = None
    background_placement_pair_stats: dict[str, dict[str, int]] | None = None
    prefill_prompt_tokens: int = 0
    prefill_cached_tokens: int = 0
    prefill_uncached_tokens: int = 0
    placement_wait_s: float = 0.0
    transfer_placement_observations: list[dict] | None = None
    offload_verified: bool = False
    transfer_cost_observation_count: int = 0
    transfer_cost_mae_ms: float = 0.0
    transfer_cost_p95_abs_error_ms: float = 0.0
    transfer_cost_underprediction_rate: float = 0.0
    goodput_sla_sweep_tok_s: dict[str, float] | None = None
    goodput_sla_sweep_tok_s_ci95: dict[str, float] | None = None
    route_decision_counts: dict[str, int] | None = None
    # Per-plan admission evidence.  This is intentionally separate from the
    # aggregate failure counters so a paper run can explain why a transfer was
    # rejected without reconstructing scheduler state from logs.
    rebalance_diagnostics: list[dict] | None = None


def build_shared_prefix(prompt_repeat: int, prefix_group: str = "shared") -> str:
    # 用重复的长文本构造可控的共享前缀，长度越大，越容易触发 prefix cache 命中
    block = (
        "Artificial intelligence is a field of computer science that aims to create systems "
        "capable of performing tasks that normally require human intelligence. These tasks "
        "include learning, reasoning, problem-solving, perception, and language understanding. "
        "Machine learning is a subset of AI that focuses on building systems that can learn "
        "from data. Deep learning is a further subset that uses neural networks with many "
        "layers. The history of AI dates back to the 1950s, but the field has seen several "
        "booms and busts. Efficient inference techniques like quantization, pruning, and "
        "knowledge distillation are active research areas. "
    )
    # 标识放在第一个 token block 内。块 hash 是前缀链式 hash，因此后续内容相同也不会
    # 让不同组错误地共享 KV block。
    group_header = f"LMPool deterministic prefix group {prefix_group}. "
    return group_header + " ".join([block] * prompt_repeat)


def build_prompts(
    tokenizer,
    num_prompts: int,
    prompt_repeat: int,
    workload: str = "locality",
    locality_prefix_groups: int = 16,
    memory_skew_prefix_groups: int = 15,
    memory_skew_warmup_prompts: int = 0,
    memory_skew_pressure_prompts: int = 0,
    memory_skew_trigger_prompts: int = 0,
    memory_skew_pressure_hot_groups: int = 0,
    memory_skew_pressure_hot_share: float = 0.8,
    memory_skew_anchor_share: float = 0.375,
    memory_skew_reuse_hot_groups: int = 0,
    memory_skew_reuse_hot_share: float = 1.0,
    memory_skew_proactive_move: bool = False,
    calibration_prefix_groups: int = 32,
    calibration_warmup_prompts: int = 0,
    load_skew_prefix_groups: int = 6,
    load_skew_warmup_prompts: int = 0,
    load_skew_hot_groups: int = 0,
    load_skew_hot_share: float = 0.8,
    seed: int = 0,
) -> list[str]:
    # locality: 多组长共享前缀，主要验证 KVCache-aware routing，避免单一前缀被每卡复制后
    # round-robin 也自然获得接近 100% 的本地命中。
    # load-skew: 先在每个 NVLink pair 的 source 预热多条长前缀，再提交高并发复用 burst。
    # 它让 forecast-driven background copy 和 replica-aware routing 面对同一组真实热点。
    # memory-skew: 分开构造可迁移的长会话与只承载压力的短锚点。压力请求只
    # 命中锚点，因此会在 source 形成容量短缺，却不会 pin 住要迁移的会话链；
    # 触发请求在压力 drain 后到达，随后精确重放会话以验证实际复用。
    # transfer-calibration: source-build 和 partner-reuse 两阶段只用于采集完整
    # dispatch-to-publish transfer transaction，不是 serving 性能 workload。
    if workload == "locality":
        locality_prefixes = [
            build_shared_prefix(prompt_repeat, f"locality-{group:04d}")
            for group in range(locality_prefix_groups)
        ]
        locality_group_order = [i % locality_prefix_groups for i in range(num_prompts)]
        random.Random(seed).shuffle(locality_group_order)
    elif workload == "load-skew":
        hot_groups = (
            int(load_skew_hot_groups)
            if int(load_skew_hot_groups) > 0
            else int(load_skew_prefix_groups)
        )
        if not 1 <= hot_groups <= load_skew_prefix_groups:
            raise ValueError("--load-skew-hot-groups must fit within --load-skew-prefix-groups")
        if not 0.0 < float(load_skew_hot_share) <= 1.0:
            raise ValueError("--load-skew-hot-share must be in (0, 1]")
        hot_prefixes = [
            build_shared_prefix(prompt_repeat, f"load-hot-{group:04d}")
            for group in range(hot_groups)
        ]
        warmup_end, _ = resolve_load_skew_phases(
            num_prompts,
            hot_groups,
            load_skew_warmup_prompts,
        )
        reuse_count = num_prompts - warmup_end
        hot_reuse_count = min(
            reuse_count,
            max(hot_groups, round(reuse_count * float(load_skew_hot_share))),
        )
        cold_reuse_count = reuse_count - hot_reuse_count
        reuse_group_order = [
            ("hot", index % hot_groups)
            for index in range(hot_reuse_count)
        ]
        reuse_group_order.extend(
            ("cold", index)
            for index in range(cold_reuse_count)
        )
        # Shuffle the weighted multiset so prefix identity cannot lock to the
        # round-robin rank period. The seed keeps the trace reproducible.
        random.Random(seed ^ 0x4C4D504F).shuffle(reuse_group_order)
    elif workload in {"memory-skew", "capacity-offload"}:
        # A session starts with a short anchor and then appends a long body.
        # Pressure and trigger requests share only the anchor. Thus a normal
        # prefix-aware router can return them to the session owner without
        # keeping the long session suffix referenced or pinned.
        if not 0.0 < float(memory_skew_anchor_share) < 1.0:
            raise ValueError("--memory-skew-anchor-share must be in (0, 1)")
        anchor_repeat = min(
            max(1, prompt_repeat - 1),
            max(1, round(prompt_repeat * float(memory_skew_anchor_share))),
        )
        session_repeat = max(1, prompt_repeat - anchor_repeat)
        anchor_prefixes = [
            build_shared_prefix(anchor_repeat, f"transfer-anchor-{group:04d}")
            for group in range(memory_skew_prefix_groups)
        ]
        session_prefixes = [
            anchor_prefixes[group]
            + " "
            + build_shared_prefix(session_repeat, f"transfer-session-{group:04d}")
            for group in range(memory_skew_prefix_groups)
        ]
        warmup_prompts, pressure_prompts, trigger_prompts, _ = resolve_memory_skew_phases(
            num_prompts,
            memory_skew_prefix_groups,
            memory_skew_warmup_prompts,
            memory_skew_pressure_prompts,
            memory_skew_trigger_prompts,
            allow_zero_trigger=memory_skew_proactive_move,
        )
        warmup_end = warmup_prompts
        pressure_end = warmup_prompts + pressure_prompts
        trigger_end = pressure_end + trigger_prompts
        pressure_hot_groups = (
            int(memory_skew_pressure_hot_groups)
            if int(memory_skew_pressure_hot_groups) > 0
            else min(2, memory_skew_prefix_groups)
        )
        if not 1 <= pressure_hot_groups <= memory_skew_prefix_groups:
            raise ValueError(
                "--memory-skew-pressure-hot-groups must fit within "
                "--memory-skew-prefix-groups"
            )
        if not 0.0 < float(memory_skew_pressure_hot_share) <= 1.0:
            raise ValueError("--memory-skew-pressure-hot-share must be in (0, 1]")
        reuse_hot_groups = (
            int(memory_skew_reuse_hot_groups)
            if int(memory_skew_reuse_hot_groups) > 0
            else memory_skew_prefix_groups
        )
        if not 1 <= reuse_hot_groups <= memory_skew_prefix_groups:
            raise ValueError(
                "--memory-skew-reuse-hot-groups must fit within "
                "--memory-skew-prefix-groups"
            )
        if not 0.0 < float(memory_skew_reuse_hot_share) <= 1.0:
            raise ValueError("--memory-skew-reuse-hot-share must be in (0, 1]")
        hot_pressure_count = min(
            pressure_prompts,
            max(
                pressure_hot_groups,
                round(pressure_prompts * float(memory_skew_pressure_hot_share)),
            ),
        )
        pressure_group_order = [
            index % pressure_hot_groups for index in range(hot_pressure_count)
        ]
        cold_groups = list(range(pressure_hot_groups, memory_skew_prefix_groups))
        pressure_group_order.extend(
            cold_groups[index % len(cold_groups)] if cold_groups else index % pressure_hot_groups
            for index in range(pressure_prompts - hot_pressure_count)
        )
        # Each phase has an independent deterministic permutation. This is a
        # workload arrival order, not a worker-placement hint, and prevents a
        # prefix group from repeatedly landing on the same round-robin rank.
        warmup_group_order = [
            index % memory_skew_prefix_groups for index in range(warmup_prompts)
        ]
        trigger_group_order = [
            index % pressure_hot_groups for index in range(trigger_prompts)
        ]
        reuse_count = num_prompts - trigger_end
        hot_reuse_count = min(
            reuse_count,
            max(
                reuse_hot_groups,
                round(reuse_count * float(memory_skew_reuse_hot_share)),
            ),
        )
        reuse_group_order = [
            index % reuse_hot_groups for index in range(hot_reuse_count)
        ]
        cold_groups = list(range(reuse_hot_groups, memory_skew_prefix_groups))
        reuse_group_order.extend(
            cold_groups[index % len(cold_groups)] if cold_groups else index % reuse_hot_groups
            for index in range(reuse_count - hot_reuse_count)
        )
        random.Random(seed ^ 0x4D534B57).shuffle(warmup_group_order)
        random.Random(seed ^ 0x50524553).shuffle(pressure_group_order)
        random.Random(seed ^ 0x54524947).shuffle(trigger_group_order)
        random.Random(seed ^ 0x52455553).shuffle(reuse_group_order)
    elif workload == "transfer-calibration":
        hot_prefixes = [
            build_shared_prefix(prompt_repeat, f"calibration-{group:04d}")
            for group in range(calibration_prefix_groups)
        ]
        warmup_end = calibration_warmup_prompts or num_prompts // 2
    else:
        raise ValueError(f"unknown workload: {workload}")
    prompts = []
    for i in range(num_prompts):
        suffix = SUFFIXES[i % len(SUFFIXES)]
        if workload == "locality":
            shared_prefix = locality_prefixes[locality_group_order[i]]
        elif workload == "load-skew":
            if i < warmup_end:
                shared_prefix = hot_prefixes[i % hot_groups]
            else:
                reuse_kind, reuse_group = reuse_group_order[i - warmup_end]
                if reuse_kind == "hot":
                    shared_prefix = hot_prefixes[reuse_group]
                else:
                    shared_prefix = build_shared_prefix(
                        prompt_repeat,
                        f"load-cold-{reuse_group:04d}",
                    )
        elif workload in {"memory-skew", "capacity-offload"}:
            if i < warmup_end:
                session_group = warmup_group_order[i]
                shared_prefix = session_prefixes[session_group]
                suffix = SUFFIXES[session_group % len(SUFFIXES)]
            elif i < pressure_end:
                pressure_index = i - warmup_end
                anchor_group = pressure_group_order[pressure_index]
                shared_prefix = (
                    anchor_prefixes[anchor_group]
                    + " "
                    + build_shared_prefix(
                        anchor_repeat,
                        f"memory-pressure-tail-{pressure_index:04d}",
                    )
                )
            elif i < trigger_end:
                trigger_index = i - pressure_end
                anchor_group = trigger_group_order[trigger_index]
                # A long unique tail makes the already loaded source request
                # additional blocks only after all pressure requests have
                # completed and their blocks are releasable.
                shared_prefix = (
                    anchor_prefixes[anchor_group]
                    + " "
                    + build_shared_prefix(
                        session_repeat,
                        f"memory-trigger-tail-{trigger_index:04d}",
                    )
                )
            else:
                reuse_index = i - trigger_end
                session_group = reuse_group_order[reuse_index]
                shared_prefix = session_prefixes[session_group]
                # Exact replay requires the same suffix used for the session
                # warm-up request; otherwise the final block hash differs.
                suffix = SUFFIXES[session_group % len(SUFFIXES)]
        elif workload == "transfer-calibration":
            phase_index = i if i < warmup_end else i - warmup_end
            shared_prefix = hot_prefixes[phase_index % calibration_prefix_groups]
        else:
            raise ValueError(f"unknown workload: {workload}")
        prompt = f"{shared_prefix} Now answer the following request: {suffix}"
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def compute_prefix_hashes(tokenizer, prompts: Iterable[str], block_size: int):
    # 先把 prompt 转成 Sequence，方便后续直接复用 Sequence 的 block 计数和 hash 逻辑
    seqs = [
        Sequence(tokenizer.encode(prompt), block_size=block_size)
        for prompt in prompts
    ]
    return seqs


def compute_sequence_prefix_hashes(seq: Sequence) -> list[int]:
    """Compute cumulative hashes for every complete logical block in a sequence."""
    block_manager = BlockManager(num_blocks=1, block_size=seq.block_size)
    hashes = []
    prefix_hash = -1
    for block_index in range(seq.num_tokens // seq.block_size):
        prefix_hash = block_manager.compute_hash(seq.block(block_index), prefix_hash)
        hashes.append(prefix_hash)
    return hashes


def profile_trace_prefix_sharing(
    tokenizer,
    prompts: list[str],
    block_size: int,
) -> dict[str, int | float]:
    """Profile block-aligned prefix reuse intrinsic to an ordered prompt trace.

    The profile assumes an unlimited logical cache and perfect placement. A
    request is shareable when at least one complete prefix block appeared in an
    earlier request. Token sharing counts the longest contiguous sequence of
    previously observed complete blocks from block zero; partial tail blocks
    are excluded because the runtime cannot publish them as reusable KV.
    """
    seqs = compute_prefix_hashes(tokenizer, prompts, block_size)
    seen_prefix_hashes: set[int] = set()
    shareable_requests = 0
    shareable_prefix_blocks = 0
    total_prompt_tokens = 0
    total_complete_blocks = 0

    for seq in seqs:
        prefix_hashes = compute_sequence_prefix_hashes(seq)
        matched_blocks = 0
        for prefix_hash in prefix_hashes:
            if prefix_hash not in seen_prefix_hashes:
                break
            matched_blocks += 1

        if matched_blocks:
            shareable_requests += 1
        shareable_prefix_blocks += matched_blocks
        total_prompt_tokens += seq.num_tokens
        total_complete_blocks += len(prefix_hashes)
        seen_prefix_hashes.update(prefix_hashes)

    shareable_prefix_tokens = shareable_prefix_blocks * block_size
    return {
        "block_size_tokens": block_size,
        "total_requests": len(seqs),
        "total_prompt_tokens": total_prompt_tokens,
        "total_complete_blocks": total_complete_blocks,
        "unique_complete_prefix_hashes": len(seen_prefix_hashes),
        "shareable_requests": shareable_requests,
        "shareable_prefix_blocks": shareable_prefix_blocks,
        "shareable_prefix_tokens": shareable_prefix_tokens,
        "request_prefix_share_rate": shareable_requests / max(len(seqs), 1),
        "token_prefix_share_ratio": (
            shareable_prefix_tokens / max(total_prompt_tokens, 1)
        ),
    }


def _percentile(values: list[float], p: float) -> float:
    # 轻量 percentile 计算，避免引入额外依赖
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values = sorted(values)
    rank = (len(values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    frac = rank - low
    return values[low] * (1 - frac) + values[high] * frac


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def decode_tpot_s(
    first_token_at: float,
    finished_at: float,
    output_tokens: int,
) -> float | None:
    """Return decode time per output token, excluding time to first token."""
    if output_tokens <= 1:
        return None
    return max(0.0, finished_at - first_token_at) / (output_tokens - 1)


def compute_goodput_sla_sweep(
    completion_times: dict[int, float],
    submit_times: dict[int, float],
    completion_token_counts: dict[int, int],
    elapsed_s: float,
    sla_thresholds_s: Iterable[float],
) -> dict[str, float]:
    """Compute several goodput thresholds from one set of request samples."""
    result = {}
    for sla_s in sla_thresholds_s:
        sla_ms_key = f"{float(sla_s) * 1000.0:g}"
        sla_tokens = sum(
            completion_token_counts[seq_id]
            for seq_id, done_at in completion_times.items()
            if done_at - submit_times[seq_id] <= float(sla_s)
        )
        result[sla_ms_key] = sla_tokens / max(elapsed_s, 1e-9)
    return result


def resolve_load_skew_phases(
    num_prompts: int,
    prefix_groups: int,
    requested_warmup_prompts: int = 0,
) -> tuple[int, int]:
    """Resolve source warm-up and burst reuse sizes for load skew."""
    if num_prompts < 2:
        raise ValueError("load-skew requires --num-prompts >= 2")
    if prefix_groups < 1:
        raise ValueError("--load-skew-prefix-groups must be >= 1")
    warmup_prompts = requested_warmup_prompts or max(
        prefix_groups,
        num_prompts // 4,
    )
    reuse_prompts = num_prompts - warmup_prompts
    if warmup_prompts < prefix_groups:
        raise ValueError(
            "load-skew warm-up phase must cover every prefix group"
        )
    if reuse_prompts < prefix_groups:
        raise ValueError(
            "load-skew reuse phase must cover every prefix group"
        )
    return warmup_prompts, reuse_prompts


def resolve_memory_skew_prefix_groups(
    num_prompts: int,
    requested: int,
    warmup_prompts: int = 0,
    pressure_prompts: int = 0,
    trigger_prompts: int = 0,
    allow_zero_trigger: bool = False,
) -> int:
    """Resolve session groups that fit the warm-up, trigger, and reuse phases."""
    warmup_requests = (
        int(warmup_prompts)
        if int(warmup_prompts) > 0
        else max(1, num_prompts // 4)
    )
    pressure_requests = (
        int(pressure_prompts)
        if int(pressure_prompts) > 0
        else max(1, num_prompts // 4)
    )
    trigger_requests = (
        int(trigger_prompts)
        if int(trigger_prompts) > 0
        else (0 if allow_zero_trigger else max(1, num_prompts // 8))
    )
    reuse_requests = num_prompts - warmup_requests - pressure_requests - trigger_requests
    maximum = min(warmup_requests, reuse_requests)
    if not allow_zero_trigger:
        maximum = min(maximum, trigger_requests)
    if requested > 0:
        if requested > maximum:
            raise ValueError(
                "--memory-skew-prefix-groups must fit in warm-up and reuse phases"
            )
        return requested
    automatic = min(15, maximum)
    return automatic if automatic % 2 == 1 else max(1, automatic - 1)


def resolve_memory_skew_phases(
    num_prompts: int,
    prefix_groups: int,
    warmup_prompts: int,
    pressure_prompts: int,
    trigger_prompts: int = 0,
    allow_zero_trigger: bool = False,
) -> tuple[int, int, int, int]:
    """Resolve warm-up, pressure, trigger, and exact-reuse phase lengths."""
    warmup = (
        int(warmup_prompts)
        if int(warmup_prompts) > 0
        else max(int(prefix_groups), int(num_prompts) // 4)
    )
    pressure = (
        int(pressure_prompts)
        if int(pressure_prompts) > 0
        else max(1, int(num_prompts) // 4)
    )
    trigger = (
        int(trigger_prompts)
        if int(trigger_prompts) > 0
        else (0 if allow_zero_trigger else int(prefix_groups))
    )
    reuse = int(num_prompts) - warmup - pressure - trigger
    if min(warmup, pressure, reuse) < 1 or (not allow_zero_trigger and trigger < 1):
        raise ValueError(
            "memory-skew requires non-empty warm-up, pressure, "
            + ("and reuse phases" if allow_zero_trigger else "trigger, and reuse phases")
        )
    if warmup < int(prefix_groups):
        raise ValueError(
            "memory-skew warm-up phase must build every session prefix group"
        )
    if reuse < int(prefix_groups):
        raise ValueError(
            "memory-skew reuse phase must cover every prefix group"
        )
    return warmup, pressure, trigger, reuse


def resolve_transfer_calibration_warmup_prompts(
    num_prompts: int,
    requested: int,
) -> int:
    """Resolve the source-build phase of a transaction calibration trace."""
    if num_prompts < 2:
        raise ValueError("transfer-calibration requires --num-prompts >= 2")
    if requested <= 0:
        if num_prompts % 2:
            raise ValueError(
                "transfer-calibration requires even --num-prompts when "
                "--calibration-warmup-prompts is 0"
            )
        return num_prompts // 2
    if requested >= num_prompts:
        raise ValueError(
            "--calibration-warmup-prompts must be smaller than --num-prompts"
        )
    return requested


def resolve_transfer_calibration_prefix_groups(
    num_prompts: int,
    requested: int,
    warmup_prompts: int | None = None,
) -> int:
    """Resolve prefix groups represented in both calibration phases."""
    warmup_requests = resolve_transfer_calibration_warmup_prompts(
        num_prompts,
        0 if warmup_prompts is None else warmup_prompts,
    )
    maximum = min(warmup_requests, num_prompts - warmup_requests)
    if requested > maximum:
        raise ValueError(
            "--calibration-prefix-groups must fit in both calibration phases"
        )
    return requested if requested > 0 else min(32, maximum)


def _visible_physical_gpu_ids(world_size: int) -> list[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw:
        ids = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                ids.append(int(item))
            except ValueError:
                return list(range(world_size))
        return ids[:world_size]
    return list(range(world_size))


def _sample_gpu_metrics_once(physical_gpu_ids: list[int]) -> list[tuple[float, float]]:
    # 通过 nvidia-smi 采样 GPU 利用率和显存利用率，属于外部观测面，不参与调度决策
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    samples = []
    wanted = set(physical_gpu_ids)
    for physical_idx, line in enumerate(output.strip().splitlines()):
        if physical_idx not in wanted:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            util = float(parts[0])
            mem_used = float(parts[1])
            mem_total = float(parts[2])
        except ValueError:
            continue
        mem_util = (mem_used / mem_total * 100.0) if mem_total > 0 else 0.0
        samples.append((util, mem_util))
    return samples


class GpuMetricSampler:
    # 后台定时采样器：benchmark 跑的同时持续抓 GPU 状态，最后再汇总 mean / p95
    def __init__(self, interval_s: float = 0.5, world_size: int = 1):
        self.interval_s = interval_s
        self.physical_gpu_ids = _visible_physical_gpu_ids(world_size)
        self.samples: list[list[tuple[float, float]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            samples = _sample_gpu_metrics_once(self.physical_gpu_ids)
            if samples:
                self.samples.append(samples)
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def summarize(self) -> tuple[float | None, float | None, float | None, float | None]:
        if not self.samples:
            return None, None, None, None
        util_values = [sample[0] for batch in self.samples for sample in batch]
        mem_values = [sample[1] for batch in self.samples for sample in batch]
        return (
            statistics.mean(util_values),
            _percentile(util_values, 0.95),
            statistics.mean(mem_values),
            _percentile(mem_values, 0.95),
        )

    def summarize_by_rank(self) -> dict[int, dict[str, float]]:
        summaries: dict[int, dict[str, float]] = {}
        if not self.samples:
            return summaries
        for logical_rank in range(len(self.physical_gpu_ids)):
            util_values = [
                batch[logical_rank][0]
                for batch in self.samples
                if logical_rank < len(batch)
            ]
            mem_values = [
                batch[logical_rank][1]
                for batch in self.samples
                if logical_rank < len(batch)
            ]
            if not util_values:
                continue
            summaries[logical_rank] = {
                "gpu_util_mean": statistics.mean(util_values),
                "gpu_util_p95": _percentile(util_values, 0.95),
                "gpu_mem_util_mean": statistics.mean(mem_values),
                "gpu_mem_util_p95": _percentile(mem_values, 0.95),
                "physical_gpu_id": self.physical_gpu_ids[logical_rank],
            }
        return summaries


def _run_independent_worker(
    gpu_index: int,
    config: dict,
    prompt_token_ids: list[list[int]],
    sampling_params: SamplingParams,
    goodput_e2e_sla_s: float,
    result_queue,
):
    # 旧版独立 multi-gpu helper：
    # - 每个 GPU 一个进程
    # - 没有全局控制面
    # - prompt 只做静态切分
    # 这条路径保留给离线 shard 对照实验。当前 main() 里的 `multi-gpu`
    # 场景使用 LLMEngine + round-robin 在线提交，不再调用这个 helper。
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    import torch.distributed as dist
    dist.init_process_group = lambda *args, **kwargs: None  # type: ignore[assignment]
    dist.destroy_process_group = lambda *args, **kwargs: None  # type: ignore[assignment]

    from lmpool.engine.model_runner import ModelRunner
    from lmpool.engine.scheduler import Scheduler

    model_runner = ModelRunner(config, rank=0, gbm=None)
    scheduler = Scheduler(
        max_num_sequences=config.get("max_num_sequences", 16),
        max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
        max_cached_blocks=config.get("max_cached_blocks", 1024),
        block_size=config.get("block_size", 256),
        eos=config.get("eos", 50256),
        global_scheduler=None,
    )

    submitted_at: dict[int, float] = {}
    ttfts: list[float] = []
    first_token_at: dict[int, float] = {}
    tpots: list[float] = []
    e2es: list[float] = []
    prefix_hits = 0
    total_tokens = 0
    seq_count = 0
    goodput_tokens = 0

    # 每张卡拿到一小段静态 shard，worker 只处理自己这份请求
    for token_ids in prompt_token_ids:
        seq = Sequence(token_ids=token_ids, block_size=config["block_size"], sampling_params=sampling_params)
        scheduler.add_sequence(seq)
        submitted_at[seq.seq_id] = time.perf_counter()
        seq_count += 1

    start_wall = time.perf_counter()
    # 本地循环只关心本 shard 的完成情况，不和其他 GPU 做任何协同
    while not scheduler.is_finished():
        scheduled, is_prefill = scheduler.schedule()
        if not scheduled:
            continue
        if is_prefill:
            prefix_hits += sum(1 for seq in scheduled if seq.num_cached_tokens > 0)
        outputs = model_runner.run(scheduled, is_prefill)
        now = time.perf_counter()
        scheduler.postprocess(scheduled, outputs)
        for seq in scheduled:
            latency = now - submitted_at[seq.seq_id]
            if seq.num_completion_tokens == 1:
                ttfts.append(latency)
                first_token_at[seq.seq_id] = now
            if seq.is_finished:
                e2es.append(latency)
                tpot = decode_tpot_s(
                    first_token_at.get(seq.seq_id, now),
                    now,
                    seq.num_completion_tokens,
                )
                if tpot is not None:
                    tpots.append(tpot)
                if latency <= goodput_e2e_sla_s:
                    goodput_tokens += seq.num_completion_tokens
        total_tokens += len(outputs)

    elapsed = time.perf_counter() - start_wall
    result_queue.put(
        {
            "gpu_index": gpu_index,
            "total_requests": seq_count,
            "total_tokens": total_tokens,
            "elapsed_s": elapsed,
            "throughput_tok_s": total_tokens / max(elapsed, 1e-9),
            "goodput_tokens": goodput_tokens,
            "mean_ttft_s": _mean(ttfts),
            "p50_ttft_s": _median(ttfts),
            "p90_ttft_s": _percentile(ttfts, 0.90),
            "p95_ttft_s": _percentile(ttfts, 0.95),
            "ttfts": ttfts,
            "tpots": tpots,
            "prefix_hit_rate": prefix_hits / max(seq_count, 1),
            "mean_e2e_s": _mean(e2es),
            "p50_e2e_s": _median(e2es),
            "p90_e2e_s": _percentile(e2es, 0.90),
            "p95_e2e_s": _percentile(e2es, 0.95),
            "e2es": e2es,
        }
    )


def run_independent_multi_gpu_benchmark(
    name: str,
    config: dict,
    prompts: list[str],
    sampling_params: SamplingParams,
    tokenizer,
    goodput_e2e_sla_s: float,
) -> ScenarioResult | None:
    # 旧版离线 baseline：先按 GPU 数量把请求静态切分，再分别启动 worker。
    # 当前 main() 不调用它；保留它是为了需要静态 shard 对照实验时复用。
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        return None

    ctx = mp.get_context("spawn")
    prompt_token_ids = [tokenizer.encode(prompt) for prompt in prompts]
    shards: list[list[list[int]]] = [[] for _ in range(gpu_count)]
    for idx, token_ids in enumerate(prompt_token_ids):
        # 轮转切 shard：第 i 个请求固定分到 i % gpu_count 的 GPU
        shards[idx % gpu_count].append(token_ids)

    result_queue = ctx.Queue()
    procs = []
    start_wall = time.perf_counter()
    sampler = GpuMetricSampler(interval_s=0.5, world_size=gpu_count)
    try:
        sampler.start()
        for gpu_index, shard in enumerate(shards):
            if not shard:
                continue
            proc = ctx.Process(
                target=_run_independent_worker,
                args=(gpu_index, dict(config), shard, sampling_params, goodput_e2e_sla_s, result_queue),
            )
            procs.append(proc)
            proc.start()

        results = []
        deadline = time.perf_counter() + 3600
        while len(results) < len(procs) and time.perf_counter() < deadline:
            try:
                results.append(result_queue.get(timeout=1))
            except Exception:
                pass

        if len(results) < len(procs):
            raise RuntimeError("independent baseline workers did not finish in time")

        elapsed = time.perf_counter() - start_wall
        total_requests = sum(item["total_requests"] for item in results)
        total_tokens = sum(item["total_tokens"] for item in results)
        # 汇总各卡结果时，把每个 worker 的 token / latency 样本合并起来
        ttfts = [lat for item in results for lat in item.get("ttfts", [])]
        tpots = [lat for item in results for lat in item.get("tpots", [])]
        e2es = [lat for item in results for lat in item.get("e2es", [])]
        prefix_hit_rate = sum(
            item["prefix_hit_rate"] * item["total_requests"] for item in results
        ) / max(total_requests, 1)
        goodput_tokens = sum(item["goodput_tokens"] for item in results)
        gpu_util_mean, gpu_util_p95, gpu_mem_util_mean, gpu_mem_util_p95 = sampler.summarize()
        rank_gpu_stats = sampler.summarize_by_rank()
        return ScenarioResult(
            name=name,
            total_requests=total_requests,
            total_tokens=total_tokens,
            elapsed_s=elapsed,
            throughput_tok_s=total_tokens / max(elapsed, 1e-9),
            goodput_tok_s=goodput_tokens / max(elapsed, 1e-9),
            mean_ttft_s=_mean(ttfts),
            p50_ttft_s=_median(ttfts),
            p90_ttft_s=_percentile(ttfts, 0.90),
            p95_ttft_s=_percentile(ttfts, 0.95),
            mean_tpot_s=_mean(tpots),
            p50_tpot_s=_median(tpots),
            p90_tpot_s=_percentile(tpots, 0.90),
            p95_tpot_s=_percentile(tpots, 0.95),
            mean_e2e_s=_mean(e2es),
            p50_e2e_s=_median(e2es),
            p90_e2e_s=_percentile(e2es, 0.90),
            p95_e2e_s=_percentile(e2es, 0.95),
            route_hit_rate=0.0,
            routed_to_prefix_owner_rate=0.0,
            prefix_hit_rate=prefix_hit_rate,
            initial_cached_token_ratio=0.0,
            prefill_attempts=total_requests,
            preemption_count=0,
            redundant_prefill_tokens=0,
            transfer_count=0,
            transfer_bytes=0,
            transfer_time_s=0.0,
            transfer_source_time_s=0.0,
            transfer_target_time_s=0.0,
            transfer_bandwidth_gib_s=0.0,
            estimated_transfer_cost_ms=0.0,
            estimated_saved_prefill_ms=0.0,
            transfer_copy_count=0,
            transfer_release_count=0,
            chain_transfer_count=0,
            hot_transfer_block_count=0,
            hot_transfer_block_ratio=0.0,
            rebalance_success=0,
            rebalance_fail=0,
            rebalance_fail_reasons={},
            background_copy_success=0,
            background_copy_fail=0,
            background_copy_fail_reasons={},
            gpu_util_mean=gpu_util_mean,
            gpu_util_p95=gpu_util_p95,
            gpu_mem_util_mean=gpu_mem_util_mean,
            gpu_mem_util_p95=gpu_mem_util_p95,
            rank_stats={
                item.get("rank", idx): {
                    "requests": item.get("total_requests", 0),
                    "output_tokens": item.get("total_tokens", 0),
                    "prefix_hit_rate": item.get("prefix_hit_rate", 0.0),
                    **rank_gpu_stats.get(item.get("rank", idx), {}),
                }
                for idx, item in enumerate(results)
            },
        )
    finally:
        sampler.stop()
        for proc in procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
            proc.close()
        result_queue.close()
        result_queue.join_thread()


def measure_single_gpu_prefix_hit_rate(
    tokenizer,
    prompts: list[str],
    block_size: int,
    max_cached_blocks: int,
) -> float:
    """Compatibility wrapper for the capacity-unbounded trace request rate."""
    del max_cached_blocks
    profile = profile_trace_prefix_sharing(tokenizer, prompts, block_size)
    return float(profile["request_prefix_share_rate"])


def run_engine_scenario(
    name: str,
    config: dict,
    prompts: list[str],
    sampling_params: SamplingParams,
    tokenizer,
    route_mode: str = "control_plane",
    goodput_e2e_sla_s: float = 2.0,
    submit_window: int = 8,
    workload: str = "locality",
) -> ScenarioResult:
    # 调用 LLMEngine
    # prompt 先进入 launcher
    # 再由控制面路由或按 round-robin 分发
    # worker 侧执行 prefill / decode
    config, rendezvous_path = prepare_benchmark_rendezvous(config)
    engine = LLMEngine(config)
    submit_times: dict[int, float] = {}
    ttfts: list[float] = []
    first_token_at: dict[int, float] = {}
    tpots: list[float] = []
    e2es: list[float] = []
    total_tokens = 0
    goodput_tokens = 0
    route_hits = 0
    routed_to_prefix_owner = 0
    route_count = 0
    pair_spill_count = 0
    replica_copy_route_count = 0
    placement_lease_route_count = 0
    route_matched_blocks = 0
    route_full_blocks = 0
    reclaimable_capacity_routes = 0
    routed_match_by_seq: dict[int, int] = {}
    stale_route_hits = 0
    prefill_seen_seq_ids: set[int] = set()
    prefill_hit_seq_ids: set[int] = set()
    reuse_phase_seq_ids: set[int] = set()
    reuse_phase_hit_seq_ids: set[int] = set()
    reuse_phase_cached_tokens = 0
    reuse_phase_prompt_tokens = 0
    phase_by_seq: dict[int, str] = {}
    phase_ttfts: dict[str, list[float]] = {
        "warmup": [],
        "pressure": [],
        "trigger": [],
        "reuse": [],
    }
    phase_e2es: dict[str, list[float]] = {
        "warmup": [],
        "pressure": [],
        "trigger": [],
        "reuse": [],
    }
    phase_output_tokens: dict[str, int] = {
        "warmup": 0,
        "pressure": 0,
        "trigger": 0,
        "reuse": 0,
    }
    phase_started_at: dict[str, float] = {}
    phase_finished_at: dict[str, float] = {}
    initial_cached_tokens = 0
    initial_prompt_tokens = 0
    prefill_attempts = 0
    preemption_count = 0
    transfer_count = 0
    transfer_bytes = 0
    transfer_time_s = 0.0
    transfer_source_time_s = 0.0
    transfer_target_time_s = 0.0
    estimated_transfer_cost_ms = 0.0
    estimated_saved_prefill_ms = 0.0
    transfer_copy_count = 0
    transfer_release_count = 0
    chain_transfer_count = 0
    hot_transfer_block_count = 0
    rebalance_success = 0
    rebalance_fail = 0
    rebalance_fail_reasons: dict[str, int] = {}
    rebalance_diagnostics: list[dict] = []
    route_decision_counts: dict[str, int] = {}
    background_copy_success = 0
    background_copy_fail = 0
    background_copy_fail_reasons: dict[str, int] = {}
    background_placement_stats: dict[str, int] = {}
    background_placement_pair_stats: dict[str, dict[str, int]] = {}
    placement_wait_s = 0.0
    rank_stats: dict[int, dict] = {}
    start_wall = 0.0
    last_progress_report = 0.0
    sampler = GpuMetricSampler(interval_s=0.5, world_size=config["world_size"])
    sampler_started = False

    def get_rank_stats(rank: int) -> dict:
        return rank_stats.setdefault(
            int(rank),
            {
                "submitted": 0,
                "warmup_submitted": 0,
                "pressure_submitted": 0,
                "trigger_submitted": 0,
                "reuse_submitted": 0,
                "finished": 0,
                "output_tokens": 0,
                "first_tokens": 0,
                "prefill_requests": 0,
                "prefill_attempts": 0,
                "prefill_prefix_hits": 0,
                "initial_cached_tokens": 0,
                "initial_prompt_tokens": 0,
                "preemption_count": 0,
                "prefill_tokens": 0,
                "prefill_prompt_tokens": 0,
                "prefill_cached_tokens": 0,
                "prefill_uncached_tokens": 0,
                "decode_tokens": 0,
                "prefill_time_s": 0.0,
                "decode_time_s": 0.0,
                "transfers": 0,
                "transfer_bytes": 0,
                "transfer_time_s": 0.0,
                "transfer_source_time_s": 0.0,
                "transfer_target_time_s": 0.0,
                "copies": 0,
                "released_blocks": 0,
                "chain_transfers": 0,
                "hot_transfer_blocks": 0,
                "rebalance_success": 0,
                "rebalance_fail": 0,
                "route_decision_counts": {},
                "background_copy_success": 0,
                "background_copy_fail": 0,
                "max_cached_blocks": 0,
                "free_blocks_after_release": -1,
            },
        )

    try:
        worker_capacities: dict[int, int] = {}
        capacity_deadline = time.monotonic() + 600.0
        while len(worker_capacities) < config["world_size"]:
            if time.monotonic() >= capacity_deadline:
                raise RuntimeError(
                    f"timed out waiting for KV capacities: received "
                    f"{len(worker_capacities)}/{config['world_size']} workers"
                )
            _finished, _first_tokens, _prefill_stats, startup_stats = engine.step()
            for item in startup_stats:
                if "max_cached_blocks" in item and "rank" in item:
                    worker_capacities[int(item["rank"])] = int(item["max_cached_blocks"])
        for rank, capacity in worker_capacities.items():
            get_rank_stats(rank)["max_cached_blocks"] = capacity

        requested_budget = int(config.get("max_cached_blocks", 0))
        actual_budget = min(worker_capacities.values())
        if config.get("require_exact_kv_block_budget") and actual_budget < requested_budget:
            raise RuntimeError(
                f"requested --kv-block-budget {requested_budget}, but workers can allocate "
                f"only {actual_budget} blocks with --gpu-memory-utilization "
                f"{config.get('gpu_memory_utilization')}. Increase GPU memory utilization "
                f"or lower the KV block budget."
            )

        next_prompt_idx = 0
        finished_count = 0
        completion_times: dict[int, float] = {}
        completion_token_counts: dict[int, int] = {}
        inflight: set[int] = set()
        effective_submit_window = len(prompts) if submit_window <= 0 else max(1, submit_window)
        transfer_workload = workload in {
            "load-skew",
            "memory-skew",
            "capacity-offload",
            "transfer-calibration",
        }
        if transfer_workload:
            if workload == "load-skew":
                warmup_end, _ = resolve_load_skew_phases(
                    len(prompts),
                    int(config["benchmark_transfer_prefix_groups"]),
                    int(config.get("benchmark_load_skew_warmup_prompts", 0)),
                )
                pressure_end = warmup_end
                trigger_end = pressure_end
                phase_ends = [warmup_end, len(prompts)]
            elif workload in {"memory-skew", "capacity-offload"}:
                warmup_prompts, pressure_prompts, trigger_prompts, _ = resolve_memory_skew_phases(
                    len(prompts),
                    int(config["benchmark_transfer_prefix_groups"]),
                    int(config.get("benchmark_memory_skew_warmup_prompts", 0)),
                    int(config.get("benchmark_memory_skew_pressure_prompts", 0)),
                    int(config.get("benchmark_memory_skew_trigger_prompts", 0)),
                    allow_zero_trigger=bool(
                        config.get("benchmark_memory_skew_proactive_move", False)
                    ),
                )
                warmup_end = warmup_prompts
                pressure_end = warmup_prompts + pressure_prompts
                trigger_end = pressure_end + trigger_prompts
                phase_ends = [warmup_end, pressure_end, trigger_end, len(prompts)]
            else:
                warmup_end = int(
                    config.get(
                        "benchmark_calibration_warmup_prompts",
                        len(prompts) // 2,
                    )
                )
                pressure_end = warmup_end
                phase_ends = [warmup_end, len(prompts)]
                trigger_end = pressure_end
            warmup_hash_chains = [
                compute_sequence_prefix_hashes(Sequence(
                    token_ids=tokenizer.encode(prompts[index]),
                    block_size=config["block_size"],
                ))
                for index in range(warmup_end)
            ]
            warmup_hash_frequency: dict[int, int] = {}
            for chain in warmup_hash_chains:
                for block_hash in chain:
                    warmup_hash_frequency[block_hash] = warmup_hash_frequency.get(block_hash, 0) + 1
            hot_prefix_hashes = {
                block_hash
                for block_hash, frequency in warmup_hash_frequency.items()
                if frequency >= (1 if workload == "transfer-calibration" else 2)
            }
            future_prefix_demands: dict[int, int] = {}
            future_start = (
                trigger_end
                if workload in {"memory-skew", "capacity-offload"}
                else pressure_end
            )
            for index in range(future_start, len(prompts)):
                seq = Sequence(
                    token_ids=tokenizer.encode(prompts[index]),
                    block_size=config["block_size"],
                )
                for block_hash in compute_sequence_prefix_hashes(seq):
                    future_prefix_demands[block_hash] = (
                        future_prefix_demands.get(block_hash, 0) + 1
                    )
        else:
            warmup_end = pressure_end = trigger_end = 0
            phase_ends = [len(prompts)]
            hot_prefix_hashes = set()
            future_prefix_demands = {}
        # Serving metrics start only after model load, CUDA/NCCL warmup, KV
        # allocation, and benchmark metadata preparation have completed.
        sampler.start()
        sampler_started = True
        start_wall = time.perf_counter()
        last_progress_report = start_wall
        current_phase_index = 0
        current_phase_end = phase_ends[current_phase_index]

        def submit_prompt(prompt: str, prompt_index: int):
            nonlocal route_hits, routed_to_prefix_owner, route_count
            nonlocal pair_spill_count, replica_copy_route_count
            nonlocal placement_lease_route_count
            nonlocal route_matched_blocks, route_full_blocks
            nonlocal reclaimable_capacity_routes
            seq = Sequence(
                token_ids=tokenizer.encode(prompt),
                block_size=config["block_size"],
                sampling_params=sampling_params,
            )
            start = time.perf_counter()
            target_rank = 0
            if route_mode == "control_plane" and engine.control_plane_client is not None:
                # 控制面模式：每个请求都先做 prefix hash，再让全局调度器决定落在哪张卡
                routed = engine.control_plane_client.route_sequence(seq, return_meta=True)
                target_rank = routed["target_rank"]
                route_info = routed.get("route_info", {})
                route_reason = str(route_info.get("reason") or "unknown")
                route_decision_counts[route_reason] = (
                    route_decision_counts.get(route_reason, 0) + 1
                )
                rank_route_counts = get_rank_stats(target_rank)["route_decision_counts"]
                rank_route_counts[route_reason] = (
                    rank_route_counts.get(route_reason, 0) + 1
                )
                route_count += 1
                pair_spill_count += int(
                    route_info.get("reason") == "prefix_hit_pair_spill"
                )
                replica_copy_route_count += int(
                    route_info.get("reason") == "prefix_hit_replica_copy"
                )
                placement_lease_route_count += int(
                    route_info.get("reason") == "placement_lease"
                )
                matched_blocks = int(route_info.get("matched_prefix_blocks", 0))
                route_matched_blocks += matched_blocks
                route_full_blocks += seq.num_tokens // seq.block_size
                routed_match_by_seq[seq.seq_id] = matched_blocks
                if route_info.get("uses_reclaimable_capacity", False):
                    reclaimable_capacity_routes += 1
                if route_info.get("prefix_hit"):
                    route_hits += 1
                    if target_rank in route_info.get("hit_summary", {}):
                        routed_to_prefix_owner += 1
            elif route_mode == "round_robin":
                # round-robin 模式只用于剥离 transfer 开销，不做全局路由打分
                target_rank = len(submit_times) % config["world_size"]
            # The launcher has already selected the destination worker. Keep
            # remote_gpu_id clear so the destination Scheduler treats this as a
            # local request and allocates local blocks before prefill.
            seq.remote_gpu_id = -1
            engine.send_queues[target_rank].put({"type": "sequence", "seq": seq})
            submit_times[seq.seq_id] = start
            if transfer_workload:
                if prompt_index < warmup_end:
                    phase_name = "warmup"
                elif prompt_index < pressure_end:
                    phase_name = "pressure"
                elif prompt_index < trigger_end:
                    phase_name = "trigger"
                else:
                    phase_name = "reuse"
                phase_by_seq[seq.seq_id] = phase_name
                phase_started_at.setdefault(phase_name, start)
                get_rank_stats(target_rank)[f"{phase_name}_submitted"] += 1
            get_rank_stats(target_rank)["submitted"] += 1
            inflight.add(seq.seq_id)

        while next_prompt_idx < current_phase_end and len(inflight) < effective_submit_window:
            submit_prompt(prompts[next_prompt_idx], next_prompt_idx)
            next_prompt_idx += 1

        # 主循环不断泵 worker 消息，直到所有请求都完成
        while finished_count < len(prompts):
            finished, first_tokens, prefill_stats, runtime_stats = engine.step()
            now = time.perf_counter()
            for item in runtime_stats:
                rank_data = get_rank_stats(item.get("rank", -1))
                if "max_cached_blocks" in item:
                    rank_data["max_cached_blocks"] = max(
                        int(rank_data.get("max_cached_blocks", 0)),
                        int(item["max_cached_blocks"]),
                    )
                transfer_count += int(item.get("transfer_count", item.get("swap_count", 0)))
                transfer_bytes += int(item.get("transfer_bytes", 0))
                transfer_time_s += float(item.get("transfer_time_s", 0.0))
                transfer_source_time_s += float(item.get("transfer_source_time_s", 0.0))
                transfer_target_time_s += float(item.get("transfer_target_time_s", 0.0))
                estimated_transfer_cost_ms += float(item.get("estimated_transfer_cost_ms", 0.0))
                estimated_saved_prefill_ms += float(item.get("estimated_saved_prefill_ms", 0.0))
                transfer_copy_count += int(item.get("transfer_copy_count", 0))
                transfer_release_count += int(item.get("transfer_release_count", 0))
                chain_transfer_count += int(item.get("chain_transfer_count", 0))
                transferred_hashes = item.get("transfer_hashes", [])
                hot_transferred = sum(
                    block_hash in hot_prefix_hashes for block_hash in transferred_hashes
                )
                hot_transfer_block_count += hot_transferred
                rebalance_success += int(item.get("rebalance_success", 0))
                rebalance_fail += int(item.get("rebalance_fail", 0))
                background_copy_success += int(item.get("background_copy_success", 0))
                background_copy_fail += int(item.get("background_copy_fail", 0))
                preemption_count += int(item.get("preemption_count", 0))
                rank_data["transfers"] += int(item.get("transfer_count", item.get("swap_count", 0)))
                rank_data["transfer_bytes"] += int(item.get("transfer_bytes", 0))
                rank_data["transfer_time_s"] += float(item.get("transfer_time_s", 0.0))
                rank_data["transfer_source_time_s"] += float(item.get("transfer_source_time_s", 0.0))
                rank_data["transfer_target_time_s"] += float(item.get("transfer_target_time_s", 0.0))
                rank_data["copies"] += int(item.get("transfer_copy_count", 0))
                rank_data["released_blocks"] += int(item.get("transfer_release_count", 0))
                rank_data["chain_transfers"] += int(item.get("chain_transfer_count", 0))
                rank_data["hot_transfer_blocks"] += hot_transferred
                rank_data["rebalance_success"] += int(item.get("rebalance_success", 0))
                rank_data["rebalance_fail"] += int(item.get("rebalance_fail", 0))
                rank_data["background_copy_success"] += int(item.get("background_copy_success", 0))
                rank_data["background_copy_fail"] += int(item.get("background_copy_fail", 0))
                rank_data["preemption_count"] += int(item.get("preemption_count", 0))
                if "free_blocks_after_release" in item:
                    rank_data["free_blocks_after_release"] = max(
                        int(rank_data.get("free_blocks_after_release", -1)),
                        int(item["free_blocks_after_release"]),
                    )
                rank_data["prefill_tokens"] += int(item.get("prefill_tokens", 0))
                rank_data["prefill_prompt_tokens"] += int(
                    item.get("prefill_prompt_tokens", item.get("prefill_tokens", 0))
                )
                rank_data["prefill_cached_tokens"] += int(
                    item.get("prefill_cached_tokens", 0)
                )
                rank_data["prefill_uncached_tokens"] += int(
                    item.get("prefill_uncached_tokens", item.get("prefill_tokens", 0))
                )
                rank_data["decode_tokens"] += int(item.get("decode_tokens", 0))
                rank_data["prefill_time_s"] += float(item.get("prefill_time_s", 0.0))
                rank_data["decode_time_s"] += float(item.get("decode_time_s", 0.0))
                rank_data["first_tokens"] += int(item.get("first_tokens", 0))
                rank_data["finished"] += int(item.get("finished", 0))
                rank_data["output_tokens"] += int(item.get("output_tokens", 0))
                for reason, count in item.get("rebalance_fail_reasons", {}).items():
                    rebalance_fail_reasons[reason] = rebalance_fail_reasons.get(reason, 0) + int(count)
                for diagnostic in item.get("rebalance_diagnostics", []):
                    if isinstance(diagnostic, dict):
                        rebalance_diagnostics.append(dict(diagnostic))
                for reason, count in item.get("background_copy_fail_reasons", {}).items():
                    background_copy_fail_reasons[reason] = (
                        background_copy_fail_reasons.get(reason, 0) + int(count)
                    )
            for seq_id, _token in first_tokens:
                if seq_id in submit_times:
                    emitted_at = engine.first_token_timestamps.pop(seq_id, now)
                    ttft = emitted_at - submit_times[seq_id]
                    ttfts.append(ttft)
                    first_token_at[seq_id] = emitted_at
                    phase = phase_by_seq.get(seq_id)
                    if phase in phase_ttfts:
                        phase_ttfts[phase].append(ttft)
                # first_tokens are grouped by worker in engine.step().
                # The rank is not attached to this tuple, so rank-level first-token
                # counts are reported from the worker runtime stats instead.
            for item in prefill_stats:
                rank_data = get_rank_stats(item.get("rank", -1))
                seq_id = item.get("seq_id")
                is_initial = bool(item.get("is_initial_prefill", item.get("prefill_attempt", 1) == 1))
                if seq_id is not None:
                    prefill_attempts += 1
                    rank_data["prefill_attempts"] += 1
                if seq_id is not None and is_initial:
                    prefill_seen_seq_ids.add(seq_id)
                    rank_data["prefill_requests"] += 1
                    cached_tokens = int(item.get("num_cached_tokens", 0))
                    prompt_tokens = int(item.get("num_prompt_tokens", 0))
                    initial_cached_tokens += cached_tokens
                    initial_prompt_tokens += prompt_tokens
                    rank_data["initial_cached_tokens"] += cached_tokens
                    rank_data["initial_prompt_tokens"] += prompt_tokens
                    if phase_by_seq.get(seq_id) == "reuse":
                        reuse_phase_seq_ids.add(seq_id)
                        reuse_phase_cached_tokens += cached_tokens
                        reuse_phase_prompt_tokens += prompt_tokens
                        if item.get("prefix_hit", False):
                            reuse_phase_hit_seq_ids.add(seq_id)
                    if routed_match_by_seq.get(seq_id, 0) > 0 and cached_tokens == 0:
                        stale_route_hits += 1
                if is_initial and item.get("prefix_hit", False):
                    if seq_id is not None:
                        prefill_hit_seq_ids.add(seq_id)
                        rank_data["prefill_prefix_hits"] += 1
            for seq_id, tokens in finished:
                inflight.discard(seq_id)
                finished_count += 1
                total_tokens += len(tokens)
                finished_at = engine.finished_timestamps.pop(seq_id, now)
                latency = finished_at - submit_times[seq_id]
                tpot = decode_tpot_s(
                    first_token_at.get(seq_id, finished_at),
                    finished_at,
                    len(tokens),
                )
                if tpot is not None:
                    tpots.append(tpot)
                e2es.append(latency)
                phase = phase_by_seq.get(seq_id)
                if phase in phase_e2es:
                    phase_e2es[phase].append(latency)
                    phase_output_tokens[phase] += len(tokens)
                    phase_finished_at[phase] = finished_at
                completion_times[seq_id] = finished_at
                completion_token_counts[seq_id] = len(tokens)
            for rank, stats in rank_stats.items():
                stats["local_prefix_hit_rate"] = (
                    stats["prefill_prefix_hits"] / max(stats["prefill_requests"], 1)
                )
                stats["initial_cached_token_ratio"] = (
                    stats["initial_cached_tokens"] / max(stats["initial_prompt_tokens"], 1)
                )
            if (
                transfer_workload
                and not inflight
                and next_prompt_idx >= current_phase_end
                and current_phase_index + 1 < len(phase_ends)
            ):
                should_flush_background = (
                    not bool(config.get("benchmark_memory_skew_proactive_move", False))
                    or workload not in {"memory-skew", "capacity-offload"}
                    or current_phase_index == 1
                )
                if (
                    config.get("enable_background_copy", False)
                    and engine.control_plane_client is not None
                    and should_flush_background
                ):
                    placement_started = time.perf_counter()
                    flush_result = engine.control_plane_client.flush_background_copies(
                        future_prefix_demands,
                        timeout_s=float(
                            config.get("background_copy_flush_timeout_s", 600.0)
                        ),
                    )
                    placement_wait_s += time.perf_counter() - placement_started
                    background_placement_stats = dict(
                        flush_result.get("placement_stats", {})
                    )
                    background_placement_pair_stats = {
                        str(pair): dict(stats)
                        for pair, stats in flush_result.get(
                            "placement_pair_stats", {}
                        ).items()
                    }
                elif engine.control_plane_client is not None:
                    # Foreground admission still needs the exact remaining
                    # ingress demand when proactive background placement is
                    # disabled. The synchronous acknowledgement establishes
                    # visibility before pressure requests reach workers.
                    engine.control_plane_client.publish_future_prefix_demands(
                        future_prefix_demands,
                        timeout_s=float(
                            config.get("background_copy_flush_timeout_s", 600.0)
                        ),
                    )
                current_phase_index += 1
                current_phase_end = phase_ends[current_phase_index]
            while next_prompt_idx < current_phase_end and len(inflight) < effective_submit_window:
                submit_prompt(prompts[next_prompt_idx], next_prompt_idx)
                next_prompt_idx += 1
            if now - last_progress_report >= 30.0:
                phase = phase_by_seq.get(next(iter(inflight)), "draining") if inflight else "draining"
                print(
                    f"[{name}] progress finished={finished_count}/{len(prompts)} "
                    f"submitted={next_prompt_idx}/{len(prompts)} inflight={len(inflight)} "
                    f"phase={phase} elapsed={now - start_wall:.1f}s",
                    flush=True,
                )
                last_progress_report = now
        transfer_placement_observations = []
        if engine.control_plane_client is not None:
            transfer_placement_observations = (
                engine.control_plane_client.get_transfer_cost_observations()
            )
        elapsed = time.perf_counter() - start_wall
    finally:
        if sampler_started:
            sampler.stop()
        engine.exit()
        if rendezvous_path is not None:
            rendezvous_path.unlink(missing_ok=True)

    gpu_util_mean, gpu_util_p95, gpu_mem_util_mean, gpu_mem_util_p95 = sampler.summarize()
    for rank, gpu_stats in sampler.summarize_by_rank().items():
        get_rank_stats(rank).update(gpu_stats)
    # goodput：只有在给定 e2e SLA 内完成的请求，才计入有效吞吐
    goodput_tokens = sum(
        completion_token_counts[seq_id] for seq_id, done_at in completion_times.items()
        if done_at - submit_times[seq_id] <= goodput_e2e_sla_s
    )
    goodput_sla_sweep_tok_s = compute_goodput_sla_sweep(
        completion_times,
        submit_times,
        completion_token_counts,
        elapsed,
        config.get("benchmark_goodput_sla_sweep_s", []),
    )

    return ScenarioResult(
        name=name,
        total_requests=len(prompts),
        total_tokens=total_tokens,
        elapsed_s=elapsed,
        throughput_tok_s=total_tokens / max(elapsed, 1e-9),
        goodput_tok_s=goodput_tokens / max(elapsed, 1e-9),
        mean_ttft_s=_mean(ttfts),
        p50_ttft_s=_median(ttfts),
        p90_ttft_s=_percentile(ttfts, 0.90),
        p95_ttft_s=_percentile(ttfts, 0.95),
        mean_tpot_s=_mean(tpots),
        p50_tpot_s=_median(tpots),
        p90_tpot_s=_percentile(tpots, 0.90),
        p95_tpot_s=_percentile(tpots, 0.95),
        mean_e2e_s=_mean(e2es),
        p50_e2e_s=_median(e2es),
        p90_e2e_s=_percentile(e2es, 0.90),
        p95_e2e_s=_percentile(e2es, 0.95),
        route_hit_rate=route_hits / max(route_count, 1),
        routed_to_prefix_owner_rate=routed_to_prefix_owner / max(route_count, 1),
        prefix_hit_rate=len(prefill_hit_seq_ids) / max(len(prefill_seen_seq_ids), 1),
        initial_cached_token_ratio=initial_cached_tokens / max(initial_prompt_tokens, 1),
        prefill_attempts=prefill_attempts,
        preemption_count=preemption_count,
        redundant_prefill_tokens=max(
            0,
            sum(int(stats["prefill_tokens"]) for stats in rank_stats.values()) - initial_prompt_tokens,
        ),
        transfer_count=transfer_count,
        transfer_bytes=transfer_bytes,
        transfer_time_s=transfer_time_s,
        transfer_source_time_s=transfer_source_time_s,
        transfer_target_time_s=transfer_target_time_s,
        transfer_bandwidth_gib_s=(
            transfer_bytes / (1024 ** 3) / transfer_time_s if transfer_time_s > 0 else 0.0
        ),
        estimated_transfer_cost_ms=estimated_transfer_cost_ms,
        estimated_saved_prefill_ms=estimated_saved_prefill_ms,
        transfer_copy_count=transfer_copy_count,
        transfer_release_count=transfer_release_count,
        chain_transfer_count=chain_transfer_count,
        hot_transfer_block_count=hot_transfer_block_count,
        hot_transfer_block_ratio=hot_transfer_block_count / max(transfer_count, 1),
        rebalance_success=rebalance_success,
        rebalance_fail=rebalance_fail,
        rebalance_fail_reasons=rebalance_fail_reasons,
        rebalance_diagnostics=rebalance_diagnostics,
        background_copy_success=background_copy_success,
        background_copy_fail=background_copy_fail,
        background_copy_fail_reasons=background_copy_fail_reasons,
        route_decision_counts=route_decision_counts,
        gpu_util_mean=gpu_util_mean,
        gpu_util_p95=gpu_util_p95,
        gpu_mem_util_mean=gpu_mem_util_mean,
        gpu_mem_util_p95=gpu_mem_util_p95,
        rank_stats=rank_stats,
        route_matched_block_ratio=route_matched_blocks / max(route_full_blocks, 1),
        reclaimable_capacity_route_rate=reclaimable_capacity_routes / max(route_count, 1),
        stale_route_hit_rate=stale_route_hits / max(route_hits, 1),
        reuse_phase_request_hit_rate=(
            len(reuse_phase_hit_seq_ids) / max(len(reuse_phase_seq_ids), 1)
        ),
        reuse_phase_token_ratio=(
            reuse_phase_cached_tokens / max(reuse_phase_prompt_tokens, 1)
        ),
        pressure_reuse_overlap_s=max(
            0.0,
            phase_finished_at.get("pressure", 0.0)
            - phase_started_at.get("reuse", float("inf")),
        ),
        phase_latency_stats={
            phase: {
                "requests": float(len(phase_e2es[phase])),
                "output_tokens": float(phase_output_tokens[phase]),
                "elapsed_s": max(
                    0.0,
                    phase_finished_at.get(phase, 0.0)
                    - phase_started_at.get(phase, 0.0),
                ),
                "throughput_tok_s": (
                    phase_output_tokens[phase]
                    / max(
                        phase_finished_at.get(phase, 0.0)
                        - phase_started_at.get(phase, 0.0),
                        1e-9,
                    )
                ),
                "mean_ttft_s": _mean(phase_ttfts[phase]),
                "p90_ttft_s": _percentile(phase_ttfts[phase], 0.90),
                "mean_e2e_s": _mean(phase_e2es[phase]),
                "p90_e2e_s": _percentile(phase_e2es[phase], 0.90),
            }
            for phase in ("warmup", "pressure", "trigger", "reuse")
            if phase_e2es[phase]
        },
        pair_spill_count=pair_spill_count,
        replica_copy_route_count=replica_copy_route_count,
        placement_lease_route_count=placement_lease_route_count,
        background_placement_stats=background_placement_stats,
        background_placement_pair_stats=background_placement_pair_stats,
        prefill_prompt_tokens=sum(
            int(stats["prefill_prompt_tokens"]) for stats in rank_stats.values()
        ),
        prefill_cached_tokens=sum(
            int(stats["prefill_cached_tokens"]) for stats in rank_stats.values()
        ),
        prefill_uncached_tokens=sum(
            int(stats["prefill_uncached_tokens"]) for stats in rank_stats.values()
        ),
        placement_wait_s=placement_wait_s,
        transfer_placement_observations=transfer_placement_observations,
        offload_verified=(
            workload in {"memory-skew", "capacity-offload"}
            and transfer_release_count > 0
        ),
        transfer_cost_observation_count=len(transfer_placement_observations),
        transfer_cost_mae_ms=_mean([
            abs(
                float(observation["predicted_cost_ms"])
                - float(observation["elapsed_ms"])
            )
            for observation in transfer_placement_observations
        ]),
        transfer_cost_p95_abs_error_ms=_percentile([
            abs(
                float(observation["predicted_cost_ms"])
                - float(observation["elapsed_ms"])
            )
            for observation in transfer_placement_observations
        ], 0.95),
        transfer_cost_underprediction_rate=(
            sum(
                float(observation["predicted_cost_ms"])
                < float(observation["elapsed_ms"])
                for observation in transfer_placement_observations
            )
            / max(len(transfer_placement_observations), 1)
        ),
        goodput_sla_sweep_tok_s=goodput_sla_sweep_tok_s,
    )


def make_config(
    world_size: int,
    enable_global_pool: bool,
    nvlink_pairs: list[tuple[int, int]] | None,
    base_config: dict | None = None,
) -> dict:
    # benchmark 里统一通过这层构造 config，避免每个场景单独拼参数时漏掉关键项
    config = dict(base_config or MODEL_CONFIG)
    config["world_size"] = world_size
    config["enable_global_pool"] = enable_global_pool
    if nvlink_pairs is not None:
        config["nvlink_topo"] = {"pairs": nvlink_pairs}
    config["use_control_plane_process"] = enable_global_pool
    return config


_STUDENT_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def confidence_interval_95(values: Iterable[float]) -> float:
    """Return the two-sided 95% Student-t half-width for repeated runs."""
    samples = [float(value) for value in values]
    if len(samples) < 2:
        return 0.0
    critical = _STUDENT_T_975.get(len(samples) - 1, 1.96)
    return critical * statistics.stdev(samples) / (len(samples) ** 0.5)


def _trial_payload(result: ScenarioResult) -> dict:
    payload = asdict(result)
    payload.pop("trial_results", None)
    return payload


def aggregate_scenario_trials(trials: list[ScenarioResult]) -> ScenarioResult:
    """Return per-scenario means and key run-to-run standard deviations."""
    if not trials:
        raise ValueError("at least one scenario trial is required")
    if len(trials) == 1:
        trials[0].repetitions = 1
        trials[0].goodput_sla_sweep_tok_s_ci95 = {
            key: 0.0
            for key in (trials[0].goodput_sla_sweep_tok_s or {})
        }
        trials[0].trial_results = [_trial_payload(trials[0])]
        return trials[0]

    def mean_attr(name: str) -> float:
        return statistics.fmean(float(getattr(result, name)) for result in trials)

    def mean_reason_map(name: str) -> dict[str, int]:
        keys = set().union(*((getattr(result, name) or {}).keys() for result in trials))
        return {
            key: round(statistics.fmean(
                (getattr(result, name) or {}).get(key, 0) for result in trials
            ))
            for key in keys
        }

    def mean_nested_count_map(name: str) -> dict[str, dict[str, int]]:
        outer_keys = set().union(*(
            (getattr(result, name) or {}).keys() for result in trials
        ))
        aggregated = {}
        for outer_key in outer_keys:
            inner_keys = set().union(*(
                (getattr(result, name) or {}).get(outer_key, {}).keys()
                for result in trials
            ))
            aggregated[outer_key] = {
                inner_key: round(statistics.fmean(
                    (getattr(result, name) or {}).get(outer_key, {}).get(inner_key, 0)
                    for result in trials
                ))
                for inner_key in inner_keys
            }
        return aggregated

    def aggregate_float_map(name: str) -> tuple[dict[str, float], dict[str, float]]:
        keys = set().union(*(
            (getattr(result, name) or {}).keys() for result in trials
        ))
        means = {}
        ci95 = {}
        for key in keys:
            values = [
                float((getattr(result, name) or {}).get(key, 0.0))
                for result in trials
            ]
            means[key] = statistics.fmean(values)
            ci95[key] = confidence_interval_95(values)
        return means, ci95

    rank_ids = sorted(set().union(*(result.rank_stats.keys() for result in trials)))
    rank_stats = {}
    for rank in rank_ids:
        keys = set().union(*(result.rank_stats.get(rank, {}).keys() for result in trials))
        rank_stats[rank] = {}
        for key in keys:
            values = [result.rank_stats.get(rank, {}).get(key, 0.0) for result in trials]
            if all(isinstance(value, (int, float)) for value in values):
                rank_stats[rank][key] = statistics.fmean(float(value) for value in values)
            elif all(isinstance(value, dict) for value in values):
                nested_keys = set().union(*(value.keys() for value in values))
                rank_stats[rank][key] = {
                    nested_key: (
                        int(statistics.fmean(
                            float(value.get(nested_key, 0)) for value in values
                        ) + 0.5)
                        if key == "route_decision_counts"
                        else round(statistics.fmean(
                            float(value.get(nested_key, 0)) for value in values
                        ))
                    )
                    for nested_key in nested_keys
                }

    def mean_count_map(name: str) -> dict[str, int]:
        keys = set().union(*((getattr(result, name) or {}).keys() for result in trials))
        return {
            key: int(statistics.fmean(
                (getattr(result, name) or {}).get(key, 0) for result in trials
            ) + 0.5)
            for key in keys
        }

    phase_names = sorted(set().union(*(
        set((result.phase_latency_stats or {}).keys())
        for result in trials
    )))
    phase_latency_stats = {}
    phase_uncertainty_metrics = {
        "throughput_tok_s",
        "mean_ttft_s",
        "p90_ttft_s",
        "mean_e2e_s",
        "p90_e2e_s",
    }
    for phase in phase_names:
        metric_names = set().union(*(
            set((result.phase_latency_stats or {}).get(phase, {}).keys())
            for result in trials
        ))
        phase_latency_stats[phase] = {}
        for metric in metric_names:
            values = [
                float((result.phase_latency_stats or {}).get(phase, {}).get(metric, 0.0))
                for result in trials
            ]
            phase_latency_stats[phase][metric] = statistics.fmean(values)
            if metric in phase_uncertainty_metrics:
                phase_latency_stats[phase][f"{metric}_std"] = statistics.stdev(values)
                phase_latency_stats[phase][f"{metric}_ci95"] = confidence_interval_95(values)

    goodput_sla_sweep, goodput_sla_sweep_ci95 = aggregate_float_map(
        "goodput_sla_sweep_tok_s"
    )

    return ScenarioResult(
        name=trials[0].name,
        total_requests=round(mean_attr("total_requests")),
        total_tokens=round(mean_attr("total_tokens")),
        elapsed_s=mean_attr("elapsed_s"),
        throughput_tok_s=mean_attr("throughput_tok_s"),
        goodput_tok_s=mean_attr("goodput_tok_s"),
        mean_ttft_s=mean_attr("mean_ttft_s"),
        p50_ttft_s=mean_attr("p50_ttft_s"),
        p90_ttft_s=mean_attr("p90_ttft_s"),
        p95_ttft_s=mean_attr("p95_ttft_s"),
        mean_tpot_s=mean_attr("mean_tpot_s"),
        p50_tpot_s=mean_attr("p50_tpot_s"),
        p90_tpot_s=mean_attr("p90_tpot_s"),
        p95_tpot_s=mean_attr("p95_tpot_s"),
        mean_e2e_s=mean_attr("mean_e2e_s"),
        p50_e2e_s=mean_attr("p50_e2e_s"),
        p90_e2e_s=mean_attr("p90_e2e_s"),
        p95_e2e_s=mean_attr("p95_e2e_s"),
        route_hit_rate=mean_attr("route_hit_rate"),
        routed_to_prefix_owner_rate=mean_attr("routed_to_prefix_owner_rate"),
        prefix_hit_rate=mean_attr("prefix_hit_rate"),
        initial_cached_token_ratio=mean_attr("initial_cached_token_ratio"),
        prefill_attempts=round(mean_attr("prefill_attempts")),
        preemption_count=round(mean_attr("preemption_count")),
        redundant_prefill_tokens=round(mean_attr("redundant_prefill_tokens")),
        transfer_count=round(mean_attr("transfer_count")),
        transfer_bytes=round(mean_attr("transfer_bytes")),
        transfer_time_s=mean_attr("transfer_time_s"),
        transfer_source_time_s=mean_attr("transfer_source_time_s"),
        transfer_target_time_s=mean_attr("transfer_target_time_s"),
        transfer_bandwidth_gib_s=mean_attr("transfer_bandwidth_gib_s"),
        estimated_transfer_cost_ms=mean_attr("estimated_transfer_cost_ms"),
        estimated_saved_prefill_ms=mean_attr("estimated_saved_prefill_ms"),
        transfer_copy_count=round(mean_attr("transfer_copy_count")),
        transfer_release_count=round(mean_attr("transfer_release_count")),
        chain_transfer_count=round(mean_attr("chain_transfer_count")),
        hot_transfer_block_count=round(mean_attr("hot_transfer_block_count")),
        hot_transfer_block_ratio=mean_attr("hot_transfer_block_ratio"),
        rebalance_success=round(mean_attr("rebalance_success")),
        rebalance_fail=round(mean_attr("rebalance_fail")),
        rebalance_fail_reasons=mean_reason_map("rebalance_fail_reasons"),
        rebalance_diagnostics=[
            dict(diagnostic)
            for trial in trials
            for diagnostic in (trial.rebalance_diagnostics or [])
        ],
        route_decision_counts=mean_count_map("route_decision_counts"),
        background_copy_success=round(mean_attr("background_copy_success")),
        background_copy_fail=round(mean_attr("background_copy_fail")),
        background_copy_fail_reasons=mean_reason_map("background_copy_fail_reasons"),
        gpu_util_mean=mean_attr("gpu_util_mean"),
        gpu_util_p95=mean_attr("gpu_util_p95"),
        gpu_mem_util_mean=mean_attr("gpu_mem_util_mean"),
        gpu_mem_util_p95=mean_attr("gpu_mem_util_p95"),
        rank_stats=rank_stats,
        trace_request_share_rate=mean_attr("trace_request_share_rate"),
        trace_token_share_ratio=mean_attr("trace_token_share_ratio"),
        theoretical_prefix_hit_rate=mean_attr("theoretical_prefix_hit_rate"),
        route_matched_block_ratio=mean_attr("route_matched_block_ratio"),
        reclaimable_capacity_route_rate=mean_attr("reclaimable_capacity_route_rate"),
        stale_route_hit_rate=mean_attr("stale_route_hit_rate"),
        reuse_phase_request_hit_rate=mean_attr("reuse_phase_request_hit_rate"),
        reuse_phase_token_ratio=mean_attr("reuse_phase_token_ratio"),
        pressure_reuse_overlap_s=mean_attr("pressure_reuse_overlap_s"),
        repetitions=len(trials),
        throughput_tok_s_std=statistics.stdev(result.throughput_tok_s for result in trials),
        goodput_tok_s_std=statistics.stdev(result.goodput_tok_s for result in trials),
        mean_ttft_s_std=statistics.stdev(result.mean_ttft_s for result in trials),
        mean_tpot_s_std=statistics.stdev(result.mean_tpot_s for result in trials),
        mean_e2e_s_std=statistics.stdev(result.mean_e2e_s for result in trials),
        p90_e2e_s_std=statistics.stdev(result.p90_e2e_s for result in trials),
        throughput_tok_s_ci95=confidence_interval_95(
            result.throughput_tok_s for result in trials
        ),
        goodput_tok_s_ci95=confidence_interval_95(
            result.goodput_tok_s for result in trials
        ),
        mean_ttft_s_ci95=confidence_interval_95(
            result.mean_ttft_s for result in trials
        ),
        mean_tpot_s_ci95=confidence_interval_95(
            result.mean_tpot_s for result in trials
        ),
        mean_e2e_s_ci95=confidence_interval_95(
            result.mean_e2e_s for result in trials
        ),
        p90_e2e_s_ci95=confidence_interval_95(
            result.p90_e2e_s for result in trials
        ),
        trial_results=[_trial_payload(result) for result in trials],
        phase_latency_stats=phase_latency_stats,
        pair_spill_count=round(mean_attr("pair_spill_count")),
        replica_copy_route_count=round(mean_attr("replica_copy_route_count")),
        placement_lease_route_count=round(mean_attr("placement_lease_route_count")),
        background_placement_stats=mean_reason_map("background_placement_stats"),
        background_placement_pair_stats=mean_nested_count_map(
            "background_placement_pair_stats"
        ),
        prefill_prompt_tokens=round(mean_attr("prefill_prompt_tokens")),
        prefill_cached_tokens=round(mean_attr("prefill_cached_tokens")),
        prefill_uncached_tokens=round(mean_attr("prefill_uncached_tokens")),
        placement_wait_s=mean_attr("placement_wait_s"),
        transfer_placement_observations=[
            dict(observation)
            for trial in trials
            for observation in (trial.transfer_placement_observations or [])
        ],
        offload_verified=all(result.offload_verified for result in trials),
        transfer_cost_observation_count=round(
            mean_attr("transfer_cost_observation_count")
        ),
        transfer_cost_mae_ms=mean_attr("transfer_cost_mae_ms"),
        transfer_cost_p95_abs_error_ms=mean_attr(
            "transfer_cost_p95_abs_error_ms"
        ),
        transfer_cost_underprediction_rate=mean_attr(
            "transfer_cost_underprediction_rate"
        ),
        goodput_sla_sweep_tok_s=goodput_sla_sweep,
        goodput_sla_sweep_tok_s_ci95=goodput_sla_sweep_ci95,
    )


def run_repeated_engine_scenario(repetitions: int, **kwargs) -> ScenarioResult:
    trials = []
    for trial in range(repetitions):
        if repetitions > 1:
            print(f"[{kwargs['name']}] trial {trial + 1}/{repetitions}")
        trial_start = time.perf_counter()
        trials.append(run_engine_scenario(**kwargs))
        if repetitions > 1:
            print(
                f"[{kwargs['name']}] trial {trial + 1}/{repetitions} completed "
                f"in {time.perf_counter() - trial_start:.1f}s",
                flush=True,
            )
    return aggregate_scenario_trials(trials)


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_summary_table(
    results: list[ScenarioResult | None],
    title: str = "Benchmark Summary",
):
    # 横向总表：把所有场景放在同一张表里，便于直接看五种配置的整体差异。
    valid_results = [result for result in results if result is not None]
    print(f"\n{title}")
    print("=" * 235)
    print(
        f"{'scenario':<22} {'tput(tok/s)':>14} {'goodput':>12} {'ttft(ms)':>12} {'tpot(ms)':>12} "
        f"{'e2e(ms)':>12} {'p90(e2e)':>12} {'p95(e2e)':>12} {'gpu util':>10} {'mem util':>10} "
        f"{'CP req hit':>11} {'CP owner':>11} {'DP req hit':>11} {'DP tok reuse':>12} "
        f"{'attempts':>9} {'preempt':>8} {'redund tok':>10} {'sent blk':>9} "
        f"{'copied':>8} {'released':>9} "
        f"{'fg ok':>7} {'fg fail':>8} {'bg ok':>7} {'bg fail':>8} "
        f"{'pinned':>8} {'no space':>9} {'no plan':>8} {'low value':>9} "
        f"{'target busy':>11} {'bg space':>8}"
    )
    for result in valid_results:
        print(
            f"{result.name:<22} "
            f"{result.throughput_tok_s:>14.2f} "
            f"{result.goodput_tok_s:>12.2f} "
            f"{result.mean_ttft_s * 1000:>12.2f} "
            f"{result.mean_tpot_s * 1000:>12.2f} "
            f"{result.mean_e2e_s * 1000:>12.2f} "
            f"{result.p90_e2e_s * 1000:>12.2f} "
            f"{result.p95_e2e_s * 1000:>12.2f} "
            f"{(result.gpu_util_mean if result.gpu_util_mean is not None else float('nan')):>10.2f} "
            f"{(result.gpu_mem_util_mean if result.gpu_mem_util_mean is not None else float('nan')):>10.2f} "
            f"{fmt_pct(result.route_hit_rate):>11} "
            f"{fmt_pct(result.routed_to_prefix_owner_rate):>11} "
            f"{fmt_pct(result.prefix_hit_rate):>11} "
            f"{fmt_pct(result.initial_cached_token_ratio):>12} "
            f"{result.prefill_attempts:>9} "
            f"{result.preemption_count:>8} "
            f"{result.redundant_prefill_tokens:>10} "
            f"{result.transfer_count:>9} "
            f"{result.transfer_copy_count:>8} "
            f"{result.transfer_release_count:>9} "
            f"{result.rebalance_success:>7} "
            f"{result.rebalance_fail:>8} "
            f"{result.background_copy_success:>7} "
            f"{result.background_copy_fail:>8} "
            f"{result.rebalance_fail_reasons.get('pinned_source', 0):>8} "
            f"{result.rebalance_fail_reasons.get('no_target_space', 0):>9} "
            f"{result.rebalance_fail_reasons.get('no_plan', 0):>8} "
            f"{result.rebalance_fail_reasons.get('low_benefit', 0):>9} "
            f"{result.rebalance_fail_reasons.get('target_busy', 0):>11} "
            f"{result.background_copy_fail_reasons.get('no_target_space', 0):>8}"
        )

    print("\nPrefix Diagnostics")
    print("=" * 158)
    print(
        f"{'scenario':<22} {'trace req share':>15} {'trace tok share':>15} {'CP blk match':>13} "
        f"{'CP req hit':>11} {'CP reclaim':>12} {'CP stale':>12} {'DP req hit':>11} "
        f"{'DP tok reuse':>12} {'actual blocks/rank':>20}"
    )
    for result in valid_results:
        block_caps = ",".join(
            str(int(stats.get("max_cached_blocks", 0)))
            for _rank, stats in sorted(result.rank_stats.items(), key=lambda item: int(item[0]))
        ) or "n/a"
        print(
            f"{result.name:<22} "
            f"{fmt_pct(result.trace_request_share_rate):>15} "
            f"{fmt_pct(result.trace_token_share_ratio):>15} "
            f"{fmt_pct(result.route_matched_block_ratio):>13} "
            f"{fmt_pct(result.route_hit_rate):>11} "
            f"{fmt_pct(result.reclaimable_capacity_route_rate):>12} "
            f"{fmt_pct(result.stale_route_hit_rate):>12} "
            f"{fmt_pct(result.prefix_hit_rate):>11} "
            f"{fmt_pct(result.initial_cached_token_ratio):>12} "
            f"{block_caps:>20}"
        )

    print("\nTransfer Diagnostics")
    print("=" * 263)
    print(
        f"{'scenario':<22} {'sent blocks':>12} {'source kept':>12} "
        f"{'source freed':>13} {'chain plans':>12} {'hot sent':>10} {'hot ratio':>11} "
        f"{'reuse req hit':>14} {'reuse tok ratio':>15} {'GiB sent':>10} "
        f"{'src ms':>10} {'dst ms':>10} {'eff GiB/s':>11} {'est cost':>10} {'est save':>10} "
        f"{'fg ok':>8} {'fg fail':>9} {'spill':>8} {'copy route':>11} {'lease route':>12} "
        f"{'place q':>9} {'place cand':>11} {'place done':>11} "
        f"{'plan run':>9} {'plan done':>10}"
    )
    for result in valid_results:
        print(
            f"{result.name:<22} "
            f"{result.transfer_count:>12} "
            f"{result.transfer_copy_count:>12} "
            f"{result.transfer_release_count:>13} "
            f"{result.chain_transfer_count:>12} "
            f"{result.hot_transfer_block_count:>10} "
            f"{fmt_pct(result.hot_transfer_block_ratio):>11} "
            f"{fmt_pct(result.reuse_phase_request_hit_rate):>14} "
            f"{fmt_pct(result.reuse_phase_token_ratio):>15} "
            f"{result.transfer_bytes / (1024 ** 3):>10.3f} "
            f"{result.transfer_source_time_s * 1000:>10.2f} "
            f"{result.transfer_target_time_s * 1000:>10.2f} "
            f"{result.transfer_bandwidth_gib_s:>11.2f} "
            f"{result.estimated_transfer_cost_ms:>10.2f} "
            f"{result.estimated_saved_prefill_ms:>10.2f} "
            f"{result.rebalance_success:>8} "
            f"{result.rebalance_fail:>9} "
            f"{result.pair_spill_count:>8} "
            f"{result.replica_copy_route_count:>11} "
            f"{result.placement_lease_route_count:>12} "
            f"{(result.background_placement_stats or {}).get('queued', 0):>9} "
            f"{(result.background_placement_stats or {}).get('dispatched', 0):>11} "
            f"{(result.background_placement_stats or {}).get('completed', 0):>11} "
            f"{(result.background_placement_stats or {}).get('plans_dispatched', 0):>9} "
            f"{(result.background_placement_stats or {}).get('plans_completed', 0):>10}"
        )

    observed_results = [
        result
        for result in valid_results
        if result.transfer_cost_observation_count > 0
    ]
    if observed_results:
        print("\nTransfer Cost Prediction")
        print("=" * 98)
        print(
            f"{'scenario':<22} {'observations':>14} {'MAE(ms)':>12} "
            f"{'P95 abs err(ms)':>17} {'underpredict':>15} {'offload':>10}"
        )
        for result in observed_results:
            print(
                f"{result.name:<22} "
                f"{result.transfer_cost_observation_count:>14} "
                f"{result.transfer_cost_mae_ms:>12.2f} "
                f"{result.transfer_cost_p95_abs_error_ms:>17.2f} "
                f"{fmt_pct(result.transfer_cost_underprediction_rate):>15} "
                f"{str(result.offload_verified):>10}"
            )

    print("\nPrefill Compute Diagnostics")
    print("=" * 118)
    print(
        f"{'scenario':<22} {'prompt tok':>12} {'cached tok':>12} "
        f"{'uncached tok':>14} {'compute reuse':>14} {'prefill wall(s)':>16} "
        f"{'placement wait(s)':>18}"
    )
    for result in valid_results:
        prefill_wall_s = sum(
            float(stats.get("prefill_time_s", 0.0))
            for stats in result.rank_stats.values()
        )
        compute_reuse = (
            result.prefill_cached_tokens / max(result.prefill_prompt_tokens, 1)
        )
        print(
            f"{result.name:<22} "
            f"{result.prefill_prompt_tokens:>12} "
            f"{result.prefill_cached_tokens:>12} "
            f"{result.prefill_uncached_tokens:>14} "
            f"{fmt_pct(compute_reuse):>14} "
            f"{prefill_wall_s:>16.3f} "
            f"{result.placement_wait_s:>18.3f}"
        )

    placement_rows = [
        (result.name, pair, stats)
        for result in valid_results
        for pair, stats in sorted(
            (result.background_placement_pair_stats or {}).items()
        )
    ]
    if placement_rows:
        print("\nPer-Pair Placement Diagnostics")
        print("=" * 142)
        print(
            f"{'scenario':<22} {'pair':>8} {'queued':>9} {'evaluated':>11} "
            f"{'candidates':>12} {'completed':>11} {'plans':>8} {'plan done':>10} "
            f"{'low benefit':>13} "
            f"{'no space':>10} {'neg-cache':>11}"
        )
        for scenario, pair, stats in placement_rows:
            print(
                f"{scenario:<22} {pair:>8} "
                f"{stats.get('queued', 0):>9} "
                f"{stats.get('evaluated', 0):>11} "
                f"{stats.get('dispatched', 0):>12} "
                f"{stats.get('completed', 0):>11} "
                f"{stats.get('plans_dispatched', 0):>8} "
                f"{stats.get('plans_completed', 0):>10} "
                f"{stats.get('dropped_low_benefit', 0):>13} "
                f"{stats.get('dropped_no_target_space', 0):>10} "
                f"{stats.get('skipped_negative_cache', 0):>11}"
            )

    if any(result.phase_latency_stats for result in valid_results):
        print("\nTransfer Workload Phase Latency")
        print("=" * 122)
        print(
            f"{'scenario':<22} {'phase':<10} {'requests':>10} {'tput(tok/s)':>13} "
            f"{'mean TTFT(ms)':>15} {'p90 TTFT(ms)':>15} "
            f"{'mean E2E(ms)':>15} {'p90 E2E(ms)':>15}"
        )
        for result in valid_results:
            for phase in ("warmup", "pressure", "trigger", "reuse"):
                stats = (result.phase_latency_stats or {}).get(phase)
                if not stats:
                    continue
                print(
                    f"{result.name:<22} {phase:<10} "
                    f"{int(round(stats.get('requests', 0.0))):>10} "
                    f"{stats.get('throughput_tok_s', 0.0):>13.2f} "
                    f"{stats.get('mean_ttft_s', 0.0) * 1000:>15.2f} "
                    f"{stats.get('p90_ttft_s', 0.0) * 1000:>15.2f} "
                    f"{stats.get('mean_e2e_s', 0.0) * 1000:>15.2f} "
                    f"{stats.get('p90_e2e_s', 0.0) * 1000:>15.2f}"
                )

    if any(result.repetitions > 1 for result in valid_results):
        print("\nRepeated-run variability (mean +/- 95% CI; sample stddev is in JSON)")
        print(
            f"{'scenario':<22} {'throughput(tok/s)':>24} {'goodput(tok/s)':>24} "
            f"{'TTFT(ms)':>24} {'decode TPOT(ms)':>24} {'E2E(ms)':>24} "
            f"{'P90 E2E(ms)':>24}"
        )
        for result in valid_results:
            print(
                f"{result.name:<22} "
                f"{result.throughput_tok_s:>10.2f} +/- {result.throughput_tok_s_ci95:<8.2f} "
                f"{result.goodput_tok_s:>10.2f} +/- {result.goodput_tok_s_ci95:<8.2f} "
                f"{result.mean_ttft_s * 1000:>10.2f} +/- {result.mean_ttft_s_ci95 * 1000:<8.2f} "
                f"{result.mean_tpot_s * 1000:>10.2f} +/- {result.mean_tpot_s_ci95 * 1000:<8.2f} "
                f"{result.mean_e2e_s * 1000:>10.2f} +/- {result.mean_e2e_s_ci95 * 1000:<8.2f} "
                f"{result.p90_e2e_s * 1000:>10.2f} +/- {result.p90_e2e_s_ci95 * 1000:<8.2f}"
            )


def save_summary_figure(
    results: list[ScenarioResult | None],
    output_path: str,
    title: str = "Benchmark Summary",
) -> None:
    # 生成一张总览图：吞吐 / goodput、延迟、prefix hit、GPU 利用率分别放在不同子图。
    valid_results = [result for result in results if result is not None]
    if not valid_results:
        return

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [result.name for result in valid_results]
    x = list(range(len(valid_results)))

    throughput = [result.throughput_tok_s for result in valid_results]
    goodput = [result.goodput_tok_s for result in valid_results]
    throughput_ci = [result.throughput_tok_s_ci95 for result in valid_results]
    goodput_ci = [result.goodput_tok_s_ci95 for result in valid_results]
    ttft_ms = [result.mean_ttft_s * 1000.0 for result in valid_results]
    tpot_ms = [result.mean_tpot_s * 1000.0 for result in valid_results]
    e2e_ms = [result.mean_e2e_s * 1000.0 for result in valid_results]
    p90_e2e_ms = [result.p90_e2e_s * 1000.0 for result in valid_results]
    ttft_ci_ms = [result.mean_ttft_s_ci95 * 1000.0 for result in valid_results]
    tpot_ci_ms = [result.mean_tpot_s_ci95 * 1000.0 for result in valid_results]
    e2e_ci_ms = [result.mean_e2e_s_ci95 * 1000.0 for result in valid_results]
    p90_e2e_ci_ms = [result.p90_e2e_s_ci95 * 1000.0 for result in valid_results]
    route_hit_pct = [result.route_hit_rate * 100.0 for result in valid_results]
    owner_hit_pct = [result.routed_to_prefix_owner_rate * 100.0 for result in valid_results]
    local_hit_pct = [result.prefix_hit_rate * 100.0 for result in valid_results]
    cached_token_pct = [result.initial_cached_token_ratio * 100.0 for result in valid_results]
    gpu_util = [result.gpu_util_mean if result.gpu_util_mean is not None else 0.0 for result in valid_results]
    gpu_mem_util = [result.gpu_mem_util_mean if result.gpu_mem_util_mean is not None else 0.0 for result in valid_results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16)
    palettes = {
        "throughput": ["#2684FC", "#00AC47"],
        "latency": ["#2684FC", "#00AC47", "#EA4335", "#FBBC04"],
        "hit": ["#2684FC", "#00AC47", "#FBBC04", "#EA4335"],
        "util": ["#A142F4", "#00A3A3"],
    }
    bar_style = {"edgecolor": "#333333", "linewidth": 0.45}
    error_style = {"elinewidth": 0.8, "ecolor": "#222222", "capthick": 0.8}

    def annotate_bars(ax, bars, suffix: str = "", decimals: int = 1):
        max_height = 0.0
        for bar in bars:
            height = bar.get_height()
            max_height = max(max_height, height)
            label = f"{height:.{decimals}f}{suffix}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )
        if max_height > 0:
            top = ax.get_ylim()[1]
            ax.set_ylim(top=max(top, max_height * 1.18))

    width = 0.38
    bars = axes[0, 0].bar(
        [i - width / 2 for i in x],
        throughput,
        yerr=throughput_ci,
        capsize=3,
        error_kw=error_style,
        width=width,
        label="throughput",
        color=palettes["throughput"][0],
        **bar_style,
    )
    annotate_bars(axes[0, 0], bars)
    bars = axes[0, 0].bar(
        [i + width / 2 for i in x],
        goodput,
        yerr=goodput_ci,
        capsize=3,
        error_kw=error_style,
        width=width,
        label="goodput",
        color=palettes["throughput"][1],
        **bar_style,
    )
    annotate_bars(axes[0, 0], bars)
    axes[0, 0].set_title("Throughput / Goodput")
    axes[0, 0].set_ylabel("tokens/s")
    axes[0, 0].set_xticks(x, names, rotation=15, ha="right")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.25)

    latency_width = 0.2
    bars = axes[0, 1].bar(
        [i - 1.5 * latency_width for i in x],
        ttft_ms,
        yerr=ttft_ci_ms,
        capsize=3,
        error_kw=error_style,
        width=latency_width,
        label="TTFT mean",
        color=palettes["latency"][0],
        **bar_style,
    )
    annotate_bars(axes[0, 1], bars, decimals=0)
    bars = axes[0, 1].bar(
        [i - 0.5 * latency_width for i in x],
        tpot_ms,
        yerr=tpot_ci_ms,
        capsize=3,
        error_kw=error_style,
        width=latency_width,
        label="Decode TPOT mean",
        color=palettes["latency"][1],
        **bar_style,
    )
    annotate_bars(axes[0, 1], bars, decimals=0)
    bars = axes[0, 1].bar(
        [i + 0.5 * latency_width for i in x],
        e2e_ms,
        yerr=e2e_ci_ms,
        capsize=3,
        error_kw=error_style,
        width=latency_width,
        label="E2E mean",
        color=palettes["latency"][2],
        **bar_style,
    )
    annotate_bars(axes[0, 1], bars, decimals=0)
    bars = axes[0, 1].bar(
        [i + 1.5 * latency_width for i in x],
        p90_e2e_ms,
        yerr=p90_e2e_ci_ms,
        capsize=3,
        error_kw=error_style,
        width=latency_width,
        label="E2E p90",
        color=palettes["latency"][3],
        **bar_style,
    )
    annotate_bars(axes[0, 1], bars, decimals=0)
    axes[0, 1].set_title("Latency")
    axes[0, 1].set_ylabel("ms")
    axes[0, 1].set_xticks(x, names, rotation=15, ha="right")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.25)

    hit_width = 0.2
    bars = axes[1, 0].bar(
        [i - 1.5 * hit_width for i in x],
        route_hit_pct,
        width=hit_width,
        label="CP request hit",
        color=palettes["hit"][0],
        **bar_style,
    )
    annotate_bars(axes[1, 0], bars, suffix="%", decimals=1)
    bars = axes[1, 0].bar(
        [i - 0.5 * hit_width for i in x],
        owner_hit_pct,
        width=hit_width,
        label="CP owner selected",
        color=palettes["hit"][1],
        **bar_style,
    )
    annotate_bars(axes[1, 0], bars, suffix="%", decimals=1)
    bars = axes[1, 0].bar(
        [i + 0.5 * hit_width for i in x],
        local_hit_pct,
        width=hit_width,
        label="DP request hit",
        color=palettes["hit"][2],
        **bar_style,
    )
    annotate_bars(axes[1, 0], bars, suffix="%", decimals=1)
    bars = axes[1, 0].bar(
        [i + 1.5 * hit_width for i in x],
        cached_token_pct,
        width=hit_width,
        label="DP token reuse",
        color=palettes["hit"][3],
        **bar_style,
    )
    annotate_bars(axes[1, 0], bars, suffix="%", decimals=1)
    axes[1, 0].set_title("Prefix Reuse Metrics")
    axes[1, 0].set_ylabel("%")
    axes[1, 0].set_xticks(x, names, rotation=15, ha="right")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.25)

    bars = axes[1, 1].bar(
        [i - width / 2 for i in x],
        gpu_util,
        width=width,
        label="GPU util",
        color=palettes["util"][0],
        **bar_style,
    )
    annotate_bars(axes[1, 1], bars, suffix="%", decimals=1)
    bars = axes[1, 1].bar(
        [i + width / 2 for i in x],
        gpu_mem_util,
        width=width,
        label="GPU mem util",
        color=palettes["util"][1],
        **bar_style,
    )
    annotate_bars(axes[1, 1], bars, suffix="%", decimals=1)
    axes[1, 1].set_title("GPU Utilization")
    axes[1, 1].set_ylabel("%")
    axes[1, 1].set_xticks(x, names, rotation=15, ha="right")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {output}")


def save_reuse_phase_figure(
    results: list[ScenarioResult | None],
    output_path: str,
    title: str = "Benchmark",
) -> None:
    """Plot the reuse phase separately from source-build and pressure traffic."""
    valid_results = [
        result
        for result in results
        if result is not None and (result.phase_latency_stats or {}).get("reuse")
    ]
    if not valid_results:
        return

    summary_output = Path(output_path)
    output = summary_output.with_name(
        f"{summary_output.stem}_reuse_phase{summary_output.suffix}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [result.name for result in valid_results]
    x = list(range(len(valid_results)))
    reuse_stats = [result.phase_latency_stats["reuse"] for result in valid_results]
    metrics = [
        ("Reuse Throughput", "tokens/s", "throughput_tok_s", 1.0, "#2684FC"),
        ("Reuse Mean TTFT", "ms", "mean_ttft_s", 1000.0, "#00AC47"),
        ("Reuse Mean E2E", "ms", "mean_e2e_s", 1000.0, "#EA4335"),
        ("Reuse P90 E2E", "ms", "p90_e2e_s", 1000.0, "#FBBC04"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(f"{title}: Reuse Phase", fontsize=16)
    for ax, (title, ylabel, key, scale, color) in zip(axes.flat, metrics):
        values = [float(stats.get(key, 0.0)) * scale for stats in reuse_stats]
        errors = [
            float(stats.get(f"{key}_ci95", 0.0)) * scale
            for stats in reuse_stats
        ]
        bars = ax.bar(
            x,
            values,
            yerr=errors,
            capsize=3,
            error_kw={
                "elinewidth": 0.8,
                "ecolor": "#222222",
                "capthick": 0.8,
            },
            color=color,
            edgecolor="#333333",
            linewidth=0.45,
        )
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )
        if values and max(values) > 0:
            ax.set_ylim(top=max(values) * 1.18)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, names, rotation=15, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved reuse phase figure: {output}")


def save_rank_stats_figure(
    results: list[ScenarioResult | None],
    output_path: str,
    title: str = "Benchmark",
) -> None:
    valid_results = [
        result for result in results
        if result is not None and result.rank_stats
    ]
    if not valid_results:
        return

    summary_output = Path(output_path)
    output = summary_output.with_name(f"{summary_output.stem}_rank_stats{summary_output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank_ids = sorted({
        int(rank)
        for result in valid_results
        for rank in result.rank_stats.keys()
    })
    if not rank_ids:
        return

    fig, axes = plt.subplots(
        len(valid_results), 4,
        figsize=(18, max(4, 3.5 * len(valid_results))),
        squeeze=False,
    )
    fig.suptitle(f"{title}: Per-Rank Diagnostics", fontsize=16)
    rank_colors = [
        "#2684FC",
        "#00AC47",
        "#FBBC04",
        "#EA4335",
        "#A142F4",
        "#00A3A3",
    ]

    def rank_value(result: ScenarioResult, rank: int, key: str, default: float = 0.0) -> float:
        stats = result.rank_stats.get(rank, result.rank_stats.get(str(rank), {}))
        return float(stats.get(key, default))

    labels = [f"rank {rank}" for rank in rank_ids]
    colors = [rank_colors[rank % len(rank_colors)] for rank in rank_ids]
    for row, result in enumerate(valid_results):
        submitted = [rank_value(result, rank, "submitted") for rank in rank_ids]
        output_tokens = [rank_value(result, rank, "output_tokens") for rank in rank_ids]
        gpu_util = [rank_value(result, rank, "gpu_util_mean") for rank in rank_ids]
        local_hit = [rank_value(result, rank, "local_prefix_hit_rate") * 100.0 for rank in rank_ids]

        for col, (title, values) in enumerate([
            ("Request Share", submitted),
            ("Output Token Share", output_tokens),
        ]):
            ax = axes[row, col]
            if sum(values) > 0:
                ax.pie(
                    values, labels=labels, colors=colors,
                    autopct=lambda pct: f"{pct:.1f}%" if pct >= 2.0 else "",
                    startangle=90, textprops={"fontsize": 8},
                )
            ax.set_title(title)

        for col, (title, values) in enumerate([
            ("GPU Utilization", gpu_util),
            ("Local Prefix Hit", local_hit),
        ], start=2):
            ax = axes[row, col]
            bars = ax.bar(rank_ids, values, color=colors)
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
            ax.set_title(title)
            ax.set_xlabel("rank")
            ax.set_ylabel("%")
            ax.set_xticks(rank_ids)
            ax.set_ylim(0, max(100.0, max(values, default=0.0) * 1.15))
            ax.grid(axis="y", linestyle="--", alpha=0.25)

        axes[row, 0].set_ylabel(result.name, fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved rank stats figure: {output}")


def save_summary_json(
    results: dict,
    output_path: str,
    *,
    metadata: dict | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved json: {output}")


def parse_args():
    # benchmark 入口参数尽量保持简单：只暴露场景规模、模型、拓扑和 SLA
    parser = argparse.ArgumentParser(description="LMPool end-to-end workload benchmark")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--prompt-repeat", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--workload",
        choices=[
            "locality",
            "load-skew",
            "memory-skew",
            "capacity-offload",
            "transfer-calibration",
        ],
        default="locality",
    )
    parser.add_argument("--locality-prefix-groups", type=int, default=16)
    parser.add_argument("--load-skew-prefix-groups", type=int, default=6)
    parser.add_argument("--load-skew-warmup-prompts", type=int, default=0)
    parser.add_argument("--load-skew-hot-groups", type=int, default=0)
    parser.add_argument("--load-skew-hot-share", type=float, default=0.8)
    parser.add_argument("--memory-skew-prefix-groups", type=int, default=0)
    parser.add_argument("--memory-skew-warmup-prompts", type=int, default=0)
    parser.add_argument("--memory-skew-pressure-prompts", type=int, default=0)
    parser.add_argument(
        "--memory-skew-pressure-hot-groups",
        type=int,
        default=0,
        help="Number of anchor groups receiving most memory-pressure requests; 0 uses two.",
    )
    parser.add_argument(
        "--memory-skew-pressure-hot-share",
        type=float,
        default=0.8,
        help="Share of memory-pressure requests assigned to hot anchor groups.",
    )
    parser.add_argument(
        "--memory-skew-anchor-share",
        type=float,
        default=0.375,
        help=(
            "Fraction of each long session reserved for the shared routing anchor; "
            "the remaining tokens form the movable session suffix."
        ),
    )
    parser.add_argument(
        "--memory-skew-reuse-hot-groups",
        type=int,
        default=0,
        help="Number of session groups receiving most post-offload reuse; 0 uses all.",
    )
    parser.add_argument(
        "--memory-skew-reuse-hot-share",
        type=float,
        default=1.0,
        help="Share of reuse requests assigned to hot session groups.",
    )
    parser.add_argument(
        "--memory-skew-trigger-prompts",
        type=int,
        default=0,
        help=(
            "Number of source-routed long trigger requests after pressure drains; "
            "defaults to one per memory-skew prefix group."
        ),
    )
    parser.add_argument("--calibration-prefix-groups", type=int, default=0)
    parser.add_argument("--calibration-warmup-prompts", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--model-name-or-path", type=str, default=MODEL_CONFIG["model_name_or_path"])
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model and KV-cache dtype; auto reads torch_dtype from config.json.",
    )
    parser.add_argument("--nvlink-pairs", type=str, default="0,1")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--kv-block-budget", type=int, default=None)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=MODEL_CONFIG["gpu_memory_utilization"],
    )
    parser.add_argument("--goodput-e2e-sla-ms", type=float, default=2000.0)
    parser.add_argument(
        "--goodput-e2e-sla-sweep-ms",
        default="2000,3000,5000,10000",
        help=(
            "Comma-separated E2E SLA thresholds reported from the same latency "
            "samples; this does not rerun inference."
        ),
    )
    parser.add_argument("--skip-pool", action="store_true")
    parser.add_argument("--output-figure", type=str, default="")
    parser.add_argument("--submit-window", type=int, default=8)
    parser.add_argument("--disable-background-copy", action="store_true")
    parser.add_argument(
        "--background-transfer-mode",
        choices=["copy", "move"],
        default=MODEL_CONFIG["background_transfer_mode"],
    )
    parser.add_argument(
        "--background-move-source-free-block-threshold",
        type=int,
        default=MODEL_CONFIG["background_move_source_free_block_threshold"],
    )
    parser.add_argument(
        "--memory-skew-proactive-move",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move a releasable session suffix after pressure and before reuse.",
    )
    parser.add_argument("--background-copy-max-blocks", type=int, default=MODEL_CONFIG["background_copy_max_blocks"])
    parser.add_argument(
        "--background-copy-batch-max-blocks",
        type=int,
        default=MODEL_CONFIG["background_copy_batch_max_blocks"],
    )
    parser.add_argument(
        "--background-copy-batch-max-candidates",
        type=int,
        default=MODEL_CONFIG["background_copy_batch_max_candidates"],
    )
    parser.add_argument("--background-copy-cooldown-s", type=float, default=MODEL_CONFIG["background_copy_cooldown_s"])
    parser.add_argument("--background-copy-hot-threshold", type=int, default=MODEL_CONFIG["background_copy_hot_threshold"])
    parser.add_argument("--background-copy-min-load-skew", type=float, default=MODEL_CONFIG["background_copy_min_load_skew"])
    parser.add_argument("--background-copy-expected-reuses", type=float, default=MODEL_CONFIG["background_copy_expected_reuses"])
    parser.add_argument("--route-load-weight", type=float, default=MODEL_CONFIG["route_load_weight"])
    parser.add_argument("--route-decode-token-weight", type=float, default=MODEL_CONFIG["route_decode_token_weight"])
    parser.add_argument("--route-owner-spill-sequence-skew", type=float, default=MODEL_CONFIG["route_owner_spill_sequence_skew"])
    parser.add_argument("--route-owner-spill-max-extra-cost", type=float, default=MODEL_CONFIG["route_owner_spill_max_extra_cost"])
    parser.add_argument(
        "--route-load-bypass-threshold",
        type=float,
        default=MODEL_CONFIG["route_load_bypass_threshold"],
    )
    parser.add_argument(
        "--route-prefill-cost-weight",
        type=float,
        default=MODEL_CONFIG["route_prefill_cost_weight"],
    )
    parser.add_argument(
        "--route-reclaim-cost-weight",
        type=float,
        default=MODEL_CONFIG["route_reclaim_cost_weight"],
    )
    parser.add_argument(
        "--foreground-transfer-cost-weight",
        type=float,
        default=MODEL_CONFIG["foreground_transfer_cost_weight"],
    )
    parser.add_argument(
        "--foreground-transfer-min-benefit-ratio",
        type=float,
        default=MODEL_CONFIG["foreground_transfer_min_benefit_ratio"],
    )
    parser.add_argument(
        "--foreground-transfer-require-idle-target",
        action=argparse.BooleanOptionalAction,
        default=MODEL_CONFIG["foreground_transfer_require_idle_target"],
        help=(
            "Admit foreground transfer only to a normal NVLink neighbour with "
            "no reported waiting, running, or pending work."
        ),
    )
    parser.add_argument(
        "--foreground-transfer-bandwidth-gib-s",
        type=float,
        default=MODEL_CONFIG["foreground_transfer_bandwidth_gib_s"],
        help="Scalar fallback used only when no size-aware profile is provided.",
    )
    parser.add_argument(
        "--foreground-transfer-profile-json",
        type=str,
        default="",
        help=(
            "Size-aware logical-pair latency profile produced by "
            "build_transfer_profile.py."
        ),
    )
    parser.add_argument(
        "--foreground-transfer-fixed-latency-ms",
        type=float,
        default=MODEL_CONFIG["foreground_transfer_fixed_latency_ms"],
    )
    parser.add_argument(
        "--foreground-transfer-interference-multiplier",
        type=float,
        default=MODEL_CONFIG["foreground_transfer_interference_multiplier"],
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIO_NAMES),
        help=(
            "Comma-separated scenario subset. This is useful for disjoint cost "
            "calibration runs, for example --scenarios multi-gpu-lmpool."
        ),
    )
    parser.add_argument(
        "--foreground-prefill-token-time-ms",
        type=float,
        default=MODEL_CONFIG["foreground_prefill_token_time_ms"],
    )
    parser.add_argument(
        "--foreground-future-reuse-discount",
        type=float,
        default=MODEL_CONFIG["foreground_future_reuse_discount"],
    )
    parser.add_argument(
        "--kv-transfer-prewarm-blocks",
        type=int,
        default=MODEL_CONFIG["kv_transfer_prewarm_blocks"],
        help="Representative KV blocks sent per NVLink pair before serving starts.",
    )
    parser.add_argument("--route-cache-queue-slack", type=float, default=MODEL_CONFIG["route_cache_queue_slack"])
    return parser.parse_args()


def resolve_kv_block_budget(args) -> int:
    """Resolve the one per-rank KV capacity used by every scenario."""
    budget = (
        args.kv_block_budget
        if args.kv_block_budget is not None
        else MODEL_CONFIG["max_cached_blocks"]
    )
    if budget < 1:
        raise ValueError("--kv-block-budget must be >= 1")
    return budget


def parse_pairs(raw: str) -> list[tuple[int, int]]:
    # 解析命令行里的 "0,1;2,3" 形式拓扑输入
    if not raw:
        return []
    pairs = []
    for item in raw.split(";"):
        a, b = item.split(",")
        pairs.append((int(a), int(b)))
    return pairs


def parse_goodput_sla_sweep_ms(raw: str, primary_ms: float) -> list[float]:
    """Return sorted unique positive SLA thresholds in seconds."""
    try:
        values_ms = [
            float(item.strip())
            for item in str(raw).split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            "--goodput-e2e-sla-sweep-ms must be a comma-separated number list"
        ) from exc
    values_ms.append(float(primary_ms))
    if any(value <= 0 for value in values_ms):
        raise ValueError("goodput E2E SLA thresholds must be positive")
    return sorted({value / 1000.0 for value in values_ms})


def apply_background_copy_args(config: dict, args) -> None:
    config["enable_background_copy"] = not args.disable_background_copy
    config["background_copy_max_blocks"] = args.background_copy_max_blocks
    config["background_copy_batch_max_blocks"] = (
        args.background_copy_batch_max_blocks
    )
    config["background_copy_batch_max_candidates"] = (
        args.background_copy_batch_max_candidates
    )
    config["background_copy_cooldown_s"] = args.background_copy_cooldown_s
    config["background_copy_hot_threshold"] = args.background_copy_hot_threshold
    config["background_copy_min_load_skew"] = args.background_copy_min_load_skew
    config["background_copy_expected_reuses"] = args.background_copy_expected_reuses
    config["background_transfer_mode"] = args.background_transfer_mode
    config["background_move_source_free_block_threshold"] = (
        args.background_move_source_free_block_threshold
    )


def apply_route_args(
    config: dict,
    args,
    transfer_latency_profile: dict | None = None,
) -> None:
    config["route_load_weight"] = args.route_load_weight
    config["route_decode_token_weight"] = args.route_decode_token_weight
    config["route_owner_spill_sequence_skew"] = args.route_owner_spill_sequence_skew
    config["route_owner_spill_max_extra_cost"] = args.route_owner_spill_max_extra_cost
    config["route_load_bypass_threshold"] = args.route_load_bypass_threshold
    config["route_prefill_cost_weight"] = args.route_prefill_cost_weight
    config["route_reclaim_cost_weight"] = args.route_reclaim_cost_weight
    config["foreground_transfer_cost_weight"] = args.foreground_transfer_cost_weight
    config["foreground_transfer_min_benefit_ratio"] = (
        args.foreground_transfer_min_benefit_ratio
    )
    config["foreground_transfer_require_idle_target"] = (
        args.foreground_transfer_require_idle_target
    )
    config["foreground_transfer_bandwidth_gib_s"] = args.foreground_transfer_bandwidth_gib_s
    config["foreground_transfer_latency_profile"] = transfer_latency_profile
    config["foreground_transfer_fixed_latency_ms"] = args.foreground_transfer_fixed_latency_ms
    config["foreground_transfer_interference_multiplier"] = (
        args.foreground_transfer_interference_multiplier
    )
    config["foreground_prefill_token_time_ms"] = args.foreground_prefill_token_time_ms
    config["foreground_future_reuse_discount"] = args.foreground_future_reuse_discount
    config["kv_transfer_prewarm_blocks"] = max(1, args.kv_transfer_prewarm_blocks)
    config["route_cache_queue_slack"] = args.route_cache_queue_slack


def main():
    # 主流程：
    # 1) 准备 prompts
    # 2) 跑 single-gpu 基线
    # 3) 跑 multi-gpu 独立基线
    # 4) 跑 routing / transfer / pool 场景
    # 5) 打印和导出结果
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if args.world_size < 1:
        raise SystemExit("--world-size must be >= 1")
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")
    requested_scenarios = {
        name.strip() for name in args.scenarios.split(",") if name.strip()
    }
    unknown_scenarios = requested_scenarios - set(SCENARIO_NAMES)
    if not requested_scenarios or unknown_scenarios:
        raise SystemExit(
            f"--scenarios must select from {','.join(SCENARIO_NAMES)}; "
            f"unknown={sorted(unknown_scenarios)}"
        )
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")
    if (
        args.workload in {"memory-skew", "capacity-offload"}
        and not args.disable_background_copy
        and args.background_transfer_mode != "move"
    ):
        raise SystemExit(
            "memory-skew requires --background-transfer-mode move or "
            "--disable-background-copy; copy cannot demonstrate source-capacity release"
        )
    try:
        kv_block_budget = resolve_kv_block_budget(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.workload == "locality" and not 1 <= args.locality_prefix_groups <= args.num_prompts:
        raise SystemExit("--locality-prefix-groups must be between 1 and --num-prompts")
    try:
        load_skew_hot_groups = (
            args.load_skew_hot_groups
            if args.load_skew_hot_groups > 0
            else args.load_skew_prefix_groups
        )
        if not 1 <= load_skew_hot_groups <= args.load_skew_prefix_groups:
            raise ValueError(
                "--load-skew-hot-groups must fit within --load-skew-prefix-groups"
            )
        load_skew_warmup_prompts, _ = resolve_load_skew_phases(
            args.num_prompts,
            load_skew_hot_groups,
            args.load_skew_warmup_prompts,
        )
    except ValueError as exc:
        if args.workload == "load-skew":
            raise SystemExit(str(exc)) from exc
        load_skew_warmup_prompts = max(1, args.num_prompts // 4)
    try:
        memory_skew_prefix_groups = resolve_memory_skew_prefix_groups(
            args.num_prompts,
            args.memory_skew_prefix_groups,
            args.memory_skew_warmup_prompts,
            args.memory_skew_pressure_prompts,
            args.memory_skew_trigger_prompts,
            allow_zero_trigger=args.memory_skew_proactive_move,
        )
        memory_skew_warmup_prompts, memory_skew_pressure_prompts, memory_skew_trigger_prompts, _ = (
            resolve_memory_skew_phases(
                args.num_prompts,
                memory_skew_prefix_groups,
                args.memory_skew_warmup_prompts,
                args.memory_skew_pressure_prompts,
                args.memory_skew_trigger_prompts,
                allow_zero_trigger=args.memory_skew_proactive_move,
            )
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    memory_skew_pressure_hot_groups = (
        args.memory_skew_pressure_hot_groups
        if args.memory_skew_pressure_hot_groups > 0
        else min(2, memory_skew_prefix_groups)
    )
    if not 1 <= memory_skew_pressure_hot_groups <= memory_skew_prefix_groups:
        raise SystemExit(
            "--memory-skew-pressure-hot-groups must fit within "
            "--memory-skew-prefix-groups"
        )
    if not 0.0 < args.memory_skew_pressure_hot_share <= 1.0:
        raise SystemExit("--memory-skew-pressure-hot-share must be in (0, 1]")
    if not 0.0 < args.memory_skew_anchor_share < 1.0:
        raise SystemExit("--memory-skew-anchor-share must be in (0, 1)")
    memory_skew_reuse_hot_groups = (
        args.memory_skew_reuse_hot_groups
        if args.memory_skew_reuse_hot_groups > 0
        else memory_skew_prefix_groups
    )
    if not 1 <= memory_skew_reuse_hot_groups <= memory_skew_prefix_groups:
        raise SystemExit(
            "--memory-skew-reuse-hot-groups must fit within "
            "--memory-skew-prefix-groups"
        )
    if not 0.0 < args.memory_skew_reuse_hot_share <= 1.0:
        raise SystemExit("--memory-skew-reuse-hot-share must be in (0, 1]")
    try:
        calibration_warmup_prompts = resolve_transfer_calibration_warmup_prompts(
            args.num_prompts,
            args.calibration_warmup_prompts,
        )
        calibration_prefix_groups = resolve_transfer_calibration_prefix_groups(
            args.num_prompts,
            args.calibration_prefix_groups,
            calibration_warmup_prompts,
        )
    except ValueError as exc:
        if args.workload == "transfer-calibration":
            raise SystemExit(str(exc)) from exc
        calibration_warmup_prompts = max(1, args.num_prompts // 2)
        calibration_prefix_groups = max(1, args.num_prompts // 2)
    visible_gpus = torch.cuda.device_count()
    if args.world_size > visible_gpus:
        raise SystemExit(
            f"--world-size {args.world_size} exceeds visible CUDA devices {visible_gpus}. "
            "Check CUDA_VISIBLE_DEVICES."
        )

    model_name = args.model_name_or_path
    try:
        runtime_model_config, model_metadata = resolve_model_runtime_config(
            model_name,
            MODEL_CONFIG,
            dtype_override=args.dtype,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot resolve model config for {model_name}: {exc}") from exc
    # The model resolver starts from MODEL_CONFIG. Apply the CLI override to
    # the shared runtime template as well as to each scenario config so the
    # exported resolved_config matches the workers that actually run.
    runtime_model_config["gpu_memory_utilization"] = args.gpu_memory_utilization
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = build_prompts(
        tokenizer,
        args.num_prompts,
        args.prompt_repeat,
        args.workload,
        locality_prefix_groups=args.locality_prefix_groups,
        load_skew_prefix_groups=args.load_skew_prefix_groups,
        load_skew_warmup_prompts=load_skew_warmup_prompts,
        load_skew_hot_groups=load_skew_hot_groups,
        load_skew_hot_share=args.load_skew_hot_share,
        memory_skew_prefix_groups=memory_skew_prefix_groups,
        memory_skew_warmup_prompts=memory_skew_warmup_prompts,
        memory_skew_pressure_prompts=memory_skew_pressure_prompts,
        memory_skew_trigger_prompts=memory_skew_trigger_prompts,
        memory_skew_pressure_hot_groups=memory_skew_pressure_hot_groups,
        memory_skew_pressure_hot_share=args.memory_skew_pressure_hot_share,
        memory_skew_anchor_share=args.memory_skew_anchor_share,
        memory_skew_reuse_hot_groups=memory_skew_reuse_hot_groups,
        memory_skew_reuse_hot_share=args.memory_skew_reuse_hot_share,
        memory_skew_proactive_move=args.memory_skew_proactive_move,
        calibration_prefix_groups=calibration_prefix_groups,
        calibration_warmup_prompts=calibration_warmup_prompts,
        seed=args.seed,
    )

    max_prompt_tokens = max(len(tokenizer.encode(prompt)) for prompt in prompts)
    required_model_length = max_prompt_tokens + args.max_tokens
    model_position_limit = int(runtime_model_config.get("max_position", 0))
    if model_position_limit and required_model_length > model_position_limit:
        raise SystemExit(
            f"tokenized prompt plus output requires {required_model_length} tokens, "
            f"but the model supports only {model_position_limit}"
        )
    runtime_model_config["max_model_length"] = max(
        int(runtime_model_config["max_model_length"]),
        required_model_length,
    )
    runtime_model_config["max_num_batched_tokens"] = max(
        int(runtime_model_config["max_num_batched_tokens"]),
        int(runtime_model_config["max_model_length"]),
    )
    # ModelRunner historically consumed the singular alias while Scheduler
    # consumed the plural key. Keep both synchronized for long-prompt workloads.
    runtime_model_config["max_num_batch_tokens"] = int(
        runtime_model_config["max_num_batched_tokens"]
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        max_model_length=runtime_model_config["max_model_length"],
    )
    goodput_e2e_sla_s = args.goodput_e2e_sla_ms / 1000.0
    try:
        goodput_sla_sweep_s = parse_goodput_sla_sweep_ms(
            args.goodput_e2e_sla_sweep_ms,
            args.goodput_e2e_sla_ms,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    nvlink_pairs = parse_pairs(args.nvlink_pairs) if args.nvlink_pairs else []
    transfer_latency_profile = None
    if args.foreground_transfer_profile_json:
        expected_bytes_per_block = (
            2
            * int(runtime_model_config["num_layers"])
            * int(runtime_model_config["block_size"])
            * int(runtime_model_config["num_kv_heads"])
            * int(runtime_model_config["head_dim"])
            * int(runtime_model_config["kv_dtype_bytes"])
        )
        try:
            transfer_latency_profile = load_transfer_latency_profile(
                args.foreground_transfer_profile_json,
                expected_bytes_per_block=expected_bytes_per_block,
                expected_pairs=nvlink_pairs,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot load transfer latency profile: {exc}") from exc
    def apply_transfer_placement(config: dict) -> None:
        # Workloads never select a worker rank. Every scenario uses its own
        # normal dispatch policy; NVLink topology is visible only to transfer
        # planning and to topology-aware routing.
        config["memory_skew_prefix_groups"] = memory_skew_prefix_groups
        config["benchmark_memory_skew_warmup_prompts"] = (
            memory_skew_warmup_prompts
        )
        config["benchmark_memory_skew_pressure_prompts"] = (
            memory_skew_pressure_prompts
        )
        config["benchmark_memory_skew_trigger_prompts"] = (
            memory_skew_trigger_prompts
        )
        config["benchmark_memory_skew_pressure_hot_groups"] = (
            memory_skew_pressure_hot_groups
        )
        config["benchmark_memory_skew_pressure_hot_share"] = (
            args.memory_skew_pressure_hot_share
        )
        config["benchmark_memory_skew_anchor_share"] = args.memory_skew_anchor_share
        config["benchmark_memory_skew_reuse_hot_groups"] = (
            memory_skew_reuse_hot_groups
        )
        config["benchmark_memory_skew_reuse_hot_share"] = (
            args.memory_skew_reuse_hot_share
        )
        config["benchmark_memory_skew_proactive_move"] = (
            args.memory_skew_proactive_move
        )
        config["benchmark_transfer_prefix_groups"] = (
            calibration_prefix_groups
            if args.workload == "transfer-calibration"
            else (
                load_skew_hot_groups
                if args.workload == "load-skew"
                else memory_skew_prefix_groups
            )
        )
        config["benchmark_load_skew_warmup_prompts"] = (
            load_skew_warmup_prompts
        )
        config["benchmark_calibration_warmup_prompts"] = (
            calibration_warmup_prompts
        )
        config["gpu_memory_utilization"] = args.gpu_memory_utilization
        config["require_exact_kv_block_budget"] = args.kv_block_budget is not None
        config["benchmark_goodput_sla_sweep_s"] = goodput_sla_sweep_s

    dataset_profile = profile_trace_prefix_sharing(
        tokenizer,
        prompts,
        block_size=int(runtime_model_config["block_size"]),
    )
    trace_request_share_rate = float(dataset_profile["request_prefix_share_rate"])
    trace_token_share_ratio = float(dataset_profile["token_prefix_share_ratio"])

    # single-gpu baseline：单卡独立执行，不启用全局池
    baseline = None
    baseline_config = make_config(1, False, None, runtime_model_config)
    baseline_config["model_name_or_path"] = model_name
    baseline_config["max_cached_blocks"] = kv_block_budget
    baseline_config["random_seed"] = args.seed
    apply_transfer_placement(baseline_config)
    if "single-gpu" in requested_scenarios:
        baseline = run_repeated_engine_scenario(
            args.repetitions,
            name="single-gpu",
            config=baseline_config,
            prompts=prompts,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            goodput_e2e_sla_s=goodput_e2e_sla_s,
            submit_window=args.submit_window,
            workload=args.workload,
        )

    # multi-gpu baseline：不共享 KV、不走控制面路由，但请求通过 round-robin 分发到多张卡
    independent_result = None
    multi_gpu_config = make_config(args.world_size, False, None, runtime_model_config)
    multi_gpu_config["model_name_or_path"] = model_name
    multi_gpu_config["max_cached_blocks"] = kv_block_budget
    multi_gpu_config["random_seed"] = args.seed
    apply_transfer_placement(multi_gpu_config)
    if "multi-gpu" in requested_scenarios:
        independent_result = run_repeated_engine_scenario(
            args.repetitions,
            name="multi-gpu",
            config=multi_gpu_config,
            prompts=prompts,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            route_mode="round_robin",
            goodput_e2e_sla_s=goodput_e2e_sla_s,
            submit_window=args.submit_window,
            workload=args.workload,
        )

    # multi-gpu-kv-routing：走控制面路由，用来测 prefix 命中带来的收益
    kv_routing = None
    routing_config = make_config(
        args.world_size,
        True,
        nvlink_pairs or None,
        runtime_model_config,
    )
    routing_config["model_name_or_path"] = model_name
    routing_config["max_cached_blocks"] = kv_block_budget
    routing_config["enable_foreground_rebalance"] = False
    routing_config["enable_transfer_aware_owner_routing"] = False
    routing_config["enable_background_copy"] = False
    routing_config["random_seed"] = args.seed
    apply_transfer_placement(routing_config)
    apply_route_args(routing_config, args, transfer_latency_profile)
    if "multi-gpu-kv-routing" in requested_scenarios:
        kv_routing = run_repeated_engine_scenario(
            args.repetitions,
            name="multi-gpu-kv-routing",
            config=routing_config,
            prompts=prompts,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            route_mode="control_plane",
            goodput_e2e_sla_s=goodput_e2e_sla_s,
            submit_window=args.submit_window,
            workload=args.workload,
        )

    # multi-gpu-kv-transfer：用 round-robin 分发，尽量隔离出 transfer / rebalance 的开销
    kv_eviction = None
    eviction_config = make_config(
        args.world_size,
        True,
        nvlink_pairs or None,
        runtime_model_config,
    )
    eviction_config["model_name_or_path"] = model_name
    eviction_config["max_cached_blocks"] = kv_block_budget
    eviction_config["random_seed"] = args.seed
    apply_transfer_placement(eviction_config)
    eviction_config["preserve_cache_via_transfer"] = args.workload in {
            "load-skew",
            "memory-skew",
            "capacity-offload",
            "transfer-calibration",
    }
    apply_background_copy_args(eviction_config, args)
    if args.workload in {"memory-skew", "capacity-offload"}:
        eviction_config["enable_foreground_rebalance"] = False
    if args.workload == "load-skew":
        eviction_config["enable_foreground_rebalance"] = False
        eviction_config["enable_transfer_aware_owner_routing"] = False
    apply_route_args(eviction_config, args, transfer_latency_profile)
    if "multi-gpu-kv-transfer" in requested_scenarios:
        kv_eviction = run_repeated_engine_scenario(
            args.repetitions,
            name="multi-gpu-kv-transfer",
            config=eviction_config,
            prompts=prompts,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            route_mode="round_robin",
            goodput_e2e_sla_s=goodput_e2e_sla_s,
            submit_window=args.submit_window,
            workload=args.workload,
        )

    pool_result = None
    if "multi-gpu-lmpool" in requested_scenarios and not args.skip_pool:
        if visible_gpus < args.world_size:
            print(f"pool scenario skipped: need {args.world_size} CUDA devices")
        else:
            # multi-gpu-lmpool：真实全局池化路径，控制面路由 + 数据面执行一起跑。
            pool_pairs = nvlink_pairs or None
            pool_config = make_config(
                args.world_size,
                True,
                pool_pairs,
                runtime_model_config,
            )
            pool_config["model_name_or_path"] = model_name
            pool_config["max_cached_blocks"] = kv_block_budget
            pool_config["random_seed"] = args.seed
            apply_transfer_placement(pool_config)
            pool_config["preserve_cache_via_transfer"] = args.workload in {
                "load-skew",
                "memory-skew",
                "capacity-offload",
                "transfer-calibration",
            }
            apply_background_copy_args(pool_config, args)
            if args.workload in {"memory-skew", "capacity-offload"}:
                pool_config["enable_foreground_rebalance"] = False
            if args.workload == "load-skew":
                pool_config["enable_foreground_rebalance"] = False
                pool_config["enable_transfer_aware_owner_routing"] = (
                    args.workload != "load-skew"
                )
            apply_route_args(pool_config, args, transfer_latency_profile)
            pool_result = run_repeated_engine_scenario(
                args.repetitions,
                name="multi-gpu-lmpool",
                config=pool_config,
                prompts=prompts,
                sampling_params=sampling_params,
                tokenizer=tokenizer,
                goodput_e2e_sla_s=goodput_e2e_sla_s,
                submit_window=args.submit_window,
                workload=args.workload,
            )

    all_results = [
        baseline,
        independent_result,
        kv_routing,
        kv_eviction,
        pool_result,
    ]
    for result in all_results:
        if result is not None:
            result.trace_request_share_rate = trace_request_share_rate
            result.trace_token_share_ratio = trace_token_share_ratio
            # Retain the old JSON field as a compatibility alias.
            result.theoretical_prefix_hit_rate = trace_request_share_rate
            for trial in result.trial_results or []:
                trial["trace_request_share_rate"] = trace_request_share_rate
                trial["trace_token_share_ratio"] = trace_token_share_ratio
                trial["theoretical_prefix_hit_rate"] = trace_request_share_rate
    summary_title = (
        f"{workload_summary_title(args.workload)} ({model_metadata['label']})"
    )
    print_summary_table(all_results, title=summary_title)
    if args.output_figure:
        save_summary_figure(all_results, args.output_figure, title=summary_title)
        save_reuse_phase_figure(all_results, args.output_figure, title=summary_title)
        save_rank_stats_figure(all_results, args.output_figure, title=summary_title)
    if args.output_json:
        payload = {
            "single-gpu": asdict(baseline) if baseline is not None else None,
            "multi-gpu": asdict(independent_result) if independent_result is not None else None,
            "multi-gpu-kv-routing": asdict(kv_routing) if kv_routing is not None else None,
            "multi-gpu-kv-transfer": asdict(kv_eviction) if kv_eviction is not None else None,
            "multi-gpu-lmpool": asdict(pool_result) if pool_result is not None else None,
        }
        run_metadata = build_run_metadata(
            "benchmark_e2e",
            args,
            model=model_metadata,
            resolved_config={
                **runtime_model_config,
                "resolved_kv_block_budget": kv_block_budget,
                "resolved_load_skew_prefix_groups": args.load_skew_prefix_groups,
                "resolved_load_skew_hot_groups": load_skew_hot_groups,
                "resolved_load_skew_warmup_prompts": (
                    load_skew_warmup_prompts
                ),
                "resolved_max_prompt_tokens": max_prompt_tokens,
                "resolved_memory_skew_prefix_groups": memory_skew_prefix_groups,
                "resolved_memory_skew_warmup_prompts": (
                    memory_skew_warmup_prompts
                ),
                "resolved_memory_skew_pressure_prompts": (
                    memory_skew_pressure_prompts
                ),
                "resolved_memory_skew_trigger_prompts": (
                    memory_skew_trigger_prompts
                ),
                "resolved_memory_skew_pressure_hot_groups": (
                    memory_skew_pressure_hot_groups
                ),
                "resolved_memory_skew_pressure_hot_share": (
                    args.memory_skew_pressure_hot_share
                ),
                "resolved_memory_skew_anchor_share": args.memory_skew_anchor_share,
                "resolved_memory_skew_reuse_hot_groups": memory_skew_reuse_hot_groups,
                "resolved_memory_skew_reuse_hot_share": args.memory_skew_reuse_hot_share,
                "resolved_calibration_prefix_groups": calibration_prefix_groups,
                "resolved_calibration_warmup_prompts": (
                    calibration_warmup_prompts
                ),
                "resolved_nvlink_pairs": nvlink_pairs,
                # Record policy values after argument parsing.  The model
                # snapshot alone does not contain these scenario controls;
                # omitting them made the JSON appear to use the defaults even
                # when the command line supplied different admission values.
                "resolved_foreground_transfer_min_benefit_ratio": (
                    args.foreground_transfer_min_benefit_ratio
                ),
                "resolved_foreground_transfer_bandwidth_gib_s": (
                    args.foreground_transfer_bandwidth_gib_s
                ),
                "resolved_foreground_transfer_fixed_latency_ms": (
                    args.foreground_transfer_fixed_latency_ms
                ),
                "resolved_foreground_transfer_interference_multiplier": (
                    args.foreground_transfer_interference_multiplier
                ),
                "resolved_disable_background_copy": args.disable_background_copy,
                "resolved_kv_transfer_prewarm_blocks": (
                    args.kv_transfer_prewarm_blocks
                ),
            },
        )
        run_metadata["dataset_profile"] = dataset_profile
        run_metadata["metric_definitions"] = {
            "ttft_s": "first token timestamp minus request submission timestamp",
            "tpot_s": (
                "completion timestamp minus first token timestamp, divided by "
                "output_tokens minus one; requests with one output token are excluded"
            ),
            "e2e_s": "completion timestamp minus request submission timestamp",
            "goodput_sla_sweep_tok_s": (
                "output-token throughput for requests meeting each E2E SLA; "
                "JSON keys are SLA thresholds in milliseconds"
            ),
            "ci95": (
                "two-sided Student-t half-width across complete scenario repetitions"
            ),
        }
        save_summary_json(payload, args.output_json, metadata=run_metadata)


if __name__ == "__main__":
    main()
