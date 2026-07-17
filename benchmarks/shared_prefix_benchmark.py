"""
运行方式示例：

1. 使用默认参数：
   CUDA_VISIBLE_DEVICES=0,2 UV_CACHE_DIR=/tmp/uvcache uv run python benchmarks/shared_prefix_benchmark.py

2. 显式指定参数：
    CUDA_VISIBLE_DEVICES=0,2 UV_CACHE_DIR=/tmp/uvcache uv run python benchmarks/shared_prefix_benchmark.py \
    --num-prompts 32 \
    --prompt-repeat 10 \
    --max-tokens 16 \
    --temperature 0.6 \
    --ignore-eos \
    --seed 0 \
    --repetitions 3 \
    --workload locality \
    --locality-prefix-groups 16 \
    --memory-skew-prefix-groups 15 \
    --world-size 2 \
    --nvlink-pairs "0,1" \
    --kv-block-budget 64 \
    --submit-window 5 \
    --background-copy-max-blocks 1 \
    --background-copy-cooldown-s 2.0 \
    --background-copy-hot-threshold 3 \
    --route-load-weight 0.03 \
    --route-load-bypass-threshold 256 \
    --route-prefill-cost-weight 1.0 \
    --route-reclaim-cost-weight 0.5 \
    --foreground-transfer-cost-weight 1.0 \
    --foreground-transfer-min-benefit-ratio 1.5 \
    --route-cache-queue-slack 256 \
    --goodput-e2e-sla-ms 10000 \
    --output-json ./benchmarks/results/shared_prefix_benchmark_202607121117.json \
    --output-figure ./benchmarks/results/shared_prefix_benchmark_202607121117.png

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
  `load-skew` 用一个热点前缀加少量冷前缀制造请求负载倾斜；
  `memory-skew` 依次执行热点前缀预热、源端一次性前缀施压、热点前缀复用三个阶段，
  用于验证 foreground transfer 是否既释放源端容量，又在 NVLink 伙伴保留可复用前缀。

9. `--locality-prefix-groups`：
  `locality` workload 中不同长共享前缀的组数，默认 16。每组请求数保持均衡，并按 `--seed`
  打乱提交顺序，避免前缀组编号与 round-robin rank 周期重合。组数必须在 1 和
  `--num-prompts` 之间；组数越多，未启用 routing 时跨 GPU 重复缓存和 cache churn 越明显。

10. `--memory-skew-prefix-groups`：
  `memory-skew` workload 中需要跨阶段保留的长热点前缀数量。默认 0 表示自动选择不超过 15
  的最大奇数，使其适配 warmup/reuse 阶段长度。每个热点在 warmup
  阶段固定到一个 source rank，reuse 阶段交错重放；使用多个热点可防止 round-robin 在第一次
  miss 后自然把唯一热点复制到所有 GPU。该值不能超过 warmup 或 reuse 阶段请求数。

11. `--output-json`：
  将各场景统计结果导出到指定 JSON 文件。脚本会自动创建父目录，并在成功后打印
  `saved json: ...`。

12. `--model-name-or-path`：
  指定要测试的模型名称或本地路径。默认使用脚本里的 Qwen 配置，对应模型结构也基于这份配置。

13. `--nvlink-pairs`：
  手动指定 NVLink 拓扑，格式如 `0,1` 或 `0,1;2,3`。这里使用的是
  `CUDA_VISIBLE_DEVICES` 之后的逻辑 GPU 编号，不是物理 GPU 编号。如果不想手动写，
  可以传空字符串，让底层逻辑尝试解析 `nvidia-smi topo -m`。命令行里包含分号时必须加引号，
  例如 `--nvlink-pairs "0,2;1,3;4,5;6,7"`。

14. `--world-size`：
  多卡场景启动多少个 data-plane worker。默认 2；八卡实验需要显式传 `--world-size 8`。
  该值不能超过 `CUDA_VISIBLE_DEVICES` 暴露出的 GPU 数。

15. `--kv-block-budget`：
  每个 rank 请求使用的 KV block 上限。五个场景必须使用同一个值，避免把容量差异误判成
  routing / transfer 收益；worker 仍会根据实际可用显存把它收敛到可分配的共同上限。
  旧参数 `--routing-max-cached-blocks` 和 `--eviction-max-cached-blocks` 仅保留命令兼容，
  如果同时提供则数值必须完全相同。

16. `--goodput-e2e-sla-ms`：
  goodput 的端到端延迟门槛，单位毫秒。只有在这个 SLA 内完成的请求，其输出 token
  才计入 goodput。因此表里的 goodput 单位是 tokens/s，不是 requests/s。

17. `--skip-pool`：
  跳过 `multi-gpu-lmpool` 场景，只跑基线、routing 和 transfer。

18. `--output-figure`：
  将五种场景的核心指标画成一张图表图片并保存到指定路径。脚本会自动创建父目录，
  使用无显示环境可用的 Matplotlib Agg 后端，并在成功后打印 `saved figure: ...`。

19. `--submit-window`：
  benchmark 中允许同时在途的请求数。值越大越接近一次性高并发提交；值越小越容易让前面请求先完成
  prefill 并上报全局页表，从而观察在线 prefix reuse。设为 0 或负数表示一次性提交全部请求。
  如果要验证 prefix hit 是否生效，建议先用 4 ~ 8；如果要模拟 burst 流量，可以设为 0 或 -1。

20. `--disable-background-copy`：
  关闭后台 speculative copy-style transfer。默认开启，用于把热点 prefix block 异步复制到 NVLink
  伙伴，服务后续请求；关闭后只保留因当前请求容量不足而同步触发的 foreground transfer。

21. `--background-copy-max-blocks`：
  每次后台 copy 最多复制多少个 prefix block。值越大越可能提高后续本地命中，但会占用更多
  NCCL / worker 时间；排查功能正确性可先用 1，验证收益时建议尝试 2。

22. `--background-copy-cooldown-s`：
  同一个 prefix 在同一组 `src -> dst` GPU 之间再次触发后台 copy 的最短间隔，单位秒。
  值越大越保守，值越小越容易在高并发下产生更多 transfer。验证后台 copy 收益时可尝试 0.5。

23. `--background-copy-hot-threshold`：
  同一个 prefix 至少被路由命中多少次后才允许后台 copy。值越大越保守，能减少无效 copy；
  值为 1 时退化为旧的 eager speculative copy。

24. `--route-load-weight`：
  旧 prefix score 中 token-aware load 的 tie-break 权重。主路由决策现在使用统一预计完成成本；
  该参数只在成本相同时参与稳定排序，通常保持默认值。

25. `--route-load-bypass-threshold`：
  冷目标的预计总成本必须比 prefix owner 至少低多少 token-equivalent cost，才允许绕过
  owner。值越小越激进，越容易牺牲 locality 换并行度。

26. `--route-prefill-cost-weight`：
  缺失 prefix token 的重算成本权重。默认 1.0，使一个缺失 token 与一个 waiting token
  使用相同成本单位；增大后路由更偏向已有 prefix 的 owner。

27. `--route-reclaim-cost-weight`：
  使用 reclaimable capacity 时，每个待回收 block 按 `block_size * weight` 计入的附加成本。
  默认 0.5，用于反映回收元数据操作及未来 cache miss 风险。

28. `--foreground-transfer-cost-weight`：
  一个 transferred block 折算成多少个 block 的 prefill 成本，默认 1.0。可根据
  `benchmark_kv_transfer.py` 与实测 prefill 时间校准。

29. `--foreground-transfer-min-benefit-ratio`：
  foreground transfer 预计保留的 prefix prefill 收益与 transfer 成本的最小比值。
  默认 1.5；未达到门槛时跳过 transfer，直接使用本地回收。

30. `--route-cache-queue-slack`：
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
6. 如果要专门验证后台 speculative copy-style transfer，所有场景仍应使用相同的
   `--kv-block-budget`；通过 `memory-skew` workload 和 background-copy 参数制造迁移需求，
   不要通过缩小某一个场景的容量制造不公平结果。
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
from lmpool.engine.global_block_manager import GlobalBlockManager
from lmpool.engine.global_scheduler import GlobalScheduler
from lmpool.engine.llm_engine import LLMEngine
from lmpool.engine.sequence import Sequence
from lmpool.sampling_parameters import SamplingParams


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
    "gpu_memory_utilization": 0.05,
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
    "route_prefill_cost_weight": 1.0,
    "route_reclaim_cost_weight": 0.5,
    "route_cache_queue_slack": 256.0,
    "enable_foreground_rebalance": True,
    "foreground_transfer_min_blocks": 2,
    "foreground_transfer_cost_weight": 1.0,
    "foreground_transfer_min_benefit_ratio": 1.5,
    "foreground_transfer_fail_cooldown_s": 2.0,
    "foreground_transfer_fail_cooldown_max_s": 30.0,
    # 后台 speculative copy-style transfer：每次只复制少量热点 prefix block，避免抢占前台推理时间。
    "enable_background_copy": True,
    "background_copy_max_blocks": 1,
    "background_copy_cooldown_s": 2.0,
    "background_copy_hot_threshold": 3,
}


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
    mean_ttpt_s: float
    p50_ttpt_s: float
    p90_ttpt_s: float
    p95_ttpt_s: float
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
    theoretical_prefix_hit_rate: float = 0.0
    route_matched_block_ratio: float = 0.0
    reclaimable_capacity_route_rate: float = 0.0
    stale_route_hit_rate: float = 0.0
    reuse_phase_request_hit_rate: float = 0.0
    reuse_phase_token_ratio: float = 0.0
    repetitions: int = 1
    throughput_tok_s_std: float = 0.0
    goodput_tok_s_std: float = 0.0
    mean_ttft_s_std: float = 0.0
    mean_e2e_s_std: float = 0.0
    phase_latency_stats: dict[str, dict[str, float]] | None = None


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
    seed: int = 0,
) -> list[str]:
    # locality: 多组长共享前缀，主要验证 KVCache-aware routing，避免单一前缀被每卡复制后
    # round-robin 也自然获得接近 100% 的本地命中。
    # load-skew: 多数请求共享热点前缀，少数请求落到冷前缀，观察 routing 是否能兼顾 locality 和 load。
    # memory-skew: 依次执行热点预热、一次性前缀施压、热点复用三个阶段，验证 transfer
    # 是否能在释放源端容量的同时，把完整可复用前缀链保留到 NVLink 伙伴。
    if workload == "locality":
        locality_prefixes = [
            build_shared_prefix(prompt_repeat, f"locality-{group:04d}")
            for group in range(locality_prefix_groups)
        ]
        locality_group_order = [i % locality_prefix_groups for i in range(num_prompts)]
        random.Random(seed).shuffle(locality_group_order)
    elif workload == "load-skew":
        hot_prefix = build_shared_prefix(
            prompt_repeat,
            "hot",
        )
        cold_prefixes = [
            build_shared_prefix(max(1, prompt_repeat // 2), f"cold-{group:04d}")
            for group in range(4)
        ]
    else:
        # Multiple hot chains prevent round-robin from learning the only hot
        # prefix locally after one miss. Each group is warmed on one source and
        # revisited in an interleaved reuse phase.
        hot_prefixes = [
            build_shared_prefix(prompt_repeat, f"transfer-hot-{group:04d}")
            for group in range(memory_skew_prefix_groups)
        ]
        warmup_end = max(1, num_prompts // 4)
        pressure_end = max(warmup_end + 1, num_prompts // 2)
    prompts = []
    for i in range(num_prompts):
        suffix = SUFFIXES[i % len(SUFFIXES)]
        if workload == "locality":
            shared_prefix = locality_prefixes[locality_group_order[i]]
        elif workload == "load-skew":
            shared_prefix = hot_prefix if i % 4 != 0 else cold_prefixes[(i // 4) % len(cold_prefixes)]
        elif workload == "memory-skew":
            if i < warmup_end:
                shared_prefix = hot_prefixes[i % memory_skew_prefix_groups]
            elif i >= pressure_end:
                shared_prefix = hot_prefixes[(i - pressure_end) % memory_skew_prefix_groups]
            else:
                shared_prefix = build_shared_prefix(
                    max(1, prompt_repeat // 2),
                    f"pressure-{i - warmup_end:04d}",
                )
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


def resolve_memory_skew_source_ranks(config: dict) -> list[int]:
    """Resolve benchmark placement without granting topology to a baseline policy."""
    explicit = [
        int(rank)
        for rank in config.get("benchmark_memory_skew_source_ranks", [])
        if 0 <= int(rank) < config["world_size"]
    ]
    if explicit:
        return sorted(set(explicit))
    pairs = config.get("nvlink_topo", {}).get("pairs") or []
    return sorted({int(pair[0]) for pair in pairs}) or [0]


def resolve_memory_skew_prefix_groups(num_prompts: int, requested: int) -> int:
    """Resolve enough hot groups to avoid one-prefix baseline saturation."""
    warmup_requests = max(1, num_prompts // 4)
    reuse_requests = num_prompts - max(warmup_requests + 1, num_prompts // 2)
    maximum = min(warmup_requests, reuse_requests)
    if requested > 0:
        if requested > maximum:
            raise ValueError(
                "--memory-skew-prefix-groups must fit in both the warmup and reuse phases"
            )
        return requested
    automatic = min(15, maximum)
    return automatic if automatic % 2 == 1 else max(1, automatic - 1)


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
            e2es.append(latency)
            if seq.num_completion_tokens == 1:
                ttfts.append(latency)
            if latency <= goodput_e2e_sla_s:
                goodput_tokens += 1
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
            mean_ttpt_s=_mean(ttfts),
            p50_ttpt_s=_median(ttfts),
            p90_ttpt_s=_percentile(ttfts, 0.90),
            p95_ttpt_s=_percentile(ttfts, 0.95),
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


def measure_single_gpu_prefix_hit_rate(tokenizer, prompts: list[str], block_size: int, max_cached_blocks: int) -> float:
    # 使用足够大的离线页表计算 workload 自身的理论命中上界；该值不受运行时 KV budget 影响。
    seqs = compute_prefix_hashes(tokenizer, prompts, block_size)
    simulation_blocks = max(max_cached_blocks, sum(seq.num_blocks for seq in seqs), 1)
    gbm = GlobalBlockManager(
        rank=0,
        world_size=1,
        num_blocks_per_gpu=simulation_blocks,
        nvlink_pairs=[],
    )
    bm = BlockManager(num_blocks=simulation_blocks, block_size=block_size, gbm=gbm)
    scheduler = GlobalScheduler(gbm=gbm, block_manager=bm)

    hits = 0
    for seq in seqs:
        prefix_hash = scheduler._compute_prefix_hash(seq)
        _, route_info = scheduler.route_sequence_meta(
            requester_rank=0,
            seq_id=seq.seq_id,
            num_tokens=seq.num_tokens,
            num_blocks=seq.num_blocks,
            prefix_hash=prefix_hash,
            return_info=True,
        )
        if route_info.get("prefix_hit"):
            hits += 1
        bm.allocate(seq)
        # The real data plane publishes a block only after prefill has written
        # valid KV. Mirror that lifecycle in this synthetic baseline, then
        # release the request reference while retaining its reusable cache.
        bm.mark_kv_ready([seq])
        bm.deallocate(seq)
    return hits / max(len(seqs), 1)


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
    ttpts: list[float] = []
    e2es: list[float] = []
    total_tokens = 0
    goodput_tokens = 0
    route_hits = 0
    routed_to_prefix_owner = 0
    route_count = 0
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
        "reuse": [],
    }
    phase_e2es: dict[str, list[float]] = {
        "warmup": [],
        "pressure": [],
        "reuse": [],
    }
    initial_cached_tokens = 0
    initial_prompt_tokens = 0
    prefill_attempts = 0
    preemption_count = 0
    transfer_count = 0
    transfer_copy_count = 0
    transfer_release_count = 0
    chain_transfer_count = 0
    hot_transfer_block_count = 0
    rebalance_success = 0
    rebalance_fail = 0
    rebalance_fail_reasons: dict[str, int] = {}
    background_copy_success = 0
    background_copy_fail = 0
    background_copy_fail_reasons: dict[str, int] = {}
    rank_stats: dict[int, dict] = {}
    start_wall = time.perf_counter()
    sampler = GpuMetricSampler(interval_s=0.5, world_size=config["world_size"])

    def get_rank_stats(rank: int) -> dict:
        return rank_stats.setdefault(
            int(rank),
            {
                "submitted": 0,
                "warmup_submitted": 0,
                "pressure_submitted": 0,
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
                "decode_tokens": 0,
                "prefill_time_s": 0.0,
                "decode_time_s": 0.0,
                "transfers": 0,
                "copies": 0,
                "released_blocks": 0,
                "chain_transfers": 0,
                "hot_transfer_blocks": 0,
                "rebalance_success": 0,
                "rebalance_fail": 0,
                "background_copy_success": 0,
                "background_copy_fail": 0,
                "max_cached_blocks": 0,
            },
        )

    try:
        sampler.start()

        next_prompt_idx = 0
        finished_count = 0
        completion_times: dict[int, float] = {}
        completion_token_counts: dict[int, int] = {}
        inflight: set[int] = set()
        effective_submit_window = len(prompts) if submit_window <= 0 else max(1, submit_window)
        if workload == "memory-skew":
            warmup_end = max(1, len(prompts) // 4)
            pressure_end = max(warmup_end + 1, len(prompts) // 2)
            phase_ends = [warmup_end, pressure_end, len(prompts)]
            source_ranks = resolve_memory_skew_source_ranks(config)
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
                if frequency >= 2
            }
        else:
            warmup_end = pressure_end = 0
            phase_ends = [len(prompts)]
            source_ranks = [0]
            hot_prefix_hashes = set()
        current_phase_index = 0
        current_phase_end = phase_ends[current_phase_index]

        def submit_prompt(prompt: str, prompt_index: int):
            nonlocal route_hits, routed_to_prefix_owner, route_count
            nonlocal route_matched_blocks, route_full_blocks
            nonlocal reclaimable_capacity_routes
            seq = Sequence(
                token_ids=tokenizer.encode(prompt),
                block_size=config["block_size"],
                sampling_params=sampling_params,
            )
            start = time.perf_counter()
            target_rank = 0
            if workload == "memory-skew" and prompt_index < warmup_end:
                # Deterministically create placement skew: warm-up and pressure
                # use only the source side of each NVLink pair. Reuse returns to
                # the scenario's normal routing/round-robin policy.
                prefix_group = prompt_index % config["memory_skew_prefix_groups"]
                target_rank = source_ranks[prefix_group % len(source_ranks)]
            elif workload == "memory-skew" and prompt_index < pressure_end:
                target_rank = source_ranks[(prompt_index - warmup_end) % len(source_ranks)]
            elif route_mode == "control_plane" and engine.control_plane_client is not None:
                # 控制面模式：每个请求都先做 prefix hash，再让全局调度器决定落在哪张卡
                routed = engine.control_plane_client.route_sequence(seq, return_meta=True)
                target_rank = routed["target_rank"]
                route_info = routed.get("route_info", {})
                route_count += 1
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
            if workload == "memory-skew":
                if prompt_index < warmup_end:
                    phase_name = "warmup"
                elif prompt_index < pressure_end:
                    phase_name = "pressure"
                else:
                    phase_name = "reuse"
                phase_by_seq[seq.seq_id] = phase_name
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
                rank_data["copies"] += int(item.get("transfer_copy_count", 0))
                rank_data["released_blocks"] += int(item.get("transfer_release_count", 0))
                rank_data["chain_transfers"] += int(item.get("chain_transfer_count", 0))
                rank_data["hot_transfer_blocks"] += hot_transferred
                rank_data["rebalance_success"] += int(item.get("rebalance_success", 0))
                rank_data["rebalance_fail"] += int(item.get("rebalance_fail", 0))
                rank_data["background_copy_success"] += int(item.get("background_copy_success", 0))
                rank_data["background_copy_fail"] += int(item.get("background_copy_fail", 0))
                rank_data["preemption_count"] += int(item.get("preemption_count", 0))
                rank_data["prefill_tokens"] += int(item.get("prefill_tokens", 0))
                rank_data["decode_tokens"] += int(item.get("decode_tokens", 0))
                rank_data["prefill_time_s"] += float(item.get("prefill_time_s", 0.0))
                rank_data["decode_time_s"] += float(item.get("decode_time_s", 0.0))
                rank_data["first_tokens"] += int(item.get("first_tokens", 0))
                rank_data["finished"] += int(item.get("finished", 0))
                rank_data["output_tokens"] += int(item.get("output_tokens", 0))
                for reason, count in item.get("rebalance_fail_reasons", {}).items():
                    rebalance_fail_reasons[reason] = rebalance_fail_reasons.get(reason, 0) + int(count)
                for reason, count in item.get("background_copy_fail_reasons", {}).items():
                    background_copy_fail_reasons[reason] = (
                        background_copy_fail_reasons.get(reason, 0) + int(count)
                    )
            for seq_id, _token in first_tokens:
                if seq_id in submit_times:
                    ttft = now - submit_times[seq_id]
                    ttfts.append(ttft)
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
                latency = now - submit_times[seq_id]
                output_tokens = max(len(tokens), 1)
                ttpts.append(latency / output_tokens)
                e2es.append(latency)
                phase = phase_by_seq.get(seq_id)
                if phase in phase_e2es:
                    phase_e2es[phase].append(latency)
                completion_times[seq_id] = now
                completion_token_counts[seq_id] = len(tokens)
            for rank, stats in rank_stats.items():
                stats["local_prefix_hit_rate"] = (
                    stats["prefill_prefix_hits"] / max(stats["prefill_requests"], 1)
                )
                stats["initial_cached_token_ratio"] = (
                    stats["initial_cached_tokens"] / max(stats["initial_prompt_tokens"], 1)
                )
            if (
                workload == "memory-skew"
                and not inflight
                and next_prompt_idx >= current_phase_end
                and current_phase_index + 1 < len(phase_ends)
            ):
                current_phase_index += 1
                current_phase_end = phase_ends[current_phase_index]
            while next_prompt_idx < current_phase_end and len(inflight) < effective_submit_window:
                submit_prompt(prompts[next_prompt_idx], next_prompt_idx)
                next_prompt_idx += 1
        elapsed = time.perf_counter() - start_wall
    finally:
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
        mean_ttpt_s=_mean(ttpts),
        p50_ttpt_s=_median(ttpts),
        p90_ttpt_s=_percentile(ttpts, 0.90),
        p95_ttpt_s=_percentile(ttpts, 0.95),
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
        transfer_copy_count=transfer_copy_count,
        transfer_release_count=transfer_release_count,
        chain_transfer_count=chain_transfer_count,
        hot_transfer_block_count=hot_transfer_block_count,
        hot_transfer_block_ratio=hot_transfer_block_count / max(transfer_count, 1),
        rebalance_success=rebalance_success,
        rebalance_fail=rebalance_fail,
        rebalance_fail_reasons=rebalance_fail_reasons,
        background_copy_success=background_copy_success,
        background_copy_fail=background_copy_fail,
        background_copy_fail_reasons=background_copy_fail_reasons,
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
        phase_latency_stats={
            phase: {
                "requests": float(len(phase_e2es[phase])),
                "mean_ttft_s": _mean(phase_ttfts[phase]),
                "p90_ttft_s": _percentile(phase_ttfts[phase], 0.90),
                "mean_e2e_s": _mean(phase_e2es[phase]),
                "p90_e2e_s": _percentile(phase_e2es[phase], 0.90),
            }
            for phase in ("warmup", "pressure", "reuse")
            if phase_e2es[phase]
        },
    )


def make_config(world_size: int, enable_global_pool: bool, nvlink_pairs: list[tuple[int, int]] | None) -> dict:
    # benchmark 里统一通过这层构造 config，避免每个场景单独拼参数时漏掉关键项
    config = dict(MODEL_CONFIG)
    config["world_size"] = world_size
    config["enable_global_pool"] = enable_global_pool
    if nvlink_pairs is not None:
        config["nvlink_topo"] = {"pairs": nvlink_pairs}
    config["use_control_plane_process"] = enable_global_pool
    return config


def aggregate_scenario_trials(trials: list[ScenarioResult]) -> ScenarioResult:
    """Return per-scenario means and key run-to-run standard deviations."""
    if not trials:
        raise ValueError("at least one scenario trial is required")
    if len(trials) == 1:
        return trials[0]

    def mean_attr(name: str) -> float:
        return statistics.fmean(float(getattr(result, name)) for result in trials)

    def mean_reason_map(name: str) -> dict[str, int]:
        keys = set().union(*(getattr(result, name).keys() for result in trials))
        return {
            key: round(statistics.fmean(getattr(result, name).get(key, 0) for result in trials))
            for key in keys
        }

    rank_ids = sorted(set().union(*(result.rank_stats.keys() for result in trials)))
    rank_stats = {}
    for rank in rank_ids:
        keys = set().union(*(result.rank_stats.get(rank, {}).keys() for result in trials))
        rank_stats[rank] = {}
        for key in keys:
            values = [result.rank_stats.get(rank, {}).get(key, 0.0) for result in trials]
            if all(isinstance(value, (int, float)) for value in values):
                rank_stats[rank][key] = statistics.fmean(float(value) for value in values)

    phase_names = sorted(set().union(*(
        set((result.phase_latency_stats or {}).keys())
        for result in trials
    )))
    phase_latency_stats = {}
    for phase in phase_names:
        metric_names = set().union(*(
            set((result.phase_latency_stats or {}).get(phase, {}).keys())
            for result in trials
        ))
        phase_latency_stats[phase] = {
            metric: statistics.fmean(
                float((result.phase_latency_stats or {}).get(phase, {}).get(metric, 0.0))
                for result in trials
            )
            for metric in metric_names
        }

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
        mean_ttpt_s=mean_attr("mean_ttpt_s"),
        p50_ttpt_s=mean_attr("p50_ttpt_s"),
        p90_ttpt_s=mean_attr("p90_ttpt_s"),
        p95_ttpt_s=mean_attr("p95_ttpt_s"),
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
        transfer_copy_count=round(mean_attr("transfer_copy_count")),
        transfer_release_count=round(mean_attr("transfer_release_count")),
        chain_transfer_count=round(mean_attr("chain_transfer_count")),
        hot_transfer_block_count=round(mean_attr("hot_transfer_block_count")),
        hot_transfer_block_ratio=mean_attr("hot_transfer_block_ratio"),
        rebalance_success=round(mean_attr("rebalance_success")),
        rebalance_fail=round(mean_attr("rebalance_fail")),
        rebalance_fail_reasons=mean_reason_map("rebalance_fail_reasons"),
        background_copy_success=round(mean_attr("background_copy_success")),
        background_copy_fail=round(mean_attr("background_copy_fail")),
        background_copy_fail_reasons=mean_reason_map("background_copy_fail_reasons"),
        gpu_util_mean=mean_attr("gpu_util_mean"),
        gpu_util_p95=mean_attr("gpu_util_p95"),
        gpu_mem_util_mean=mean_attr("gpu_mem_util_mean"),
        gpu_mem_util_p95=mean_attr("gpu_mem_util_p95"),
        rank_stats=rank_stats,
        theoretical_prefix_hit_rate=mean_attr("theoretical_prefix_hit_rate"),
        route_matched_block_ratio=mean_attr("route_matched_block_ratio"),
        reclaimable_capacity_route_rate=mean_attr("reclaimable_capacity_route_rate"),
        stale_route_hit_rate=mean_attr("stale_route_hit_rate"),
        reuse_phase_request_hit_rate=mean_attr("reuse_phase_request_hit_rate"),
        reuse_phase_token_ratio=mean_attr("reuse_phase_token_ratio"),
        repetitions=len(trials),
        throughput_tok_s_std=statistics.pstdev(result.throughput_tok_s for result in trials),
        goodput_tok_s_std=statistics.pstdev(result.goodput_tok_s for result in trials),
        mean_ttft_s_std=statistics.pstdev(result.mean_ttft_s for result in trials),
        mean_e2e_s_std=statistics.pstdev(result.mean_e2e_s for result in trials),
        phase_latency_stats=phase_latency_stats,
    )


def run_repeated_engine_scenario(repetitions: int, **kwargs) -> ScenarioResult:
    trials = []
    for trial in range(repetitions):
        if repetitions > 1:
            print(f"[{kwargs['name']}] trial {trial + 1}/{repetitions}")
        trials.append(run_engine_scenario(**kwargs))
    return aggregate_scenario_trials(trials)


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_summary_table(results: list[ScenarioResult | None]):
    # 横向总表：把所有场景放在同一张表里，便于直接看五种配置的整体差异。
    valid_results = [result for result in results if result is not None]
    print("\nBenchmark Summary")
    print("=" * 225)
    print(
        f"{'scenario':<22} {'tput(tok/s)':>14} {'goodput':>12} {'ttft(ms)':>12} {'ttpt(ms)':>12} "
        f"{'e2e(ms)':>12} {'p90(e2e)':>12} {'p95(e2e)':>12} {'gpu util':>10} {'mem util':>10} "
        f"{'CP req hit':>11} {'CP owner':>11} {'DP req hit':>11} {'DP tok reuse':>12} "
        f"{'attempts':>9} {'preempt':>8} {'redund tok':>10} {'sent blk':>9} {'retained':>8} "
        f"{'fg ok':>7} {'fg fail':>8} {'bg ok':>7} {'bg fail':>8} "
        f"{'pinned':>8} {'no space':>9} {'no plan':>8} {'low value':>9} {'bg space':>8}"
    )
    for result in valid_results:
        print(
            f"{result.name:<22} "
            f"{result.throughput_tok_s:>14.2f} "
            f"{result.goodput_tok_s:>12.2f} "
            f"{result.mean_ttft_s * 1000:>12.2f} "
            f"{result.mean_ttpt_s * 1000:>12.2f} "
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
            f"{result.rebalance_success:>7} "
            f"{result.rebalance_fail:>8} "
            f"{result.background_copy_success:>7} "
            f"{result.background_copy_fail:>8} "
            f"{result.rebalance_fail_reasons.get('pinned_source', 0):>8} "
            f"{result.rebalance_fail_reasons.get('no_target_space', 0):>9} "
            f"{result.rebalance_fail_reasons.get('no_plan', 0):>8} "
            f"{result.rebalance_fail_reasons.get('low_benefit', 0):>9} "
            f"{result.background_copy_fail_reasons.get('no_target_space', 0):>8}"
        )

    print("\nPrefix Diagnostics")
    print("=" * 140)
    print(
        f"{'scenario':<22} {'trace upper':>12} {'CP blk match':>13} "
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
            f"{fmt_pct(result.theoretical_prefix_hit_rate):>12} "
            f"{fmt_pct(result.route_matched_block_ratio):>13} "
            f"{fmt_pct(result.route_hit_rate):>11} "
            f"{fmt_pct(result.reclaimable_capacity_route_rate):>12} "
            f"{fmt_pct(result.stale_route_hit_rate):>12} "
            f"{fmt_pct(result.prefix_hit_rate):>11} "
            f"{fmt_pct(result.initial_cached_token_ratio):>12} "
            f"{block_caps:>20}"
        )

    print("\nTransfer Diagnostics")
    print("=" * 151)
    print(
        f"{'scenario':<22} {'sent blocks':>12} {'source kept':>12} "
        f"{'source freed':>13} {'chain plans':>12} {'hot sent':>10} {'hot ratio':>11} "
        f"{'reuse req hit':>14} "
        f"{'reuse tok ratio':>15} {'fg ok':>8} {'fg fail':>9}"
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
            f"{result.rebalance_success:>8} "
            f"{result.rebalance_fail:>9}"
        )

    if any(result.phase_latency_stats for result in valid_results):
        print("\nMemory-Skew Phase Latency")
        print("=" * 104)
        print(
            f"{'scenario':<22} {'phase':<10} {'requests':>10} "
            f"{'mean TTFT(ms)':>15} {'p90 TTFT(ms)':>15} "
            f"{'mean E2E(ms)':>15} {'p90 E2E(ms)':>15}"
        )
        for result in valid_results:
            for phase in ("warmup", "pressure", "reuse"):
                stats = (result.phase_latency_stats or {}).get(phase)
                if not stats:
                    continue
                print(
                    f"{result.name:<22} {phase:<10} "
                    f"{int(round(stats.get('requests', 0.0))):>10} "
                    f"{stats.get('mean_ttft_s', 0.0) * 1000:>15.2f} "
                    f"{stats.get('p90_ttft_s', 0.0) * 1000:>15.2f} "
                    f"{stats.get('mean_e2e_s', 0.0) * 1000:>15.2f} "
                    f"{stats.get('p90_e2e_s', 0.0) * 1000:>15.2f}"
                )

    if any(result.repetitions > 1 for result in valid_results):
        print("\nRepeated-run variability (mean +/- population stddev)")
        print(f"{'scenario':<22} {'throughput(tok/s)':>24} {'goodput(tok/s)':>24} {'TTFT(ms)':>24} {'E2E(ms)':>24}")
        for result in valid_results:
            print(
                f"{result.name:<22} "
                f"{result.throughput_tok_s:>10.2f} +/- {result.throughput_tok_s_std:<8.2f} "
                f"{result.goodput_tok_s:>10.2f} +/- {result.goodput_tok_s_std:<8.2f} "
                f"{result.mean_ttft_s * 1000:>10.2f} +/- {result.mean_ttft_s_std * 1000:<8.2f} "
                f"{result.mean_e2e_s * 1000:>10.2f} +/- {result.mean_e2e_s_std * 1000:<8.2f}"
            )


def save_summary_figure(results: list[ScenarioResult | None], output_path: str) -> None:
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
    ttft_ms = [result.mean_ttft_s * 1000.0 for result in valid_results]
    ttpt_ms = [result.mean_ttpt_s * 1000.0 for result in valid_results]
    e2e_ms = [result.mean_e2e_s * 1000.0 for result in valid_results]
    p90_e2e_ms = [result.p90_e2e_s * 1000.0 for result in valid_results]
    route_hit_pct = [result.route_hit_rate * 100.0 for result in valid_results]
    owner_hit_pct = [result.routed_to_prefix_owner_rate * 100.0 for result in valid_results]
    local_hit_pct = [result.prefix_hit_rate * 100.0 for result in valid_results]
    cached_token_pct = [result.initial_cached_token_ratio * 100.0 for result in valid_results]
    gpu_util = [result.gpu_util_mean if result.gpu_util_mean is not None else 0.0 for result in valid_results]
    gpu_mem_util = [result.gpu_mem_util_mean if result.gpu_mem_util_mean is not None else 0.0 for result in valid_results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Shared Prefix Benchmark Summary", fontsize=16)
    palettes = {
        "throughput": ["#0072B2", "#E69F00"],
        "latency": ["#009E73", "#D55E00", "#CC79A7", "#56B4E9"],
        "hit": ["#332288", "#117733", "#DDCC77", "#CC6677"],
        "util": ["#882255", "#44AA99"],
    }
    bar_style = {"edgecolor": "#333333", "linewidth": 0.45}

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
        width=width,
        label="throughput",
        color=palettes["throughput"][0],
        **bar_style,
    )
    annotate_bars(axes[0, 0], bars)
    bars = axes[0, 0].bar(
        [i + width / 2 for i in x],
        goodput,
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
        width=latency_width,
        label="TTFT mean",
        color=palettes["latency"][0],
        **bar_style,
    )
    annotate_bars(axes[0, 1], bars, decimals=0)
    bars = axes[0, 1].bar(
        [i - 0.5 * latency_width for i in x],
        ttpt_ms,
        width=latency_width,
        label="TTPT mean",
        color=palettes["latency"][1],
        **bar_style,
    )
    annotate_bars(axes[0, 1], bars, decimals=0)
    bars = axes[0, 1].bar(
        [i + 0.5 * latency_width for i in x],
        e2e_ms,
        width=latency_width,
        label="E2E mean",
        color=palettes["latency"][2],
        **bar_style,
    )
    annotate_bars(axes[0, 1], bars, decimals=0)
    bars = axes[0, 1].bar(
        [i + 1.5 * latency_width for i in x],
        p90_e2e_ms,
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


def save_rank_stats_figure(results: list[ScenarioResult | None], output_path: str) -> None:
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
    fig.suptitle("Per-Rank Benchmark Diagnostics", fontsize=16)
    rank_colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]

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


def save_summary_json(results: dict, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved json: {output}")


def parse_args():
    # benchmark 入口参数尽量保持简单：只暴露场景规模、模型、拓扑和 SLA
    parser = argparse.ArgumentParser(description="Shared-prefix high-concurrency benchmark")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--prompt-repeat", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--workload", choices=["locality", "load-skew", "memory-skew"], default="locality")
    parser.add_argument("--locality-prefix-groups", type=int, default=16)
    parser.add_argument("--memory-skew-prefix-groups", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--model-name-or-path", type=str, default=MODEL_CONFIG["model_name_or_path"])
    parser.add_argument("--nvlink-pairs", type=str, default="0,1")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--kv-block-budget", type=int, default=None)
    parser.add_argument("--routing-max-cached-blocks", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eviction-max-cached-blocks", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--goodput-e2e-sla-ms", type=float, default=2000.0)
    parser.add_argument("--skip-pool", action="store_true")
    parser.add_argument("--output-figure", type=str, default="")
    parser.add_argument("--submit-window", type=int, default=8)
    parser.add_argument("--disable-background-copy", action="store_true")
    parser.add_argument("--background-copy-max-blocks", type=int, default=MODEL_CONFIG["background_copy_max_blocks"])
    parser.add_argument("--background-copy-cooldown-s", type=float, default=MODEL_CONFIG["background_copy_cooldown_s"])
    parser.add_argument("--background-copy-hot-threshold", type=int, default=MODEL_CONFIG["background_copy_hot_threshold"])
    parser.add_argument("--route-load-weight", type=float, default=MODEL_CONFIG["route_load_weight"])
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
    parser.add_argument("--route-cache-queue-slack", type=float, default=MODEL_CONFIG["route_cache_queue_slack"])
    return parser.parse_args()


def resolve_kv_block_budget(args) -> int:
    """Resolve the one per-rank KV capacity used by every scenario."""
    requested_budgets = {
        value
        for value in (
            args.kv_block_budget,
            args.routing_max_cached_blocks,
            args.eviction_max_cached_blocks,
        )
        if value is not None
    }
    if len(requested_budgets) > 1:
        raise ValueError(
            "KV block budgets must be equal across scenarios. Use one "
            "--kv-block-budget value instead of different routing/eviction budgets."
        )
    budget = requested_budgets.pop() if requested_budgets else MODEL_CONFIG["max_cached_blocks"]
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


def apply_background_copy_args(config: dict, args) -> None:
    config["enable_background_copy"] = not args.disable_background_copy
    config["background_copy_max_blocks"] = args.background_copy_max_blocks
    config["background_copy_cooldown_s"] = args.background_copy_cooldown_s
    config["background_copy_hot_threshold"] = args.background_copy_hot_threshold


def apply_route_args(config: dict, args) -> None:
    config["route_load_weight"] = args.route_load_weight
    config["route_load_bypass_threshold"] = args.route_load_bypass_threshold
    config["route_prefill_cost_weight"] = args.route_prefill_cost_weight
    config["route_reclaim_cost_weight"] = args.route_reclaim_cost_weight
    config["foreground_transfer_cost_weight"] = args.foreground_transfer_cost_weight
    config["foreground_transfer_min_benefit_ratio"] = (
        args.foreground_transfer_min_benefit_ratio
    )
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
    try:
        kv_block_budget = resolve_kv_block_budget(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.workload == "locality" and not 1 <= args.locality_prefix_groups <= args.num_prompts:
        raise SystemExit("--locality-prefix-groups must be between 1 and --num-prompts")
    try:
        memory_skew_prefix_groups = resolve_memory_skew_prefix_groups(
            args.num_prompts,
            args.memory_skew_prefix_groups,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    visible_gpus = torch.cuda.device_count()
    if args.world_size > visible_gpus:
        raise SystemExit(
            f"--world-size {args.world_size} exceeds visible CUDA devices {visible_gpus}. "
            "Check CUDA_VISIBLE_DEVICES."
        )

    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = build_prompts(
        tokenizer,
        args.num_prompts,
        args.prompt_repeat,
        args.workload,
        locality_prefix_groups=args.locality_prefix_groups,
        memory_skew_prefix_groups=memory_skew_prefix_groups,
        seed=args.seed,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        max_model_length=MODEL_CONFIG["max_model_length"],
    )
    goodput_e2e_sla_s = args.goodput_e2e_sla_ms / 1000.0
    nvlink_pairs = parse_pairs(args.nvlink_pairs) if args.nvlink_pairs else []
    memory_skew_source_ranks = sorted({int(pair[0]) for pair in nvlink_pairs}) or [0]

    def apply_memory_skew_placement(config: dict) -> None:
        # Workload placement must be identical across all multi-GPU scenarios,
        # including baselines that intentionally do not expose topology to the
        # engine policy itself.
        config["benchmark_memory_skew_source_ranks"] = (
            [0] if config["world_size"] == 1 else memory_skew_source_ranks
        )
        config["memory_skew_prefix_groups"] = memory_skew_prefix_groups

    # single-gpu baseline：单卡独立执行，不启用全局池
    baseline_config = make_config(1, False, None)
    baseline_config["model_name_or_path"] = model_name
    baseline_config["max_cached_blocks"] = kv_block_budget
    baseline_config["random_seed"] = args.seed
    apply_memory_skew_placement(baseline_config)
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
    theoretical_prefix_hit_rate = measure_single_gpu_prefix_hit_rate(
        tokenizer,
        prompts,
        block_size=baseline_config["block_size"],
        max_cached_blocks=baseline_config["max_cached_blocks"],
    )
    baseline.theoretical_prefix_hit_rate = theoretical_prefix_hit_rate

    # multi-gpu baseline：不共享 KV、不走控制面路由，但请求通过 round-robin 分发到多张卡
    multi_gpu_config = make_config(args.world_size, False, None)
    multi_gpu_config["model_name_or_path"] = model_name
    multi_gpu_config["max_cached_blocks"] = kv_block_budget
    multi_gpu_config["random_seed"] = args.seed
    apply_memory_skew_placement(multi_gpu_config)
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
    routing_config = make_config(args.world_size, True, nvlink_pairs or None)
    routing_config["model_name_or_path"] = model_name
    routing_config["max_cached_blocks"] = kv_block_budget
    routing_config["enable_foreground_rebalance"] = False
    routing_config["enable_background_copy"] = False
    routing_config["random_seed"] = args.seed
    apply_memory_skew_placement(routing_config)
    apply_route_args(routing_config, args)
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
    eviction_config = make_config(args.world_size, True, nvlink_pairs or None)
    eviction_config["model_name_or_path"] = model_name
    eviction_config["max_cached_blocks"] = kv_block_budget
    eviction_config["random_seed"] = args.seed
    apply_memory_skew_placement(eviction_config)
    eviction_config["preserve_cache_via_transfer"] = args.workload == "memory-skew"
    apply_background_copy_args(eviction_config, args)
    apply_route_args(eviction_config, args)
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
    if not args.skip_pool:
        if visible_gpus < args.world_size:
            print(f"pool scenario skipped: need {args.world_size} CUDA devices")
        else:
            # multi-gpu-lmpool：真实全局池化路径，控制面路由 + 数据面执行一起跑。
            pool_pairs = nvlink_pairs or None
            pool_config = make_config(args.world_size, True, pool_pairs)
            pool_config["model_name_or_path"] = model_name
            pool_config["max_cached_blocks"] = kv_block_budget
            pool_config["random_seed"] = args.seed
            apply_memory_skew_placement(pool_config)
            pool_config["preserve_cache_via_transfer"] = args.workload == "memory-skew"
            apply_background_copy_args(pool_config, args)
            apply_route_args(pool_config, args)
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
            result.theoretical_prefix_hit_rate = theoretical_prefix_hit_rate
    print_summary_table(all_results)
    if args.output_figure:
        save_summary_figure(all_results, args.output_figure)
        save_rank_stats_figure(all_results, args.output_figure)
    if args.output_json:
        payload = {
            "single-gpu": asdict(baseline),
            "multi-gpu": asdict(independent_result) if independent_result is not None else None,
            "multi-gpu-kv-routing": asdict(kv_routing) if kv_routing is not None else None,
            "multi-gpu-kv-transfer": asdict(kv_eviction) if kv_eviction is not None else None,
            "multi-gpu-lmpool": asdict(pool_result) if pool_result is not None else None,
        }
        save_summary_json(payload, args.output_json)


if __name__ == "__main__":
    main()
