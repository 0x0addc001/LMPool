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
    --world-size 2 \
    --nvlink-pairs "0,1" \
    --submit-window 5 \
    --background-copy-max-blocks 1 \
    --background-copy-cooldown-s 2.0 \
    --background-copy-hot-threshold 3 \
    --goodput-e2e-sla-ms 10000 \
    --output-json ./benchmarks/results/shared_prefix_benchmark_202607121117.json \
    --output-figure ./benchmarks/results/shared_prefix_benchmark_202607121117.png

参数说明：
1. `--num-prompts`：
  本次压测总共生成多少条请求。脚本会为每条请求构造同一个共享前缀和不同后缀；
  值越大，并发压力越高，统计结果也更稳定。

2. `--prompt-repeat`：
  共享前缀重复多少次。值越大，公共前缀越长，越容易观察 prefix cache / 路由收益。

3. `--max-tokens`：
  每条请求最多生成多少个输出 token。值越大，decode 阶段占比越高，也更容易触发 transfer 压力。

4. `--temperature`：
  采样温度。benchmark 默认主要看系统性能，通常保持固定值即可，不建议在不同实验间频繁改动。

5. `--output-json`：
  将各场景统计结果导出到指定 JSON 文件。脚本会自动创建父目录，并在成功后打印
  `saved json: ...`。

6. `--model-name-or-path`：
  指定要测试的模型名称或本地路径。默认使用脚本里的 Qwen 配置，对应模型结构也基于这份配置。

7. `--nvlink-pairs`：
  手动指定 NVLink 拓扑，格式如 `0,1` 或 `0,1;2,3`。这里使用的是
  `CUDA_VISIBLE_DEVICES` 之后的逻辑 GPU 编号，不是物理 GPU 编号。如果不想手动写，
  可以传空字符串，让底层逻辑尝试解析 `nvidia-smi topo -m`。命令行里包含分号时必须加引号，
  例如 `--nvlink-pairs "0,2;1,3;4,5;6,7"`。

8. `--world-size`：
  多卡场景启动多少个 data-plane worker。默认 2；八卡实验需要显式传 `--world-size 8`。
  该值不能超过 `CUDA_VISIBLE_DEVICES` 暴露出的 GPU 数。

9. `--routing-max-cached-blocks`：
  `multi-gpu-kv-routing` 场景的 KV block 上限。该场景启用控制面路由和全局页表，
  但显式关闭 foreground rebalance 和 background copy，主要用来观察 prefix-hit routing 的收益和控制面开销。

10. `--eviction-max-cached-blocks`：
  `multi-gpu-kv-transfer` 场景的 KV block 上限。通常设得更小，用来更容易触发
  transfer / rebalance；请求仍按 round-robin 下发，用来隔离跨 GPU KV 搬运开销。

11. `--goodput-e2e-sla-ms`：
  goodput 的端到端延迟门槛，单位毫秒。只有在这个 SLA 内完成的请求，其输出 token
  才计入 goodput。因此表里的 goodput 单位是 tokens/s，不是 requests/s。

12. `--skip-pool`：
  跳过 `multi-gpu-lmpool` 场景，只跑基线、routing 和 transfer。

13. `--output-figure`：
  将五种场景的核心指标画成一张图表图片并保存到指定路径。脚本会自动创建父目录，
  使用无显示环境可用的 Matplotlib Agg 后端，并在成功后打印 `saved figure: ...`。

14. `--submit-window`：
  benchmark 中允许同时在途的请求数。值越大越接近一次性高并发提交；值越小越容易让前面请求先完成
  prefill 并上报全局页表，从而观察在线 prefix reuse。设为 0 或负数表示一次性提交全部请求。
  如果要验证 prefix hit 是否生效，建议先用 4 ~ 8；如果要模拟 burst 流量，可以设为 0 或 -1。

15. `--disable-background-copy`：
  关闭后台 speculative copy-style transfer。默认开启，用于把热点 prefix block 异步复制到 NVLink
  伙伴，服务后续请求；关闭后只保留前台 move-style transfer。

16. `--background-copy-max-blocks`：
  每次后台 copy 最多复制多少个 prefix block。值越大越可能提高后续本地命中，但会占用更多
  NCCL / worker 时间；排查功能正确性可先用 1，验证收益时建议尝试 2。

17. `--background-copy-cooldown-s`：
  同一个 prefix 在同一组 `src -> dst` GPU 之间再次触发后台 copy 的最短间隔，单位秒。
  值越大越保守，值越小越容易在高并发下产生更多 transfer。验证后台 copy 收益时可尝试 0.5。

18. `--background-copy-hot-threshold`：
  同一个 prefix 至少被路由命中多少次后才允许后台 copy。值越大越保守，能减少无效 copy；
  值为 1 时退化为旧的 eager speculative copy。

说明：
1. 建议显式设置 CUDA_VISIBLE_DEVICES，避免在共享机器上误用其他 GPU。
2. 如果物理 GPU 0 和 2 之间有 NVLink，可以使用 `CUDA_VISIBLE_DEVICES=0,2`。
   但脚本内部看到的是重映射后的逻辑 GPU `0,1`，因此 `--nvlink-pairs` 应写成 `0,1`，而不是 `0,2`。
3. `multi-gpu`，`multi-gpu-kv-transfer` 场景当前采用 round-robin 分发。
4. 表里的 prefix hit 是 worker 在 prefill 时实际观察到的本地 prefix cache 命中率，
   round-robin 基线也会统计，因此可横向对比。它不是控制面路由命中率。
5. `multi-gpu-lmpool` 需要至少 2 张可见 CUDA GPU。
6. 如果要专门验证后台 speculative copy-style transfer，建议先用
   `--eviction-max-cached-blocks 32 --background-copy-max-blocks 2 --background-copy-cooldown-s 0.5`。
   `--eviction-max-cached-blocks 8` 更适合压力/失败原因分析，不适合证明 copy 收益。
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import statistics
import sys
import threading
import time
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
    "route_load_weight": 0.01,
    "route_waiting_token_weight": 1.0,
    "route_running_token_weight": 0.25,
    "route_running_sequence_weight": 32.0,
    "route_load_bypass_threshold": 512.0,
    "route_cache_queue_slack": 512.0,
    "enable_foreground_rebalance": True,
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
    transfer_count: int
    transfer_copy_count: int
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


def build_shared_prefix(prompt_repeat: int) -> str:
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
    return " ".join([block] * prompt_repeat)


def build_prompts(tokenizer, num_prompts: int, prompt_repeat: int) -> list[str]:
    # 每个 prompt 都共享同一段前缀，只在尾部附加不同任务，适合测前缀复用和路由收益
    shared_prefix = build_shared_prefix(prompt_repeat)
    prompts = []
    for i in range(num_prompts):
        suffix = SUFFIXES[i % len(SUFFIXES)]
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


def _sample_gpu_metrics_once() -> list[tuple[float, float]]:
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
    for line in output.strip().splitlines():
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
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: list[list[tuple[float, float]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            samples = _sample_gpu_metrics_once()
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
    sampler = GpuMetricSampler(interval_s=0.5)
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
            transfer_count=0,
            transfer_copy_count=0,
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
    # 单卡基线里也会用同样的 prefix hash 逻辑，主要用于计算“理论上能命中多少前缀”
    gbm = GlobalBlockManager(
        rank=0,
        world_size=1,
        num_blocks_per_gpu=max_cached_blocks,
        nvlink_pairs=[],
    )
    bm = BlockManager(num_blocks=max_cached_blocks, block_size=block_size, gbm=gbm)
    scheduler = GlobalScheduler(gbm=gbm, block_manager=bm)

    hits = 0
    seqs = compute_prefix_hashes(tokenizer, prompts, block_size)
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
) -> ScenarioResult:
    # 调用 LLMEngine
    # prompt 先进入 launcher
    # 再由控制面路由或按 round-robin 分发
    # worker 侧执行 prefill / decode
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
    prefill_seen_seq_ids: set[int] = set()
    prefill_hit_seq_ids: set[int] = set()
    transfer_count = 0
    transfer_copy_count = 0
    rebalance_success = 0
    rebalance_fail = 0
    rebalance_fail_reasons: dict[str, int] = {}
    background_copy_success = 0
    background_copy_fail = 0
    background_copy_fail_reasons: dict[str, int] = {}
    start_wall = time.perf_counter()
    sampler = GpuMetricSampler(interval_s=0.5)

    try:
        sampler.start()

        next_prompt_idx = 0
        finished_count = 0
        completion_times: dict[int, float] = {}
        completion_token_counts: dict[int, int] = {}
        inflight: set[int] = set()
        effective_submit_window = len(prompts) if submit_window <= 0 else max(1, submit_window)

        def submit_prompt(prompt: str):
            nonlocal route_hits, routed_to_prefix_owner, route_count
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
                route_count += 1
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
            inflight.add(seq.seq_id)

        while next_prompt_idx < len(prompts) and len(inflight) < effective_submit_window:
            submit_prompt(prompts[next_prompt_idx])
            next_prompt_idx += 1

        # 主循环不断泵 worker 消息，直到所有请求都完成
        while finished_count < len(prompts):
            finished, first_tokens, prefill_stats, runtime_stats = engine.step()
            now = time.perf_counter()
            for item in runtime_stats:
                transfer_count += int(item.get("transfer_count", item.get("swap_count", 0)))
                transfer_copy_count += int(item.get("transfer_copy_count", 0))
                rebalance_success += int(item.get("rebalance_success", 0))
                rebalance_fail += int(item.get("rebalance_fail", 0))
                background_copy_success += int(item.get("background_copy_success", 0))
                background_copy_fail += int(item.get("background_copy_fail", 0))
                for reason, count in item.get("rebalance_fail_reasons", {}).items():
                    rebalance_fail_reasons[reason] = rebalance_fail_reasons.get(reason, 0) + int(count)
                for reason, count in item.get("background_copy_fail_reasons", {}).items():
                    background_copy_fail_reasons[reason] = (
                        background_copy_fail_reasons.get(reason, 0) + int(count)
                    )
            for seq_id, _token in first_tokens:
                if seq_id in submit_times:
                    ttfts.append(now - submit_times[seq_id])
            for item in prefill_stats:
                seq_id = item.get("seq_id")
                if seq_id is not None:
                    prefill_seen_seq_ids.add(seq_id)
                if item.get("prefix_hit", False):
                    if seq_id is not None:
                        prefill_hit_seq_ids.add(seq_id)
            for seq_id, tokens in finished:
                inflight.discard(seq_id)
                finished_count += 1
                total_tokens += len(tokens)
                latency = now - submit_times[seq_id]
                output_tokens = max(len(tokens), 1)
                ttpts.append(latency / output_tokens)
                e2es.append(latency)
                completion_times[seq_id] = now
                completion_token_counts[seq_id] = len(tokens)
            while next_prompt_idx < len(prompts) and len(inflight) < effective_submit_window:
                submit_prompt(prompts[next_prompt_idx])
                next_prompt_idx += 1
        elapsed = time.perf_counter() - start_wall
    finally:
        sampler.stop()
        engine.exit()

    gpu_util_mean, gpu_util_p95, gpu_mem_util_mean, gpu_mem_util_p95 = sampler.summarize()
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
        transfer_count=transfer_count,
        transfer_copy_count=transfer_copy_count,
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


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_summary_table(results: list[ScenarioResult | None]):
    # 横向总表：把所有场景放在同一张表里，便于直接看五种配置的整体差异。
    valid_results = [result for result in results if result is not None]
    print("\nBenchmark Summary")
    print("=" * 180)
    print(
        f"{'scenario':<22} {'tput(tok/s)':>14} {'goodput':>12} {'ttft(ms)':>12} {'ttpt(ms)':>12} "
        f"{'e2e(ms)':>12} {'p90(e2e)':>12} {'p95(e2e)':>12} {'gpu util':>10} {'mem util':>10} "
        f"{'route hit':>11} {'owner hit':>11} {'local hit':>11} {'transfers':>9} {'copies':>8} "
        f"{'fg ok':>7} {'fg fail':>8} {'bg ok':>7} {'bg fail':>8} "
        f"{'pinned':>8} {'no space':>9} {'no plan':>8} {'bg space':>8}"
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
            f"{result.transfer_count:>9} "
            f"{result.transfer_copy_count:>8} "
            f"{result.rebalance_success:>7} "
            f"{result.rebalance_fail:>8} "
            f"{result.background_copy_success:>7} "
            f"{result.background_copy_fail:>8} "
            f"{result.rebalance_fail_reasons.get('pinned_source', 0):>8} "
            f"{result.rebalance_fail_reasons.get('no_target_space', 0):>9} "
            f"{result.rebalance_fail_reasons.get('no_plan', 0):>8} "
            f"{result.background_copy_fail_reasons.get('no_target_space', 0):>8}"
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
    gpu_util = [result.gpu_util_mean if result.gpu_util_mean is not None else 0.0 for result in valid_results]
    gpu_mem_util = [result.gpu_mem_util_mean if result.gpu_mem_util_mean is not None else 0.0 for result in valid_results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Shared Prefix Benchmark Summary", fontsize=16)

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
    bars = axes[0, 0].bar([i - width / 2 for i in x], throughput, width=width, label="throughput")
    annotate_bars(axes[0, 0], bars)
    bars = axes[0, 0].bar([i + width / 2 for i in x], goodput, width=width, label="goodput")
    annotate_bars(axes[0, 0], bars)
    axes[0, 0].set_title("Throughput / Goodput")
    axes[0, 0].set_ylabel("tokens/s")
    axes[0, 0].set_xticks(x, names, rotation=15, ha="right")
    axes[0, 0].legend()

    latency_width = 0.2
    bars = axes[0, 1].bar([i - 1.5 * latency_width for i in x], ttft_ms, width=latency_width, label="TTFT mean")
    annotate_bars(axes[0, 1], bars, decimals=0)
    bars = axes[0, 1].bar([i - 0.5 * latency_width for i in x], ttpt_ms, width=latency_width, label="TTPT mean")
    annotate_bars(axes[0, 1], bars, decimals=0)
    bars = axes[0, 1].bar([i + 0.5 * latency_width for i in x], e2e_ms, width=latency_width, label="E2E mean")
    annotate_bars(axes[0, 1], bars, decimals=0)
    bars = axes[0, 1].bar([i + 1.5 * latency_width for i in x], p90_e2e_ms, width=latency_width, label="E2E p90")
    annotate_bars(axes[0, 1], bars, decimals=0)
    axes[0, 1].set_title("Latency")
    axes[0, 1].set_ylabel("ms")
    axes[0, 1].set_xticks(x, names, rotation=15, ha="right")
    axes[0, 1].legend()

    hit_width = 0.25
    bars = axes[1, 0].bar([i - hit_width for i in x], route_hit_pct, width=hit_width, label="route")
    annotate_bars(axes[1, 0], bars, suffix="%", decimals=1)
    bars = axes[1, 0].bar(x, owner_hit_pct, width=hit_width, label="owner")
    annotate_bars(axes[1, 0], bars, suffix="%", decimals=1)
    bars = axes[1, 0].bar([i + hit_width for i in x], local_hit_pct, width=hit_width, label="local")
    annotate_bars(axes[1, 0], bars, suffix="%", decimals=1)
    axes[1, 0].set_title("Prefix Hit Rate")
    axes[1, 0].set_ylabel("%")
    axes[1, 0].set_xticks(x, names, rotation=15, ha="right")
    axes[1, 0].legend()

    bars = axes[1, 1].bar([i - width / 2 for i in x], gpu_util, width=width, label="GPU util")
    annotate_bars(axes[1, 1], bars, suffix="%", decimals=1)
    bars = axes[1, 1].bar([i + width / 2 for i in x], gpu_mem_util, width=width, label="GPU mem util")
    annotate_bars(axes[1, 1], bars, suffix="%", decimals=1)
    axes[1, 1].set_title("GPU Utilization")
    axes[1, 1].set_ylabel("%")
    axes[1, 1].set_xticks(x, names, rotation=15, ha="right")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {output}")


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
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--model-name-or-path", type=str, default=MODEL_CONFIG["model_name_or_path"])
    parser.add_argument("--nvlink-pairs", type=str, default="0,1")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--routing-max-cached-blocks", type=int, default=1024)
    parser.add_argument("--eviction-max-cached-blocks", type=int, default=256)
    parser.add_argument("--goodput-e2e-sla-ms", type=float, default=2000.0)
    parser.add_argument("--skip-pool", action="store_true")
    parser.add_argument("--output-figure", type=str, default="")
    parser.add_argument("--submit-window", type=int, default=8)
    parser.add_argument("--disable-background-copy", action="store_true")
    parser.add_argument("--background-copy-max-blocks", type=int, default=MODEL_CONFIG["background_copy_max_blocks"])
    parser.add_argument("--background-copy-cooldown-s", type=float, default=MODEL_CONFIG["background_copy_cooldown_s"])
    parser.add_argument("--background-copy-hot-threshold", type=int, default=MODEL_CONFIG["background_copy_hot_threshold"])
    return parser.parse_args()


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
    visible_gpus = torch.cuda.device_count()
    if args.world_size > visible_gpus:
        raise SystemExit(
            f"--world-size {args.world_size} exceeds visible CUDA devices {visible_gpus}. "
            "Check CUDA_VISIBLE_DEVICES."
        )

    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = build_prompts(tokenizer, args.num_prompts, args.prompt_repeat)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_length=MODEL_CONFIG["max_model_length"],
    )
    goodput_e2e_sla_s = args.goodput_e2e_sla_ms / 1000.0

    # single-gpu baseline：单卡独立执行，不启用全局池
    baseline_config = make_config(1, False, None)
    baseline_config["model_name_or_path"] = model_name
    baseline = run_engine_scenario(
        "single-gpu",
        baseline_config,
        prompts,
        sampling_params,
        tokenizer,
        goodput_e2e_sla_s=goodput_e2e_sla_s,
        submit_window=args.submit_window,
    )
    baseline_hit_rate = measure_single_gpu_prefix_hit_rate(
        tokenizer,
        prompts,
        block_size=baseline_config["block_size"],
        max_cached_blocks=baseline_config["max_cached_blocks"],
    )
    baseline.prefix_hit_rate = baseline_hit_rate

    # multi-gpu baseline：不共享 KV、不走控制面路由，但请求通过 round-robin 分发到多张卡
    multi_gpu_config = make_config(args.world_size, False, None)
    multi_gpu_config["model_name_or_path"] = model_name
    independent_result = run_engine_scenario(
        "multi-gpu",
        multi_gpu_config,
        prompts,
        sampling_params,
        tokenizer,
        route_mode="round_robin",
        goodput_e2e_sla_s=goodput_e2e_sla_s,
        submit_window=args.submit_window,
    )

    # multi-gpu-kv-routing：走控制面路由，用来测 prefix 命中带来的收益
    routing_config = make_config(args.world_size, True, parse_pairs(args.nvlink_pairs) if args.nvlink_pairs else None)
    routing_config["model_name_or_path"] = model_name
    routing_config["max_cached_blocks"] = args.routing_max_cached_blocks
    routing_config["enable_foreground_rebalance"] = False
    routing_config["enable_background_copy"] = False
    kv_routing = run_engine_scenario(
        "multi-gpu-kv-routing",
        routing_config,
        prompts,
        sampling_params,
        tokenizer,
        route_mode="control_plane",
        goodput_e2e_sla_s=goodput_e2e_sla_s,
        submit_window=args.submit_window,
    )

    # multi-gpu-kv-transfer：用 round-robin 分发，尽量隔离出 transfer / rebalance 的开销
    eviction_config = make_config(args.world_size, True, parse_pairs(args.nvlink_pairs) if args.nvlink_pairs else None)
    eviction_config["model_name_or_path"] = model_name
    eviction_config["max_cached_blocks"] = args.eviction_max_cached_blocks
    apply_background_copy_args(eviction_config, args)
    kv_eviction = run_engine_scenario(
        "multi-gpu-kv-transfer",
        eviction_config,
        prompts,
        sampling_params,
        tokenizer,
        route_mode="round_robin",
        goodput_e2e_sla_s=goodput_e2e_sla_s,
        submit_window=args.submit_window,
    )

    pool_result = None
    if not args.skip_pool:
        if visible_gpus < args.world_size:
            print(f"pool scenario skipped: need {args.world_size} CUDA devices")
        else:
            # multi-gpu-lmpool：真实全局池化路径，控制面路由 + 数据面执行一起跑。
            pool_pairs = parse_pairs(args.nvlink_pairs) if args.nvlink_pairs else None
            pool_config = make_config(args.world_size, True, pool_pairs)
            pool_config["model_name_or_path"] = model_name
            apply_background_copy_args(pool_config, args)
            pool_result = run_engine_scenario(
                "multi-gpu-lmpool",
                pool_config,
                prompts,
                sampling_params,
                tokenizer,
                goodput_e2e_sla_s=goodput_e2e_sla_s,
                submit_window=args.submit_window,
            )

    all_results = [
        baseline,
        independent_result,
        kv_routing,
        kv_eviction,
        pool_result,
    ]
    print_summary_table(all_results)
    if args.output_figure:
        save_summary_figure(all_results, args.output_figure)
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
