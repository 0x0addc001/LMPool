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
}


SUFFIXES = [
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
    name: str
    total_requests: int
    total_tokens: int
    elapsed_s: float
    throughput_tok_s: float
    goodput_tok_s: float
    mean_ttft_s: float
    p50_ttft_s: float
    p95_ttft_s: float
    mean_ttpt_s: float
    p50_ttpt_s: float
    p95_ttpt_s: float
    mean_e2e_s: float
    p50_e2e_s: float
    p95_e2e_s: float
    prefix_hit_rate: float
    gpu_util_mean: float | None
    gpu_util_p95: float | None
    gpu_mem_util_mean: float | None
    gpu_mem_util_p95: float | None


def build_shared_prefix(prompt_repeat: int) -> str:
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
    seqs = [
        Sequence(tokenizer.encode(prompt), block_size=block_size)
        for prompt in prompts
    ]
    return seqs


def _percentile(values: list[float], p: float) -> float:
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

    for token_ids in prompt_token_ids:
        seq = Sequence(token_ids=token_ids, block_size=config["block_size"], sampling_params=sampling_params)
        scheduler.add_sequence(seq)
        submitted_at[seq.seq_id] = time.perf_counter()
        seq_count += 1

    start_wall = time.perf_counter()
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
            "p95_ttft_s": statistics.quantiles(ttfts, n=20)[18] if len(ttfts) >= 20 else (max(ttfts) if ttfts else 0.0),
            "ttfts": ttfts,
            "prefix_hit_rate": prefix_hits / max(seq_count, 1),
            "mean_e2e_s": _mean(e2es),
            "p50_e2e_s": _median(e2es),
            "p95_e2e_s": statistics.quantiles(e2es, n=20)[18] if len(e2es) >= 20 else (max(e2es) if e2es else 0.0),
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
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        return None

    ctx = mp.get_context("spawn")
    prompt_token_ids = [tokenizer.encode(prompt) for prompt in prompts]
    shards: list[list[list[int]]] = [[] for _ in range(gpu_count)]
    for idx, token_ids in enumerate(prompt_token_ids):
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
            p95_ttft_s=statistics.quantiles(ttfts, n=20)[18] if len(ttfts) >= 20 else (max(ttfts) if ttfts else 0.0),
            mean_ttpt_s=_mean(ttfts),
            p50_ttpt_s=_median(ttfts),
            p95_ttpt_s=statistics.quantiles(ttfts, n=20)[18] if len(ttfts) >= 20 else (max(ttfts) if ttfts else 0.0),
            mean_e2e_s=_mean(e2es),
            p50_e2e_s=_median(e2es),
            p95_e2e_s=statistics.quantiles(e2es, n=20)[18] if len(e2es) >= 20 else (max(e2es) if e2es else 0.0),
            prefix_hit_rate=prefix_hit_rate,
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
) -> ScenarioResult:
    engine = LLMEngine(config)
    submit_times: dict[int, float] = {}
    ttfts: list[float] = []
    e2es: list[float] = []
    total_tokens = 0
    goodput_tokens = 0
    route_hits = 0
    start_wall = time.perf_counter()
    sampler = GpuMetricSampler(interval_s=0.5)

    try:
        sampler.start()
        for prompt in prompts:
            seq = Sequence(
                token_ids=tokenizer.encode(prompt),
                block_size=config["block_size"],
                sampling_params=sampling_params,
            )
            start = time.perf_counter()
            target_rank = 0
            if route_mode == "control_plane" and engine.control_plane_client is not None:
                routed = engine.control_plane_client.route_sequence(seq, return_meta=True)
                target_rank = routed["target_rank"]
                if routed.get("route_info", {}).get("prefix_hit"):
                    route_hits += 1
            elif route_mode == "round_robin":
                target_rank = len(submit_times) % config["world_size"]
            seq.remote_gpu_id = target_rank
            engine.send_queues[target_rank].put({"type": "sequence", "seq": seq})
            submit_times[seq.seq_id] = start

        finished_count = 0
        completion_times: dict[int, float] = {}
        completion_token_counts: dict[int, int] = {}
        while finished_count < len(prompts):
            finished, _, _ = engine.step()
            now = time.perf_counter()
            for seq_id, tokens in finished:
                finished_count += 1
                total_tokens += len(tokens)
                ttfts.append(now - submit_times[seq_id])
                e2es.append(now - submit_times[seq_id])
                completion_times[seq_id] = now
                completion_token_counts[seq_id] = len(tokens)
        elapsed = time.perf_counter() - start_wall
    finally:
        sampler.stop()
        engine.exit()

    if engine.control_plane_client is None:
        route_hits = 0

    gpu_util_mean, gpu_util_p95, gpu_mem_util_mean, gpu_mem_util_p95 = sampler.summarize()
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
        p95_ttft_s=statistics.quantiles(ttfts, n=20)[18] if len(ttfts) >= 20 else (max(ttfts) if ttfts else 0.0),
        mean_ttpt_s=_mean(ttfts),
        p50_ttpt_s=_median(ttfts),
        p95_ttpt_s=statistics.quantiles(ttfts, n=20)[18] if len(ttfts) >= 20 else (max(ttfts) if ttfts else 0.0),
        mean_e2e_s=_mean(e2es),
        p50_e2e_s=_median(e2es),
        p95_e2e_s=statistics.quantiles(e2es, n=20)[18] if len(e2es) >= 20 else (max(e2es) if e2es else 0.0),
        prefix_hit_rate=route_hits / max(len(prompts), 1),
        gpu_util_mean=gpu_util_mean,
        gpu_util_p95=gpu_util_p95,
        gpu_mem_util_mean=gpu_mem_util_mean,
        gpu_mem_util_p95=gpu_mem_util_p95,
    )


def make_config(world_size: int, enable_global_pool: bool, nvlink_pairs: list[tuple[int, int]] | None) -> dict:
    config = dict(MODEL_CONFIG)
    config["world_size"] = world_size
    config["enable_global_pool"] = enable_global_pool
    if nvlink_pairs is not None:
        config["nvlink_topo"] = {"pairs": nvlink_pairs}
    config["use_control_plane_process"] = enable_global_pool
    return config


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_report(title: str, baseline: ScenarioResult, compare: ScenarioResult | None):
    print("\nBenchmark Summary")
    print("=" * 80)
    print(title)
    print(
        f"{'scenario':<20} {'tput(tok/s)':>14} {'goodput':>12} {'ttft(ms)':>12} {'ttpt(ms)':>12} "
        f"{'e2e(ms)':>12} {'p95(e2e)':>12} {'gpu util':>10} {'mem util':>10} {'prefix hit':>14}"
    )
    print(
        f"{baseline.name:<20} "
        f"{baseline.throughput_tok_s:>14.2f} "
        f"{baseline.goodput_tok_s:>12.2f} "
        f"{baseline.mean_ttft_s * 1000:>12.2f} "
        f"{baseline.mean_ttpt_s * 1000:>12.2f} "
        f"{baseline.mean_e2e_s * 1000:>12.2f} "
        f"{baseline.p95_e2e_s * 1000:>12.2f} "
        f"{(baseline.gpu_util_mean if baseline.gpu_util_mean is not None else float('nan')):>10.2f} "
        f"{(baseline.gpu_mem_util_mean if baseline.gpu_mem_util_mean is not None else float('nan')):>10.2f} "
        f"{fmt_pct(baseline.prefix_hit_rate):>14}"
    )
    if compare is not None:
        print(
            f"{compare.name:<20} "
            f"{compare.throughput_tok_s:>14.2f} "
            f"{compare.goodput_tok_s:>12.2f} "
            f"{compare.mean_ttft_s * 1000:>12.2f} "
            f"{compare.mean_ttpt_s * 1000:>12.2f} "
            f"{compare.mean_e2e_s * 1000:>12.2f} "
            f"{compare.p95_e2e_s * 1000:>12.2f} "
            f"{(compare.gpu_util_mean if compare.gpu_util_mean is not None else float('nan')):>10.2f} "
            f"{(compare.gpu_mem_util_mean if compare.gpu_mem_util_mean is not None else float('nan')):>10.2f} "
            f"{fmt_pct(compare.prefix_hit_rate):>14}"
        )
        print("-" * 80)
        print(f"throughput uplift: {(compare.throughput_tok_s / max(baseline.throughput_tok_s, 1e-9) - 1.0) * 100:.2f}%")
        print(f"goodput uplift: {(compare.goodput_tok_s / max(baseline.goodput_tok_s, 1e-9) - 1.0) * 100:.2f}%")
        print(f"TTFT reduction: {(1.0 - compare.mean_ttft_s / max(baseline.mean_ttft_s, 1e-9)) * 100:.2f}%")
        print(f"TTPT reduction: {(1.0 - compare.mean_ttpt_s / max(baseline.mean_ttpt_s, 1e-9)) * 100:.2f}%")
        print(f"E2E reduction: {(1.0 - compare.mean_e2e_s / max(baseline.mean_e2e_s, 1e-9)) * 100:.2f}%")
        print(f"prefix-hit lift: {(compare.prefix_hit_rate - baseline.prefix_hit_rate) * 100:.2f} pct points")


def parse_args():
    parser = argparse.ArgumentParser(description="Shared-prefix high-concurrency benchmark")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--prompt-repeat", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--model-name-or-path", type=str, default=MODEL_CONFIG["model_name_or_path"])
    parser.add_argument("--nvlink-pairs", type=str, default="0,1")
    parser.add_argument("--routing-max-cached-blocks", type=int, default=1024)
    parser.add_argument("--eviction-max-cached-blocks", type=int, default=256)
    parser.add_argument("--goodput-e2e-sla-ms", type=float, default=2000.0)
    parser.add_argument("--skip-pool", action="store_true")
    return parser.parse_args()


def parse_pairs(raw: str) -> list[tuple[int, int]]:
    if not raw:
        return []
    pairs = []
    for item in raw.split(";"):
        a, b = item.split(",")
        pairs.append((int(a), int(b)))
    return pairs


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = build_prompts(tokenizer, args.num_prompts, args.prompt_repeat)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_length=MODEL_CONFIG["max_model_length"],
    )
    goodput_e2e_sla_s = args.goodput_e2e_sla_ms / 1000.0

    baseline_config = make_config(1, False, None)
    baseline_config["model_name_or_path"] = model_name
    baseline = run_engine_scenario(
        "single-gpu",
        baseline_config,
        prompts,
        sampling_params,
        tokenizer,
        goodput_e2e_sla_s=goodput_e2e_sla_s,
    )
    baseline_hit_rate = measure_single_gpu_prefix_hit_rate(
        tokenizer,
        prompts,
        block_size=baseline_config["block_size"],
        max_cached_blocks=baseline_config["max_cached_blocks"],
    )
    baseline.prefix_hit_rate = baseline_hit_rate

    independent_result = run_independent_multi_gpu_benchmark(
        "multi-gpu",
        baseline_config,
        prompts,
        sampling_params,
        tokenizer,
        goodput_e2e_sla_s,
    )

    routing_config = make_config(2, True, parse_pairs(args.nvlink_pairs) if args.nvlink_pairs else None)
    routing_config["model_name_or_path"] = model_name
    routing_config["max_cached_blocks"] = args.routing_max_cached_blocks
    kv_routing = run_engine_scenario(
        "multi-gpu-kv-routing",
        routing_config,
        prompts,
        sampling_params,
        tokenizer,
        route_mode="control_plane",
        goodput_e2e_sla_s=goodput_e2e_sla_s,
    )

    eviction_config = make_config(2, True, parse_pairs(args.nvlink_pairs) if args.nvlink_pairs else None)
    eviction_config["model_name_or_path"] = model_name
    eviction_config["max_cached_blocks"] = args.eviction_max_cached_blocks
    kv_eviction = run_engine_scenario(
        "multi-gpu-kv-swapping",
        eviction_config,
        prompts,
        sampling_params,
        tokenizer,
        route_mode="round_robin",
        goodput_e2e_sla_s=goodput_e2e_sla_s,
    )

    pool_result = None
    if not args.skip_pool:
        if torch.cuda.device_count() < 2:
            print("pool scenario skipped: need at least 2 CUDA devices")
        else:
            pool_pairs = parse_pairs(args.nvlink_pairs) if args.nvlink_pairs else None
            pool_config = make_config(2, True, pool_pairs)
            pool_config["model_name_or_path"] = model_name
            pool_result = run_engine_scenario(
                "multi-gpu-lmpool",
                pool_config,
                prompts,
                sampling_params,
                tokenizer,
                goodput_e2e_sla_s=goodput_e2e_sla_s,
            )

    print_report("single-gpu vs multi-gpu", baseline, independent_result)
    print_report("single-gpu vs multi-gpu-kv-routing", baseline, kv_routing)
    print_report("single-gpu vs multi-gpu-kv-swapping", baseline, kv_eviction)
    if pool_result is not None:
        print()
        print_report("single-gpu vs multi-gpu-lmpool", baseline, pool_result)
    if args.output_json:
        payload = {
            "single-gpu": asdict(baseline),
            "multi-gpu": asdict(independent_result) if independent_result is not None else None,
            "multi-gpu-kv-routing": asdict(kv_routing) if kv_routing is not None else None,
            "multi-gpu-kv-swapping": asdict(kv_eviction) if kv_eviction is not None else None,
            "multi-gpu-lmpool": asdict(pool_result) if pool_result is not None else None,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
