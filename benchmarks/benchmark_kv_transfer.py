"""
KV transfer microbenchmark.

Purpose:
  Validate the data-plane transfer primitive directly, without mixing in
  routing, prefill, decode, or control-plane queueing effects.

Example:
  CUDA_VISIBLE_DEVICES=0,2 UV_CACHE_DIR=/tmp/uvcache uv run python benchmarks/benchmark_kv_transfer.py \
    --model-name-or-path /path/to/Qwen3-0.6B \
    --block-size 256 \
    --block-counts 1,2,4,8 \
    --iterations 100 \
    --warmup 20 \
    --output-json benchmarks/results/kv_transfer.json \
    --output-figure benchmarks/results/kv_transfer.png

Notes:
  - GPU ids are logical ids after CUDA_VISIBLE_DEVICES remapping.
  - The benchmark uses the existing kv_transfer.swap_in legacy API, whose
    semantic role is transfer-in / transfer-out.
  - Reported bandwidth counts both K and V tensors across all layers.
  - Use --block-counts 1,2,4,8 for a paper-ready payload-size sweep.
"""

import argparse
import json
import os
import socket
import sys
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lmpool.engine.kv_transfer import prewarm_p2p_pairs, swap_in

try:
    from .benchmark_utils import (
        build_run_metadata,
        normalize_dtype_name,
        resolve_model_runtime_config,
    )
except ImportError:
    from benchmark_utils import (
        build_run_metadata,
        normalize_dtype_name,
        resolve_model_runtime_config,
    )


_TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fill_source_blocks(kv_cache: torch.Tensor, blocks: list[int]) -> None:
    for block_id in blocks:
        kv_cache[0, :, block_id].fill_(float(block_id + 1))
        kv_cache[1, :, block_id].fill_(float(block_id + 1001))


def _check_target_blocks(kv_cache: torch.Tensor, src_blocks: list[int], dst_blocks: list[int]) -> bool:
    for src_block, dst_block in zip(src_blocks, dst_blocks):
        expected_k = float(src_block + 1)
        expected_v = float(src_block + 1001)
        if not torch.all(kv_cache[0, :, dst_block] == expected_k).item():
            return False
        if not torch.all(kv_cache[1, :, dst_block] == expected_v).item():
            return False
    return True


def _worker(rank: int, args, port: int, result_queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=2,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )

    src_blocks = list(range(args.num_transfer_blocks))
    dst_blocks = list(range(args.num_transfer_blocks, args.num_transfer_blocks * 2))
    num_blocks = args.num_transfer_blocks * 2
    transfer_dtype = _TORCH_DTYPES[args.resolved_dtype]
    kv_cache = torch.zeros(
        2,
        args.num_layers,
        num_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
        dtype=transfer_dtype,
        device=f"cuda:{rank}",
    )
    if rank == 0:
        _fill_source_blocks(kv_cache, src_blocks)

    prewarm_p2p_pairs(
        [(0, 1)],
        num_layers=args.num_layers,
        block_size=args.block_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_blocks=args.num_transfer_blocks,
        dtype=transfer_dtype,
    )
    dist.barrier()
    total_iters = args.warmup + args.iterations
    measured_ms: list[float] = []

    for iteration in range(total_iters):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if rank == 0:
            swap_in(
                remote_gpu=0,
                remote_blocks=src_blocks,
                local_gpu=1,
                kv_cache=kv_cache,
                num_layers=args.num_layers,
                block_size=args.block_size,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                negotiate_blocks=False,
            )
        else:
            swap_in(
                remote_gpu=0,
                remote_blocks=src_blocks,
                local_gpu=1,
                kv_cache=kv_cache,
                num_layers=args.num_layers,
                block_size=args.block_size,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                local_target_blocks=dst_blocks,
                negotiate_blocks=False,
            )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if iteration >= args.warmup:
            measured_ms.append(elapsed_ms)

    ok = True
    if rank == 1:
        ok = _check_target_blocks(kv_cache, src_blocks, dst_blocks)

    bytes_per_iter = (
        args.num_transfer_blocks
        * args.num_layers
        * args.block_size
        * args.num_kv_heads
        * args.head_dim
        * 2  # K and V
        * kv_cache.element_size()
    )
    result_queue.put({
        "rank": rank,
        "ok": bool(ok),
        "mean_ms": statistics.mean(measured_ms),
        "p95_ms": max(measured_ms) if len(measured_ms) < 2 else statistics.quantiles(measured_ms, n=20)[18],
        "bytes_per_iter": bytes_per_iter,
    })
    dist.destroy_process_group()


def parse_block_counts(raw: str, fallback: int) -> list[int]:
    """Parse a stable, deduplicated transfer-block sweep."""
    values = [fallback] if not raw.strip() else [int(item) for item in raw.split(",")]
    if any(value < 1 for value in values):
        raise ValueError("transfer block counts must all be >= 1")
    return list(dict.fromkeys(values))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="KV transfer microbenchmark")
    parser.add_argument(
        "--model-name-or-path",
        default="",
        help="Optional local/Hugging Face model whose KV geometry and dtype are measured.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Transfer payload dtype; auto uses model config or float16 without a model.",
    )
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--num-transfer-blocks", type=int, default=2)
    parser.add_argument(
        "--block-counts",
        default="",
        help="Comma-separated payload sweep; overrides --num-transfer-blocks.",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-figure", default="")
    return parser.parse_args(argv)


def resolve_transfer_contract(args) -> tuple[argparse.Namespace, dict | None, dict]:
    """Resolve one model-consistent KV payload contract before spawning workers."""
    model_metadata = None
    resolved_config: dict = {}
    model_geometry = {}
    if args.model_name_or_path:
        resolved_config, model_metadata = resolve_model_runtime_config(
            args.model_name_or_path,
            {"max_model_length": 2048},
            dtype_override=args.dtype,
        )
        model_geometry = {
            "num_layers": resolved_config["num_layers"],
            "num_kv_heads": resolved_config["num_kv_heads"],
            "head_dim": resolved_config["head_dim"],
        }

    fallbacks = {
        "num_layers": 28,
        "num_kv_heads": 8,
        "head_dim": 128,
    }
    for name, fallback in fallbacks.items():
        explicit = getattr(args, name)
        model_value = model_geometry.get(name)
        if explicit is not None and model_value is not None and explicit != model_value:
            option = name.replace("_", "-")
            raise ValueError(
                f"--{option}={explicit} conflicts with model config value {model_value}"
            )
        setattr(args, name, explicit if explicit is not None else model_value or fallback)

    args.resolved_dtype = (
        resolved_config["torch_dtype"]
        if resolved_config
        else normalize_dtype_name(args.dtype, auto_fallback="float16")
    )
    resolved_config = {
        **resolved_config,
        "block_size": args.block_size,
        "num_layers": args.num_layers,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "torch_dtype": args.resolved_dtype,
    }
    return args, model_metadata, resolved_config


def run_transfer_case(args, num_transfer_blocks: int) -> dict:
    case_args = argparse.Namespace(**vars(args))
    case_args.num_transfer_blocks = int(num_transfer_blocks)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    port = _find_free_port()
    procs = [
        ctx.Process(target=_worker, args=(rank, case_args, port, result_queue))
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()

    results = []
    deadline = time.time() + 300
    try:
        while time.time() < deadline and len(results) < 2:
            try:
                results.append(result_queue.get(timeout=1))
            except Exception:
                pass
    finally:
        for proc in procs:
            proc.join(timeout=20)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=20)
            proc.close()
        result_queue.close()
        result_queue.join_thread()

    if len(results) != 2 or not all(item["ok"] for item in results):
        raise RuntimeError(
            f"transfer validation failed for {num_transfer_blocks} blocks: {results}"
        )

    rank1 = next(item for item in results if item["rank"] == 1)
    mean_ms = rank1["mean_ms"]
    p95_ms = rank1["p95_ms"]
    bytes_per_iter = rank1["bytes_per_iter"]
    gib = bytes_per_iter / (1024 ** 3)
    bandwidth = gib / max(mean_ms / 1000.0, 1e-9)
    return {
        "num_transfer_blocks": int(num_transfer_blocks),
        "bytes_per_iteration": int(bytes_per_iter),
        "gib_per_iteration": float(gib),
        "mean_latency_ms": float(mean_ms),
        "p95_latency_ms": float(p95_ms),
        "effective_bandwidth_gib_s": float(bandwidth),
        "data_validation": "passed",
    }


def save_results_json(
    results: list[dict],
    output_path: str,
    *,
    metadata: dict,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"saved json: {output}")


def save_results_figure(
    results: list[dict],
    output_path: str,
    *,
    title: str = "KV Transfer Microbenchmark",
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(item["num_transfer_blocks"]) for item in results]
    x = list(range(len(results)))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(title, fontsize=13)

    mean_bars = axes[0].bar(
        [value - width / 2 for value in x],
        [item["mean_latency_ms"] for item in results],
        width,
        label="Mean",
        color="#4477AA",
    )
    p95_bars = axes[0].bar(
        [value + width / 2 for value in x],
        [item["p95_latency_ms"] for item in results],
        width,
        label="P95",
        color="#EE6677",
    )
    axes[0].bar_label(mean_bars, fmt="%.2f", fontsize=8, padding=2)
    axes[0].bar_label(p95_bars, fmt="%.2f", fontsize=8, padding=2)
    axes[0].set_title("KV Transfer Latency")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("Transferred KV blocks")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)

    bandwidth_bars = axes[1].bar(
        x,
        [item["effective_bandwidth_gib_s"] for item in results],
        color="#228833",
    )
    axes[1].bar_label(bandwidth_bars, fmt="%.2f", fontsize=8, padding=2)
    axes[1].set_title("Effective KV Transfer Bandwidth")
    axes[1].set_ylabel("Bandwidth (GiB/s)")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Transferred KV blocks")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {output}")


def main() -> None:
    args = parse_args()
    try:
        args, model_metadata, resolved_config = resolve_transfer_contract(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot resolve transfer benchmark config: {exc}") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("requires at least 2 visible CUDA devices")
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    for name in ("num_layers", "block_size", "num_kv_heads", "head_dim"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    try:
        block_counts = parse_block_counts(
            args.block_counts,
            args.num_transfer_blocks,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    results = [run_transfer_case(args, block_count) for block_count in block_counts]

    metadata = build_run_metadata(
        "benchmark_kv_transfer",
        args,
        model=model_metadata,
        resolved_config=resolved_config,
    )

    print("KV Transfer Benchmark")
    print("=" * 80)
    print(
        f"{'blocks':>8} {'payload(GiB)':>14} {'mean(ms)':>12} "
        f"{'p95(ms)':>12} {'bandwidth(GiB/s)':>18} {'validation':>12}"
    )
    for result in results:
        print(
            f"{result['num_transfer_blocks']:>8d} "
            f"{result['gib_per_iteration']:>14.3f} "
            f"{result['mean_latency_ms']:>12.3f} "
            f"{result['p95_latency_ms']:>12.3f} "
            f"{result['effective_bandwidth_gib_s']:>18.2f} "
            f"{result['data_validation']:>12}"
        )
    if args.output_figure:
        model_label = (
            model_metadata["label"]
            if model_metadata is not None
            else f"custom-{args.resolved_dtype}"
        )
        save_results_figure(
            results,
            args.output_figure,
            title=f"KV Transfer Microbenchmark: {model_label}",
        )
    if args.output_json:
        save_results_json(results, args.output_json, metadata=metadata)


if __name__ == "__main__":
    main()
