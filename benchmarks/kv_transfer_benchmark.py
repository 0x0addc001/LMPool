"""
KV transfer microbenchmark.

Purpose:
  Validate the data-plane transfer primitive directly, without mixing in
  routing, prefill, decode, or control-plane queueing effects.

Example:
  CUDA_VISIBLE_DEVICES=0,2 UV_CACHE_DIR=/tmp/uvcache uv run python benchmarks/kv_transfer_benchmark.py \
    --num-layers 28 \
    --block-size 256 \
    --num-kv-heads 8 \
    --head-dim 128 \
    --num-transfer-blocks 2 \
    --iterations 5 \
    --warmup 1

Notes:
  - GPU ids are logical ids after CUDA_VISIBLE_DEVICES remapping.
  - The benchmark uses the existing kv_transfer.swap_in legacy API, whose
    semantic role is transfer-in / transfer-out.
  - Reported bandwidth counts both K and V tensors across all layers.
"""

import argparse
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
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=2,
        rank=rank,
    )
    torch.cuda.set_device(rank)

    src_blocks = list(range(args.num_transfer_blocks))
    dst_blocks = list(range(args.num_transfer_blocks, args.num_transfer_blocks * 2))
    num_blocks = args.num_transfer_blocks * 2
    kv_cache = torch.zeros(
        2,
        args.num_layers,
        num_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
        dtype=torch.float16,
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
        * torch.finfo(torch.float16).bits
        // 8
    )
    result_queue.put({
        "rank": rank,
        "ok": bool(ok),
        "mean_ms": statistics.mean(measured_ms),
        "p95_ms": max(measured_ms) if len(measured_ms) < 2 else statistics.quantiles(measured_ms, n=20)[18],
        "bytes_per_iter": bytes_per_iter,
    })
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="KV transfer microbenchmark")
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-transfer-blocks", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("requires at least 2 visible CUDA devices")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    port = _find_free_port()
    procs = [
        ctx.Process(target=_worker, args=(rank, args, port, result_queue))
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
        raise SystemExit(f"transfer validation failed: {results}")

    rank1 = next(item for item in results if item["rank"] == 1)
    mean_ms = rank1["mean_ms"]
    p95_ms = rank1["p95_ms"]
    bytes_per_iter = rank1["bytes_per_iter"]
    gib = bytes_per_iter / (1024 ** 3)
    bandwidth = gib / max(mean_ms / 1000.0, 1e-9)

    print("KV Transfer Benchmark")
    print("=" * 80)
    print(f"blocks transferred: {args.num_transfer_blocks}")
    print(f"bytes per iteration: {bytes_per_iter} ({gib:.3f} GiB)")
    print(f"mean latency: {mean_ms:.3f} ms")
    print(f"p95 latency: {p95_ms:.3f} ms")
    print(f"effective bandwidth: {bandwidth:.2f} GiB/s")
    print("data validation: passed")


if __name__ == "__main__":
    main()
