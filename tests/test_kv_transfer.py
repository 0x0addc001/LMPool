import os
import socket
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lmpool.engine.kv_transfer import _allocate_empty_block, _compute_tag, swap_in, swap_out


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fill_block(kv_cache: torch.Tensor, block_id: int, base: float) -> None:
    kv_cache[0, block_id].fill_(base)
    kv_cache[1, block_id].fill_(base + 0.5)


def _assert_block(kv_cache: torch.Tensor, block_id: int, base: float) -> None:
    expected_k = torch.full_like(kv_cache[0, block_id], base)
    expected_v = torch.full_like(kv_cache[1, block_id], base + 0.5)
    assert torch.equal(kv_cache[0, block_id].cpu(), expected_k.cpu())
    assert torch.equal(kv_cache[1, block_id].cpu(), expected_v.cpu())


def test_kv_transfer_helpers_are_stable():
    kv_cache = torch.zeros(2, 4, 2, 1, 1)
    view = _allocate_empty_block(kv_cache, 2)
    assert view.shape == (2, 2, 1, 1)
    assert _compute_tag(3, 4, True) != _compute_tag(3, 4, False)
    assert _compute_tag(3, 4, True) != _compute_tag(4, 4, True)


def _swap_worker(rank: int, world_size: int, port: int, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)

    num_blocks = 4
    block_size = 2
    num_kv_heads = 1
    head_dim = 1
    kv_cache = torch.zeros(
        2,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=f"cuda:{rank}",
    )

    if rank == 0:
        _fill_block(kv_cache, 0, 1.0)
        _fill_block(kv_cache, 1, 2.0)

    dist.barrier()

    if rank == 0:
        swap_out(
            local_gpu=0,
            blocks_to_evict=[0, 1],
            target_gpu=1,
            kv_cache=kv_cache,
            num_layers=1,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        _assert_block(kv_cache, 0, 1.0)
        _assert_block(kv_cache, 1, 2.0)
    else:
        swap_in(
            remote_gpu=0,
            remote_blocks=[0, 1],
            local_gpu=1,
            kv_cache=kv_cache,
            num_layers=1,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            local_target_blocks=[2, 3],
        )

    dist.barrier()

    if rank == 1:
        _assert_block(kv_cache, 2, 1.0)
        _assert_block(kv_cache, 3, 2.0)
        result_queue.put({"rank": rank, "ok": True})
    elif rank == 0:
        result_queue.put({"rank": rank, "ok": True})

    dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("RUN_NCCL_INTEGRATION") != "1"
    or not torch.cuda.is_available()
    or torch.cuda.device_count() < 2,
    reason="requires RUN_NCCL_INTEGRATION=1 and at least 2 CUDA devices with NCCL support",
)
def test_nccl_swap_out_and_swap_in_round_trip():
    ctx = mp.get_context("spawn")
    port = _find_free_port()
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(target=_swap_worker, args=(rank, 2, port, result_queue))
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()

    try:
        deadline = time.time() + 60
        seen = []
        while time.time() < deadline and len(seen) < 2:
            try:
                seen.append(result_queue.get(timeout=1))
            except Exception:
                pass
        assert len(seen) == 2
        assert {item["rank"] for item in seen} == {0, 1}
        assert all(item["ok"] for item in seen)
    finally:
        for proc in procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
            proc.close()
        result_queue.close()
        result_queue.join_thread()
