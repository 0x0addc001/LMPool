import pytest
import torch
import torch.distributed as dist

from lmpool.engine.block_manager import BlockManager
from lmpool.engine.global_block_manager import BlockLocation, GlobalBlockManager
from lmpool.engine.sequence import Sequence


class DummyGBM:
    def __init__(self):
        self.commits = []
        self.eviction_requests = []

    def _commit_alloc(self, rank, block_ids, hashes):
        self.commits.append((rank, list(block_ids), list(hashes)))

    def lookup_prefix(self, prefix_hash, requester_rank=None):
        return [
            BlockLocation(1, 7, prefix_hash, 2.0),
        ]

    def _get_nvlink_partner(self, rank):
        return 1 if rank == 0 else 0 if rank == 1 else None

    def select_eviction_candidates(self, rank, shortage):
        self.eviction_requests.append((rank, shortage))
        return [(block_id, 1) for block_id in range(shortage)]

    def record_block_transfer(self, **kwargs):
        self.commits.append(("transfer", kwargs))


def test_compute_hash_accepts_python_lists_and_tensors():
    bm = BlockManager(num_blocks=4, block_size=2)
    h1 = bm.compute_hash([1, 2], -1)
    h2 = bm.compute_hash(torch.tensor([1, 2]), -1)
    assert h1 == h2


def test_allocate_deallocate_and_global_commit(monkeypatch):
    gbm = DummyGBM()
    bm = BlockManager(num_blocks=4, block_size=2, gbm=gbm)
    seq = Sequence([1, 2, 3, 4], block_size=2)

    bm.allocate(seq)

    assert seq.block_table == [0, 1]
    assert seq.num_cached_tokens == 0
    assert len(gbm.commits) == 2
    assert bm.get_local_block_hashes()

    bm.deallocate(seq)
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0


def test_remote_prefix_and_swap_helpers(monkeypatch):
    gbm = DummyGBM()
    bm = BlockManager(num_blocks=6, block_size=2, gbm=gbm)
    seq = Sequence([10, 20, 30, 40], block_size=2)

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    hit, gpu = bm.try_allocate_remote(seq)
    assert hit is True
    assert gpu == 1
    assert seq.is_remote_prefix is True
    assert seq.remote_gpu_id == 1
    assert seq.pending_swap_in == [7]

    assert bm.reserve_free_blocks(1) == [0]
    bm.register_swap_in_blocks([0], [123])
    assert bm.get_local_block_hashes()[0] == 123
    bm.release_blocks([0])
    assert 0 in bm.free_block_ids


def test_allocate_with_swap_uses_eviction_candidates(monkeypatch):
    gbm = DummyGBM()
    bm = BlockManager(num_blocks=2, block_size=2, gbm=gbm)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    # 构造两个“已占用但当前无人引用”的冷块，模拟已经被 swap_out 的候选块。
    for block_id in range(2):
        block = bm._allocate_block(block_id)
        block.ref_count = 0
        block.update(100 + block_id, [block_id, block_id + 1])

    bm.free_block_ids.clear()
    seq = Sequence([5, 6, 7, 8], block_size=2)
    assert bm.allocate_with_swap(seq) is True
    assert gbm.eviction_requests == [(0, 2)]


def test_shared_prefix_ref_count_keeps_block_out_of_free_list():
    bm = BlockManager(num_blocks=4, block_size=2)
    seq1 = Sequence([1, 2], block_size=2)
    seq2 = Sequence([1, 2], block_size=2)

    bm.allocate(seq1)
    bm.allocate(seq2)

    shared_block_id = seq1.block_table[0]
    assert seq2.block_table[0] == shared_block_id
    assert bm.blocks[shared_block_id].ref_count == 2

    bm.deallocate(seq1)
    assert bm.blocks[shared_block_id].ref_count == 1
    assert shared_block_id not in bm.free_block_ids

    bm.deallocate(seq2)
    assert bm.blocks[shared_block_id].ref_count == 0
    assert shared_block_id in bm.free_block_ids


def test_transfer_in_block_can_be_reused_as_trusted_prefix_hit():
    bm = BlockManager(num_blocks=4, block_size=2)
    h = bm.compute_hash([1, 2], -1)

    reserved = bm.reserve_free_blocks(1)
    bm.register_swap_in_blocks(reserved, [h])

    seq = Sequence([1, 2, 3, 4], block_size=2)
    bm.allocate(seq)

    assert seq.block_table[0] == reserved[0]
    assert seq.num_cached_tokens == 2


def test_non_contiguous_cache_hit_does_not_count_as_prefix_hit():
    bm = BlockManager(num_blocks=4, block_size=2)
    h0 = bm.compute_hash([1, 2], -1)
    h1 = bm.compute_hash([3, 4], h0)

    reserved = bm.reserve_free_blocks(1)
    bm.register_swap_in_blocks(reserved, [h1])

    seq = Sequence([1, 2, 3, 4], block_size=2)
    bm.allocate(seq)

    assert reserved[0] in seq.block_table
    assert seq.num_cached_tokens == 0


def test_can_allocate_counts_only_blocks_that_need_new_storage():
    bm = BlockManager(num_blocks=2, block_size=2)
    cached = Sequence([1, 2], block_size=2)
    bm.allocate(cached)

    seq = Sequence([1, 2, 3, 4], block_size=2)

    assert len(bm.free_block_ids) == 1
    assert seq.num_blocks == 2
    assert bm.num_required_new_blocks(seq) == 1
    assert bm.can_allocate(seq) is True

    bm.allocate(seq)
    assert seq.num_cached_tokens == 2
