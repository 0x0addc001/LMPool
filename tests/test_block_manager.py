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
    assert gbm.commits == []
    assert bm.get_local_block_hashes() == {}

    bm.mark_kv_ready([seq])
    assert len(gbm.commits) == 2
    assert bm.get_local_block_hashes()

    bm.deallocate(seq)
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0


def test_can_append_checks_capacity_after_token_crosses_block_boundary():
    bm = BlockManager(num_blocks=1, block_size=2)
    seq = Sequence([1, 2], block_size=2)
    bm.allocate(seq)

    seq.append_token(3)

    assert bm.can_append(seq) is False
    with pytest.raises(RuntimeError, match="new KV block is required"):
        bm.append(seq)


def test_append_allocates_new_block_after_boundary_token_is_added():
    bm = BlockManager(num_blocks=2, block_size=2)
    seq = Sequence([1, 2], block_size=2)
    bm.allocate(seq)

    seq.append_token(3)

    assert bm.can_append(seq) is True
    bm.append(seq)
    assert seq.block_table == [0, 1]


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
    bm.mark_kv_ready([seq1])
    bm.allocate(seq2)

    shared_block_id = seq1.block_table[0]
    assert seq2.block_table[0] == shared_block_id
    assert bm.blocks[shared_block_id].ref_count == 2

    bm.deallocate(seq1)
    assert bm.blocks[shared_block_id].ref_count == 1
    assert shared_block_id not in bm.free_block_ids

    bm.deallocate(seq2)
    assert bm.blocks[shared_block_id].ref_count == 0
    assert shared_block_id not in bm.free_block_ids
    assert shared_block_id in bm.get_evictable_block_hashes()

    seq3 = Sequence([1, 2], block_size=2)
    bm.allocate(seq3)
    assert seq3.block_table == [shared_block_id]
    assert seq3.num_cached_tokens == 2


def test_reclaim_cached_blocks_evicts_deepest_prefix_leaf_first():
    bm = BlockManager(num_blocks=4, block_size=2)
    seq = Sequence([1, 2, 3, 4, 5, 6], block_size=2)
    bm.allocate(seq)
    bm.mark_kv_ready([seq])
    block_ids = list(seq.block_table)
    bm.deallocate(seq)

    assert bm.reclaim_cached_blocks(1) == 1
    assert block_ids[0] in bm.used_block_ids
    assert block_ids[1] in bm.used_block_ids
    assert block_ids[2] not in bm.used_block_ids

    assert bm.reclaim_cached_blocks(1) == 1
    assert block_ids[0] in bm.used_block_ids
    assert block_ids[1] not in bm.used_block_ids


def test_reclaim_cached_blocks_preserves_shared_ancestor_until_branches_are_gone():
    bm = BlockManager(num_blocks=5, block_size=2)
    left = Sequence([1, 2, 3, 4], block_size=2)
    right = Sequence([1, 2, 5, 6], block_size=2)
    bm.allocate(left)
    bm.mark_kv_ready([left])
    ancestor = left.block_table[0]
    left_leaf = left.block_table[1]
    bm.deallocate(left)
    bm.allocate(right)
    bm.mark_kv_ready([right])
    right_leaf = right.block_table[1]
    bm.deallocate(right)

    assert bm.reclaim_cached_blocks(2) == 2
    assert ancestor in bm.used_block_ids
    assert left_leaf not in bm.used_block_ids
    assert right_leaf not in bm.used_block_ids

    assert bm.reclaim_cached_blocks(1) == 1
    assert ancestor not in bm.used_block_ids


def test_reclaim_cached_blocks_prefers_low_frequency_over_older_hot_block():
    bm = BlockManager(num_blocks=2, block_size=2)
    hot = Sequence([1, 2], block_size=2)
    cold = Sequence([3, 4], block_size=2)

    bm.allocate(hot)
    bm.mark_kv_ready([hot])
    hot_id = hot.block_table[0]
    bm.deallocate(hot)
    bm.allocate(cold)
    bm.mark_kv_ready([cold])
    cold_id = cold.block_table[0]
    bm.deallocate(cold)

    bm.blocks[hot_id].access_count = 8
    bm.blocks[hot_id].last_access_time = 1.0
    bm.blocks[cold_id].access_count = 1
    bm.blocks[cold_id].last_access_time = 10.0

    assert bm.reclaim_cached_blocks(1) == 1
    assert hot_id in bm.used_block_ids
    assert cold_id not in bm.used_block_ids


def test_reclaim_for_sequence_evicts_lru_cache_but_keeps_required_prefix():
    bm = BlockManager(num_blocks=2, block_size=2)
    old = Sequence([1, 2], block_size=2)
    keep = Sequence([3, 4], block_size=2)
    bm.allocate(old)
    bm.mark_kv_ready([old])
    bm.deallocate(old)
    bm.allocate(keep)
    bm.mark_kv_ready([keep])
    bm.deallocate(keep)

    old_id = bm.hash_to_block_id[bm.compute_hash([1, 2], -1)]
    keep_id = bm.hash_to_block_id[bm.compute_hash([3, 4], -1)]
    incoming = Sequence([3, 4, 5], block_size=2)

    assert bm.reclaim_for_sequence(incoming) == 1
    assert old_id in bm.free_block_ids
    assert keep_id not in bm.free_block_ids
    assert bm.can_allocate(incoming) is True


def test_reclaim_for_sequence_keeps_other_routed_waiting_prefixes():
    bm = BlockManager(num_blocks=3, block_size=2)
    first = Sequence([1, 2], block_size=2)
    promised = Sequence([3, 4], block_size=2)
    for seq in (first, promised):
        bm.allocate(seq)
        bm.mark_kv_ready([seq])
        bm.deallocate(seq)

    first_id = bm.hash_to_block_id[bm.compute_hash([1, 2], -1)]
    promised_hash = bm.compute_hash([3, 4], -1)
    promised_id = bm.hash_to_block_id[promised_hash]
    incoming = Sequence([5, 6, 7, 8], block_size=2)

    protected = bm.resolve_cached_block_ids([promised_hash])
    assert bm.reclaim_for_sequence(incoming, protected_block_ids=protected) == 1
    assert first_id in bm.free_block_ids
    assert promised_id not in bm.free_block_ids


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
    bm.mark_kv_ready([cached])

    seq = Sequence([1, 2, 3, 4], block_size=2)

    assert len(bm.free_block_ids) == 1
    assert seq.num_blocks == 2
    assert bm.num_required_new_blocks(seq) == 1
    assert bm.can_allocate(seq) is True

    bm.allocate(seq)
    assert seq.num_cached_tokens == 2
