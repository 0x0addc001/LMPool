from lmpool.engine.global_block_manager import BlockLocation, GlobalBlockManager
from lmpool.engine.global_scheduler import GlobalScheduler
from lmpool.engine.sequence import Sequence


class DummyBlockManager:
    def __init__(self):
        self.calls = []

    def compute_hash(self, token_ids, prefix_hash_value):
        return prefix_hash_value + sum(token_ids)


def test_route_sequence_meta_prefers_prefix_hit_then_free_gpu():
    gbm = GlobalBlockManager(rank=0, world_size=3, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [2, 3, 1]
    gbm.global_page_table = {
        123: [
            BlockLocation(0, 0, 123, 1.0),
            BlockLocation(1, 1, 123, 1.0),
            BlockLocation(1, 2, 123, 1.0),
        ]
    }
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    seq = Sequence([1, 2, 3, 4], block_size=2)

    assert scheduler.route_sequence_meta(
        requester_rank=0,
        seq_id=seq.seq_id,
        num_tokens=seq.num_tokens,
        num_blocks=seq.num_blocks,
        prefix_hash=123,
    ) == 1


def test_route_sequence_meta_prefers_hit_even_when_free_is_insufficient():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [4, 1]
    gbm.global_page_table = {
        456: [
            BlockLocation(1, 0, 456, 1.0),
            BlockLocation(1, 1, 456, 1.0),
        ]
    }
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    seq = Sequence([1, 2, 3, 4], block_size=2)

    assert scheduler.route_sequence_meta(
        requester_rank=0,
        seq_id=seq.seq_id,
        num_tokens=seq.num_tokens,
        num_blocks=seq.num_blocks,
        prefix_hash=456,
    ) == 1


def test_route_sequence_meta_falls_back_to_free_gpu_on_prefix_miss():
    gbm = GlobalBlockManager(rank=0, world_size=3, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [1, 4, 2]
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    seq = Sequence([1, 2, 3, 4], block_size=2)

    assert scheduler.route_sequence_meta(
        requester_rank=0,
        seq_id=seq.seq_id,
        num_tokens=seq.num_tokens,
        num_blocks=seq.num_blocks,
        prefix_hash=999,
    ) == 1


def test_route_sequence_meta_falls_back_to_most_free_local_or_partner():
    gbm = GlobalBlockManager(rank=0, world_size=3, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [1, 2, 4]
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    seq = Sequence([1, 2, 3], block_size=2)

    assert scheduler.route_sequence_meta(
        requester_rank=0,
        seq_id=seq.seq_id,
        num_tokens=seq.num_tokens,
        num_blocks=seq.num_blocks,
        prefix_hash=None,
    ) == 1


def test_plan_rebalance_groups_transfers():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 4]
    gbm.block_access_time[0] = {0: 5.0, 1: 10.0}
    gbm.block_hash[0] = {0: 11, 1: 22}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    plan = scheduler.plan_rebalance(gpu_id=0, needed_blocks=1)
    assert plan is not None
    assert plan["gpu_id"] == 0
    assert plan["needed_blocks"] == 1
    assert plan["transfers"][0]["src_gpu"] == 0
