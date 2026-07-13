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


def test_route_sequence_meta_penalizes_busy_prefix_owner():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=8, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [4, 4]
    gbm.running_sequences_per_gpu = [0, 20]
    gbm.global_page_table = {
        321: [
            BlockLocation(0, 0, 321, 1.0),
            BlockLocation(1, 1, 321, 1.0),
            BlockLocation(1, 2, 321, 1.0),
        ]
    }
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    target, info = scheduler.route_sequence_meta(
        requester_rank=0,
        seq_id=7,
        num_tokens=4,
        num_blocks=2,
        prefix_hash=321,
        return_info=True,
    )

    assert target == 0
    assert info["prefix_hit"] is True
    assert info["queue_pressure"][1] > info["queue_pressure"][0]
    assert info["load_score"][1] > info["load_score"][0]


def test_route_sequence_meta_bypasses_overloaded_prefix_owner():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=8, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [4, 4]
    gbm.waiting_tokens_per_gpu = [0, 4096]
    gbm.global_page_table = {
        654: [
            BlockLocation(1, 1, 654, 1.0),
            BlockLocation(1, 2, 654, 1.0),
        ]
    }
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    target, info = scheduler.route_sequence_meta(
        requester_rank=0,
        seq_id=9,
        num_tokens=4,
        num_blocks=2,
        prefix_hash=654,
        return_info=True,
    )

    assert target == 0
    assert info["prefix_hit"] is True
    assert info["reason"] == "prefix_hit_load_bypass"
    assert info["load_score"][1] > info["load_score"][0]


def test_route_sequence_meta_no_hit_uses_queue_pressure_before_free_blocks():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=8, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [3, 8]
    gbm.running_sequences_per_gpu = [0, 10]
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    assert scheduler.route_sequence_meta(
        requester_rank=0,
        seq_id=8,
        num_tokens=4,
        num_blocks=2,
        prefix_hash=None,
    ) == 0


def test_route_sequence_meta_from_ingress_rank_uses_real_gpu_candidates():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [1, 3]
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    seq = Sequence([1, 2, 3], block_size=2)

    assert scheduler.route_sequence_meta(
        requester_rank=-1,
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

    plan = scheduler.plan_rebalance(gpu_id=0, needed_blocks=1, allow_copy=True)
    assert plan is not None
    assert plan["gpu_id"] == 0
    assert plan["needed_blocks"] == 1
    assert plan["mode"] == "move"
    assert plan["transfers"][0]["src_gpu"] == 0
    assert plan["transfers"][0]["mode"] == "move"


def test_plan_rebalance_uses_copy_for_pinned_blocks_when_move_is_impossible():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 2]
    gbm.block_access_time[0] = {}
    gbm.block_hash[0] = {0: 11, 1: 22}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    plan = scheduler.plan_rebalance(gpu_id=0, needed_blocks=1, allow_copy=True)

    assert plan is not None
    assert plan["mode"] == "copy"
    assert plan["transfers"][0]["mode"] == "copy"
    assert plan["transfers"][0]["src_gpu"] == 0
    assert plan["transfers"][0]["dst_gpu"] == 1


def test_plan_rebalance_does_not_copy_by_default_for_foreground_shortage():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 2]
    gbm.block_access_time[0] = {}
    gbm.block_hash[0] = {0: 11, 1: 22}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    assert scheduler.plan_rebalance(gpu_id=0, needed_blocks=1) is None


def test_plan_rebalance_does_not_use_recursive_target_victims():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=8, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 4]
    gbm.block_access_time[0] = {i: float(i) for i in range(5)}
    gbm.block_hash[0] = {i: 100 + i for i in range(5)}
    gbm.block_access_time[1] = {0: 1.0}
    gbm.block_hash[1] = {0: 300}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    assert scheduler.plan_rebalance(gpu_id=0, needed_blocks=5) is None
