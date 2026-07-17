from lmpool.engine.global_block_manager import BlockLocation, GlobalBlockManager
from lmpool.engine.global_scheduler import GlobalScheduler
from lmpool.engine.sequence import Sequence


class DummyBlockManager:
    def __init__(self):
        self.calls = []

    def compute_hash(self, token_ids, prefix_hash_value):
        return prefix_hash_value + sum(token_ids)


def test_route_sequence_meta_does_not_overweight_duplicate_prefix_replicas():
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
    ) == 0


def test_route_sequence_meta_falls_back_when_prefix_owner_has_no_space():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [4, 0]
    gbm.global_page_table = {
        456: [
            BlockLocation(1, 0, 456, 1.0),
            BlockLocation(1, 1, 456, 1.0),
        ]
    }
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    seq = Sequence([1, 2, 3, 4], block_size=2)

    target, info = scheduler.route_sequence_meta(
        requester_rank=0,
        seq_id=seq.seq_id,
        num_tokens=seq.num_tokens,
        num_blocks=seq.num_blocks,
        prefix_hash=456,
        return_info=True,
    )

    assert target == 0
    assert info["reason"] == "prefix_owner_full_fallback"


def test_route_sequence_meta_requests_rebalance_only_when_all_candidates_are_full():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 0]
    gbm.global_page_table = {456: [BlockLocation(1, 0, 456, 1.0)]}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    target, info = scheduler.route_sequence_meta(
        requester_rank=0,
        seq_id=11,
        num_tokens=4,
        num_blocks=2,
        prefix_hash=456,
        return_info=True,
    )

    assert target == 1
    assert info["reason"] == "prefix_hit_needs_rebalance"


def test_route_sequence_meta_uses_longest_contiguous_prefix_when_deepest_hash_misses():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=8, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [8, 2]
    gbm.global_page_table = {
        11: [BlockLocation(1, 0, 11, 1.0)],
        22: [BlockLocation(1, 1, 22, 1.0)],
    }
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    target, info = scheduler.route_sequence_meta(
        requester_rank=0,
        seq_id=12,
        num_tokens=8,
        num_blocks=4,
        prefix_hash=33,
        prefix_hashes=[11, 22, 33],
        return_info=True,
    )

    assert target == 1
    assert info["matched_prefix_blocks"] == 2
    assert info["required_new_blocks"] == 2
    assert info["hit_summary"] == {1: [0, 1]}


def test_route_sequence_meta_rejects_noncontiguous_deep_hash():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=8, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [8, 2]
    gbm.global_page_table = {
        22: [BlockLocation(1, 1, 22, 1.0)],
    }
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    target, info = scheduler.route_sequence_meta(
        requester_rank=0,
        seq_id=13,
        num_tokens=6,
        num_blocks=3,
        prefix_hash=22,
        prefix_hashes=[11, 22],
        return_info=True,
    )

    assert target == 0
    assert info["prefix_hit"] is False
    assert info["reason"] == "most_free_no_prefix_hit"


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


def test_route_sequence_meta_ingress_bypasses_overloaded_owner_to_seed_free_gpu():
    gbm = GlobalBlockManager(rank=0, world_size=4, num_blocks_per_gpu=8, nvlink_pairs=[(0, 1), (2, 3)])
    gbm.free_blocks_per_gpu = [4, 4, 4, 4]
    gbm.waiting_tokens_per_gpu = [0, 4096, 0, 0]
    gbm.global_page_table = {
        777: [
            BlockLocation(1, 0, 777, 1.0),
        ]
    }
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    target, info = scheduler.route_sequence_meta(
        requester_rank=-1,
        seq_id=10,
        num_tokens=512,
        num_blocks=2,
        prefix_hash=777,
        return_info=True,
    )

    assert target != 1
    assert info["prefix_hit"] is True
    assert info["reason"] == "prefix_hit_load_bypass"


def test_route_cost_keeps_long_prefix_on_moderately_loaded_owner():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=8, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [8, 8]
    gbm.waiting_tokens_per_gpu = [0, 600]
    gbm.global_page_table = {
        11: [BlockLocation(1, 0, 11, 1.0)],
        22: [BlockLocation(1, 1, 22, 1.0)],
    }
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    target, info = scheduler.route_sequence_meta(
        requester_rank=-1,
        seq_id=19,
        num_tokens=512,
        num_blocks=2,
        prefix_hash=22,
        prefix_hashes=[11, 22],
        return_info=True,
    )

    assert target == 1
    assert info["reason"] == "prefix_hit"
    assert info["estimated_costs"][1]["prefill"] == 0
    assert info["estimated_costs"][0]["prefill"] == 512


def test_route_bypasses_owner_to_idle_gpu_with_reclaimable_capacity():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=2, nvlink_pairs=[(0, 1)])
    gbm.update_gpu_state(
        0,
        free_blocks=0,
        block_hashes={0: 11, 1: 22},
        evictable_block_hashes={1: 22},
        block_parent_hashes={0: -1, 1: 11},
    )
    gbm.update_gpu_state(
        1,
        free_blocks=1,
        block_hashes={0: 777},
        evictable_block_hashes={0: 777},
        block_parent_hashes={0: -1},
        waiting_tokens=4096,
    )
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    target, info = scheduler.route_sequence_meta(
        requester_rank=-1,
        seq_id=18,
        num_tokens=512,
        num_blocks=2,
        prefix_hash=777,
        prefix_hashes=[777],
        return_info=True,
    )

    assert target == 0
    assert info["reason"] == "prefix_hit_load_bypass"
    assert info["target_free_blocks"] == 0
    assert info["target_reclaimable_blocks"] == 2
    assert info["target_effective_capacity"] == 2
    assert info["uses_reclaimable_capacity"] is True


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
    scheduler.foreground_transfer_min_benefit_ratio = 1.0

    plan = scheduler.plan_rebalance(gpu_id=0, needed_blocks=1, allow_copy=True)
    assert plan is not None
    assert plan["gpu_id"] == 0
    assert plan["needed_blocks"] == 1
    assert plan["mode"] == "chain_move"
    assert plan["transfers"][0]["src_gpu"] == 0
    assert plan["transfers"][0]["mode"] == "chain_move"


def test_plan_rebalance_transfers_complete_chain_and_releases_only_leaf():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 4]
    gbm.block_hash[0] = {0: 11, 1: 22}
    gbm.block_parent_hash[0] = {0: -1, 1: 11}
    gbm.block_access_time[0] = {1: 5.0}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    scheduler.foreground_transfer_min_benefit_ratio = 1.0

    plan = scheduler.plan_rebalance(gpu_id=0, needed_blocks=1)

    assert plan is not None
    transfer = plan["transfers"][0]
    assert transfer["mode"] == "chain_move"
    assert transfer["src_blocks"] == [0, 1]
    assert transfer["hashes"] == [11, 22]
    assert transfer["parent_hashes"] == [-1, 11]
    assert transfer["release_source_blocks"] == [1]


def test_plan_rebalance_copies_pinned_ancestor_but_releases_unpinned_leaf():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 4]
    gbm.block_hash[0] = {0: 11, 1: 22}
    gbm.block_parent_hash[0] = {0: -1, 1: 11}
    gbm.block_access_time[0] = {1: 5.0}
    gbm.pinned_block_ids[0] = {0}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    scheduler.foreground_transfer_min_benefit_ratio = 1.0

    plan = scheduler.plan_rebalance(gpu_id=0, needed_blocks=1)

    assert plan is not None
    transfer = plan["transfers"][0]
    assert transfer["src_blocks"] == [0, 1]
    assert transfer["release_source_blocks"] == [1]


def test_plan_rebalance_releases_linear_chain_suffix_up_to_shortage():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=5, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 3]
    gbm.block_hash[0] = {0: 11, 1: 22, 2: 33}
    gbm.block_parent_hash[0] = {0: -1, 1: 11, 2: 22}
    gbm.block_access_time[0] = {2: 5.0}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    scheduler.foreground_transfer_min_benefit_ratio = 1.0

    plan = scheduler.plan_rebalance(gpu_id=0, needed_blocks=2)

    assert plan is not None
    transfer = plan["transfers"][0]
    assert transfer["src_blocks"] == [0, 1, 2]
    assert transfer["release_source_blocks"] == [2, 1]


def test_plan_rebalance_can_release_target_resident_ancestor_without_resending_it():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 1]
    gbm.block_hash[0] = {0: 11, 1: 22}
    gbm.block_parent_hash[0] = {0: -1, 1: 11}
    gbm.block_access_time[0] = {1: 5.0}
    gbm.block_hash[1] = {0: 11}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    scheduler.foreground_transfer_min_benefit_ratio = 1.0

    plan = scheduler.plan_rebalance(gpu_id=0, needed_blocks=2)

    assert plan is not None
    transfer = plan["transfers"][0]
    assert transfer["src_blocks"] == [1]
    assert transfer["release_source_blocks"] == [1, 0]


def test_plan_rebalance_reuses_ancestors_already_present_on_target():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 1]
    gbm.block_hash[0] = {0: 11, 1: 22}
    gbm.block_parent_hash[0] = {0: -1, 1: 11}
    gbm.block_access_time[0] = {1: 5.0}
    gbm.block_hash[1] = {0: 11}
    gbm.global_page_table = {
        11: [BlockLocation(0, 0, 11, 1.0), BlockLocation(1, 0, 11, 1.0)],
        22: [BlockLocation(0, 1, 22, 1.0)],
    }
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    scheduler.foreground_transfer_min_benefit_ratio = 1.0

    plan = scheduler.plan_rebalance(gpu_id=0, needed_blocks=1)

    assert plan is not None
    transfer = plan["transfers"][0]
    assert transfer["src_blocks"] == [1]
    assert transfer["release_source_blocks"] == [1]


def test_plan_rebalance_shares_one_planned_ancestor_across_two_branches():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 3]
    gbm.block_hash[0] = {0: 11, 1: 22, 2: 33}
    gbm.block_parent_hash[0] = {0: -1, 1: 11, 2: 11}
    gbm.block_access_time[0] = {1: 5.0, 2: 6.0}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    scheduler.foreground_transfer_min_benefit_ratio = 1.0

    plan = scheduler.plan_rebalance(gpu_id=0, needed_blocks=2)

    assert plan is not None
    transfer = plan["transfers"][0]
    assert set(transfer["src_blocks"]) == {0, 1, 2}
    assert set(transfer["release_source_blocks"]) == {1, 2}


def test_plan_rebalance_prioritizes_high_frequency_chain_over_newer_cold_block():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=3, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 2]
    gbm.block_hash[0] = {0: 11, 1: 22}
    gbm.block_parent_hash[0] = {0: -1, 1: -1}
    gbm.block_access_time[0] = {0: 1.0, 1: 10.0}
    gbm.block_access_count[0] = {0: 8, 1: 1}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    scheduler.foreground_transfer_min_benefit_ratio = 1.0

    plan = scheduler.plan_rebalance(gpu_id=0, needed_blocks=1)

    assert plan is not None
    transfer = plan["transfers"][0]
    assert transfer["src_blocks"] == [0]
    assert transfer["access_counts"] == [8]


def test_plan_rebalance_excludes_source_blocks_reserved_by_pending_plans():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 4]
    gbm.block_access_time[0] = {0: 5.0, 1: 10.0}
    gbm.block_hash[0] = {0: 11, 1: 22}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    scheduler.foreground_transfer_min_benefit_ratio = 1.0

    plan = scheduler.plan_rebalance(
        gpu_id=0,
        needed_blocks=1,
        excluded_source_blocks={0},
    )

    assert plan is not None
    assert plan["transfers"][0]["src_blocks"] == [1]


def test_plan_rebalance_uses_copy_for_pinned_blocks_when_move_is_impossible():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 2]
    gbm.block_access_time[0] = {}
    gbm.block_hash[0] = {0: 11, 1: 22}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())
    scheduler.foreground_transfer_min_benefit_ratio = 1.0

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


def test_plan_rebalance_rejects_cold_transfer_below_benefit_threshold():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 4]
    gbm.block_hash[0] = {0: 11}
    gbm.block_access_time[0] = {0: 1.0}
    gbm.block_access_count[0] = {0: 1}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    assert scheduler.plan_rebalance(gpu_id=0, needed_blocks=1) is None
    assert scheduler.last_rebalance_fail_reason == "low_benefit"


def test_plan_rebalance_accepts_hot_transfer_and_reports_cost():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 4]
    gbm.block_hash[0] = {0: 11}
    gbm.block_access_time[0] = {0: 1.0}
    gbm.block_access_count[0] = {0: 3}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    plan = scheduler.plan_rebalance(gpu_id=0, needed_blocks=1)

    assert plan is not None
    assert plan["estimated_benefit_ratio"] == 3.0
    assert plan["estimated_saved_prefill"] > plan["estimated_transfer_cost"]


def test_plan_rebalance_does_not_use_recursive_target_victims():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=8, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [0, 4]
    gbm.block_access_time[0] = {i: float(i) for i in range(5)}
    gbm.block_hash[0] = {i: 100 + i for i in range(5)}
    gbm.block_access_time[1] = {0: 1.0}
    gbm.block_hash[1] = {0: 300}
    scheduler = GlobalScheduler(gbm=gbm, block_manager=DummyBlockManager())

    assert scheduler.plan_rebalance(gpu_id=0, needed_blocks=5) is None
