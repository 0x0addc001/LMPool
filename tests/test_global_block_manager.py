from lmpool.engine.global_block_manager import (
    BlockLocation,
    GlobalBlockManager,
    detect_nvlink_pairs_from_nvidia_smi,
)


def test_lookup_prefix_prefers_nvlink_partner():
    gbm = GlobalBlockManager(
        rank=0,
        world_size=3,
        num_blocks_per_gpu=4,
        nvlink_pairs=[(0, 1)],
    )
    gbm.global_page_table = {
        99: [
            BlockLocation(0, 0, 99, 1.0),
            BlockLocation(1, 1, 99, 2.0),
        ]
    }

    hits = gbm.lookup_prefix(99, requester_rank=0)
    assert [loc.gpu_id for loc in hits] == [1, 0]


def test_update_allocate_and_free_global_state():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.update_gpu_state(0, 3, {0: 111, 1: 222})
    assert gbm.get_free_blocks_count(0) == 3
    assert gbm.get_block_hash(0, 0) == 111
    assert gbm.get_block_location(111)[0].gpu_id == 0

    allocated = gbm.allocate_global(1, 2, [333, 444])
    assert allocated == [0, 1]
    assert gbm.get_free_blocks_count(1) == 2
    gbm.free_global(1, [0])
    assert gbm.get_free_blocks_count(1) == 3


def test_update_gpu_state_uses_only_evictable_blocks_for_eviction():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.update_gpu_state(
        0,
        free_blocks=0,
        block_hashes={0: 111, 1: 222},
        evictable_block_hashes={1: 222},
        pinned_block_hashes={0: 111},
    )
    gbm.free_blocks_per_gpu[1] = 1

    assert gbm.select_eviction_candidates(0, 1, allow_recursive=False) == [(1, 1)]


def test_global_block_manager_reconstructs_root_to_leaf_chain():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.update_gpu_state(
        0,
        free_blocks=1,
        block_hashes={0: 11, 1: 22, 2: 33},
        evictable_block_hashes={2: 33},
        block_parent_hashes={0: -1, 1: 11, 2: 22},
    )

    assert gbm.get_prefix_chain(0, 2) == [0, 1, 2]


def test_update_gpu_state_tracks_queue_pressure_snapshot():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.update_gpu_state(
        0,
        free_blocks=2,
        block_hashes={},
        waiting_sequences=3,
        running_sequences=4,
    )

    assert gbm.waiting_sequences_per_gpu[0] == 3
    assert gbm.running_sequences_per_gpu[0] == 4
    assert gbm.get_queue_pressure(0) == 11.0


def test_worker_snapshot_does_not_erase_pending_route_load():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.reserve_route_load(1, num_tokens=2048, seq_id=17)

    # This snapshot may have been produced before the routed request reached
    # the worker. The pending admission must remain visible to routing.
    gbm.update_gpu_state(
        1,
        free_blocks=4,
        block_hashes={},
        waiting_sequences=0,
        running_sequences=0,
        waiting_tokens=0,
        running_tokens=0,
    )
    assert gbm.get_load_score(1) == 2080.0

    # An unrelated admission must not consume this sequence's reservation.
    gbm.acknowledge_route_load(1, num_tokens=2048, seq_id=18)
    assert gbm.get_load_score(1) == 2080.0

    # The worker publishes its admitted waiting state before acknowledging the
    # optimistic reservation. Removing pending load must therefore leave the
    # real waiting load visible.
    gbm.update_gpu_state(
        1,
        free_blocks=4,
        block_hashes={},
        waiting_sequences=1,
        running_sequences=0,
        waiting_tokens=2048,
        running_tokens=0,
    )
    gbm.acknowledge_route_load(1, num_tokens=2048, seq_id=17)
    assert gbm.get_load_score(1) == 2048.0


def test_select_eviction_candidates_can_recurse_to_nvlink_partner():
    gbm = GlobalBlockManager(
        rank=0,
        world_size=2,
        num_blocks_per_gpu=2,
        nvlink_pairs=[(0, 1)],
    )
    gbm.free_blocks_per_gpu = [0, 0]
    gbm.block_access_time[0] = {0: 10.0, 1: 20.0}
    gbm.block_hash[0] = {0: 100, 1: 200}
    gbm.block_access_time[1] = {0: 5.0}
    gbm.block_hash[1] = {0: 300}
    gbm.global_page_table = {
        100: [BlockLocation(0, 0, 100, 10.0)],
        200: [BlockLocation(0, 1, 200, 20.0)],
        300: [BlockLocation(1, 0, 300, 5.0)],
    }

    candidates = gbm.select_eviction_candidates(0, 1)
    assert candidates == [(0, 1)]
    assert gbm.get_block_location(300) == []


def test_select_eviction_candidates_can_disable_recursive_overcommit():
    gbm = GlobalBlockManager(
        rank=0,
        world_size=2,
        num_blocks_per_gpu=8,
        nvlink_pairs=[(0, 1)],
    )
    gbm.free_blocks_per_gpu = [0, 4]
    gbm.block_access_time[0] = {i: float(i) for i in range(5)}
    gbm.block_hash[0] = {i: 100 + i for i in range(5)}
    gbm.block_access_time[1] = {0: 1.0}
    gbm.block_hash[1] = {0: 300}

    assert gbm.select_eviction_candidates(0, 5, allow_recursive=False) == []


def test_select_eviction_candidates_returns_empty_without_nvlink_partner():
    gbm = GlobalBlockManager(
        rank=0,
        world_size=2,
        num_blocks_per_gpu=2,
        nvlink_pairs=[],
    )
    gbm.free_blocks_per_gpu = [0, 0]
    gbm.block_access_time[0] = {0: 10.0}
    gbm.block_hash[0] = {0: 100}

    assert gbm.select_eviction_candidates(0, 1) == []


def test_select_eviction_candidates_returns_empty_when_partner_has_no_victim():
    gbm = GlobalBlockManager(
        rank=0,
        world_size=2,
        num_blocks_per_gpu=2,
        nvlink_pairs=[(0, 1)],
    )
    gbm.free_blocks_per_gpu = [0, 0]
    gbm.block_access_time[0] = {0: 10.0}
    gbm.block_hash[0] = {0: 100}
    gbm.block_access_time[1] = {}
    gbm.block_hash[1] = {}

    assert gbm.select_eviction_candidates(0, 1) == []


def test_record_block_transfer_moves_location():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [1, 3]
    gbm.block_hash[0] = {2: 555}
    gbm.block_access_time[0] = {2: 1.0}
    gbm.global_page_table = {555: [BlockLocation(0, 2, 555, 1.0)]}

    gbm.record_block_transfer(block_id=2, src_gpu=0, dst_gpu=1, new_block_id=3)

    assert {(loc.gpu_id, loc.block_id) for loc in gbm.get_block_location(555)} == {(1, 3)}
    assert gbm.get_block_hash(1, 3) == 555
    assert gbm.get_free_blocks_count(0) == 2
    assert gbm.get_free_blocks_count(1) == 2


def test_record_block_copy_keeps_source_location():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    gbm.free_blocks_per_gpu = [1, 3]
    gbm.block_hash[0] = {2: 555}
    gbm.block_access_time[0] = {2: 1.0}
    gbm.global_page_table = {555: [BlockLocation(0, 2, 555, 1.0)]}

    gbm.record_block_copy(block_id=2, src_gpu=0, dst_gpu=1, new_block_id=3)

    locations = {(loc.gpu_id, loc.block_id) for loc in gbm.get_block_location(555)}
    assert locations == {(0, 2), (1, 3)}
    assert gbm.get_free_blocks_count(0) == 1
    assert gbm.get_free_blocks_count(1) == 2

    assert gbm.get_block_hash(1, 3) == 555


def test_detect_nvlink_pairs_from_nvidia_smi_parses_topology(monkeypatch):
    topo = """
        GPU0    GPU1    CPU Affinity
        GPU0     X      NV1     0-15
        GPU1    NV1      X      0-15
    """

    def fake_check_output(*args, **kwargs):
        return topo

    monkeypatch.setattr("lmpool.engine.global_block_manager.subprocess.check_output", fake_check_output)

    assert detect_nvlink_pairs_from_nvidia_smi(2) == [(0, 1)]


def test_detect_nvlink_pairs_from_nvidia_smi_returns_empty_on_failure(monkeypatch):
    def fake_check_output(*args, **kwargs):
        raise RuntimeError("missing nvidia-smi")

    monkeypatch.setattr("lmpool.engine.global_block_manager.subprocess.check_output", fake_check_output)

    assert detect_nvlink_pairs_from_nvidia_smi(2) == []
