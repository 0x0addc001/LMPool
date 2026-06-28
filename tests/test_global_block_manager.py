from lmpool.engine.global_block_manager import BlockLocation, GlobalBlockManager


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

    assert gbm.get_block_location(555)[0].gpu_id == 1
    assert gbm.get_block_hash(1, 3) == 555
