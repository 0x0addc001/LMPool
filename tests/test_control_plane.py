import threading
import time
import queue

from lmpool.engine.block_manager import BlockManager
from lmpool.engine.control_plane import ControlPlaneClient, control_plane_process
from lmpool.engine.global_block_manager import BlockLocation, GlobalBlockManager
from lmpool.engine.sequence import Sequence


def _start_control_plane(world_size=2, max_cached_blocks=8, pairs=((0, 1),)):
    request_queue = queue.Queue()
    response_queues = {rank: queue.Queue() for rank in range(world_size)}
    config = {
        "world_size": world_size,
        "max_cached_blocks": max_cached_blocks,
        "nvlink_topo": {"pairs": list(pairs)},
        "log_level": "ERROR",
    }
    thread = threading.Thread(
        target=control_plane_process,
        args=(config, request_queue, response_queues),
        daemon=True,
    )
    thread.start()
    return config, request_queue, response_queues, thread


def _stop_control_plane(request_queue, thread, timeout=10):
    request_queue.put({"type": "shutdown"})
    thread.join(timeout=timeout)


def test_route_prefers_nvlink_and_capacity_fallback():
    config, request_queue, response_queues, thread = _start_control_plane()
    try:
        bm = BlockManager(num_blocks=8, block_size=2)
        client0 = ControlPlaneClient(0, request_queue, response_queues[0])
        client0.block_manager = bm

        prefix_tokens = [11, 22]
        prefix_hash = bm.compute_hash(prefix_tokens, -1)
        seq = Sequence(token_ids=prefix_tokens + [33], block_size=2)

        request_queue.put({
            "type": "block_state",
            "rank": 0,
            "free_blocks": 6,
            "block_hashes": {0: 1001},
        })
        request_queue.put({
            "type": "block_state",
            "rank": 1,
            "free_blocks": 0,
            "block_hashes": {0: prefix_hash},
        })
        time.sleep(0.2)

        target = client0.route_sequence(seq)
        assert target == 1
    finally:
        _stop_control_plane(request_queue, thread)


def test_rebalance_round_trip_executes_on_source_and_target():
    config, request_queue, response_queues, thread = _start_control_plane()
    try:
        client0 = ControlPlaneClient(0, request_queue, response_queues[0])
        client1 = ControlPlaneClient(1, request_queue, response_queues[1])

        seen = []

        def make_executor(rank):
            def _executor(plan):
                seen.append((rank, plan["plan_id"], plan["gpu_id"], tuple(t["dst_gpu"] for t in plan["transfers"])))
                return True

            return _executor

        client0.set_rebalance_executor(make_executor(0))
        client1.set_rebalance_executor(make_executor(1))

        stop = threading.Event()

        def pump_target():
            while not stop.is_set():
                client1.pump_async_messages()
                time.sleep(0.01)

        thread = threading.Thread(target=pump_target, daemon=True)
        thread.start()
        try:
            request_queue.put({
                "type": "block_state",
                "rank": 0,
                "free_blocks": 0,
                "block_hashes": {0: 101, 1: 202},
            })
            request_queue.put({
                "type": "block_state",
                "rank": 1,
                "free_blocks": 4,
                "block_hashes": {},
            })
            time.sleep(0.2)

            assert client0.rebalance(0, 1) is True
        finally:
            stop.set()
            thread.join(timeout=5)

        ranks = {entry[0] for entry in seen}
        assert ranks == {0, 1}
        assert all(entry[2] == 0 for entry in seen)
        assert any(1 in entry[3] for entry in seen)
    finally:
        _stop_control_plane(request_queue, thread)


def test_nvlink_only_block_manager_targets_peer_only():
    gbm = GlobalBlockManager(
        rank=0,
        world_size=3,
        num_blocks_per_gpu=4,
        nvlink_pairs=[(0, 1)],
    )
    gbm.free_blocks_per_gpu = [0, 0, 4]
    gbm.block_access_time[0] = {0: 10.0, 1: 20.0}
    gbm.block_hash[0] = {0: 111, 1: 222}
    gbm.block_access_time[1] = {0: 5.0}
    gbm.block_hash[1] = {0: 333}

    candidates = gbm.select_eviction_candidates(0, 1)
    assert candidates == [(0, 1)]


def test_lookup_prefix_uses_requester_rank_for_nvlink_affinity():
    gbm = GlobalBlockManager(
        rank=0,
        world_size=3,
        num_blocks_per_gpu=4,
        nvlink_pairs=[(0, 1)],
    )
    gbm.global_page_table = {
        1234: [
            BlockLocation(0, 0, 1234, 1.0),
            BlockLocation(1, 0, 1234, 2.0),
        ]
    }

    hits = gbm.lookup_prefix(1234, requester_rank=0)
    assert [loc.gpu_id for loc in hits] == [1, 0]


def test_allocate_registers_all_full_blocks_into_global_page_table():
    gbm = GlobalBlockManager(
        rank=0,
        world_size=2,
        num_blocks_per_gpu=8,
        nvlink_pairs=[(0, 1)],
    )
    bm = BlockManager(num_blocks=8, block_size=2, gbm=gbm)
    seq = Sequence(token_ids=[1, 2, 3, 4], block_size=2)

    bm.allocate(seq)

    assert len(seq.block_table) == 2
    hashes = list(gbm.global_page_table.keys())
    assert len(hashes) == 2
