import threading
import time
import queue

from lmpool.engine.block_manager import BlockManager
from lmpool.engine.control_plane import ControlPlaneClient, control_plane_process
from lmpool.engine.global_block_manager import BlockLocation, GlobalBlockManager
from lmpool.engine.sequence import Sequence


def _start_control_plane(world_size=2, max_cached_blocks=8, pairs=((0, 1),), extra_config=None):
    request_queue = queue.Queue()
    response_queues = {rank: queue.Queue() for rank in range(world_size)}
    response_queues[-1] = queue.Queue()
    config = {
        "world_size": world_size,
        "max_cached_blocks": max_cached_blocks,
        "nvlink_topo": {"pairs": list(pairs)},
        "log_level": "ERROR",
    }
    if extra_config:
        config.update(extra_config)
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


def test_route_falls_back_when_nvlink_prefix_owner_has_no_capacity():
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
        assert target == 0
    finally:
        _stop_control_plane(request_queue, thread)


def test_route_cache_reuses_valid_prefix_owner():
    config, request_queue, response_queues, thread = _start_control_plane(
        extra_config={"enable_route_cache": True, "route_cache_queue_slack": 100000.0}
    )
    try:
        bm = BlockManager(num_blocks=8, block_size=2)
        client0 = ControlPlaneClient(0, request_queue, response_queues[0])
        client0.block_manager = bm

        prefix_tokens = [11, 22]
        prefix_hash = bm.compute_hash(prefix_tokens, -1)
        request_queue.put({
            "type": "block_state",
            "rank": 1,
            "free_blocks": 6,
            "block_hashes": {0: prefix_hash},
        })
        time.sleep(0.2)

        first = client0.route_sequence(
            Sequence(token_ids=prefix_tokens + [33], block_size=2),
            return_meta=True,
        )
        second = client0.route_sequence(
            Sequence(token_ids=prefix_tokens + [44], block_size=2),
            return_meta=True,
        )

        assert first["target_rank"] == 1
        assert second["target_rank"] == 1
        assert second["route_info"]["reason"] == "route_cache"
    finally:
        _stop_control_plane(request_queue, thread)


def test_route_cache_bypasses_congested_cached_owner():
    config, request_queue, response_queues, thread = _start_control_plane(
        extra_config={"enable_route_cache": True}
    )
    try:
        bm = BlockManager(num_blocks=8, block_size=2)
        client0 = ControlPlaneClient(0, request_queue, response_queues[0])
        client0.block_manager = bm

        prefix_tokens = [11, 22]
        prefix_hash = bm.compute_hash(prefix_tokens, -1)
        request_queue.put({
            "type": "block_state",
            "rank": 0,
            "free_blocks": 6,
            "block_hashes": {},
            "waiting_sequences": 0,
            "running_sequences": 0,
        })
        request_queue.put({
            "type": "block_state",
            "rank": 1,
            "free_blocks": 6,
            "block_hashes": {0: prefix_hash},
            "waiting_sequences": 0,
            "running_sequences": 0,
        })
        time.sleep(0.2)

        first = client0.route_sequence(
            Sequence(token_ids=prefix_tokens + [33], block_size=2),
            return_meta=True,
        )
        assert first["target_rank"] == 1

        request_queue.put({
            "type": "block_state",
            "rank": 1,
            "free_blocks": 6,
            "block_hashes": {0: prefix_hash},
            "waiting_sequences": 0,
            "running_sequences": 20,
        })
        time.sleep(0.2)

        second = client0.route_sequence(
            Sequence(token_ids=prefix_tokens + [44], block_size=2),
            return_meta=True,
        )
        assert second["target_rank"] == 0
        assert second["route_info"]["reason"] != "route_cache"
    finally:
        _stop_control_plane(request_queue, thread)


def test_route_cache_counts_optimistic_load_before_worker_report():
    config, request_queue, response_queues, thread = _start_control_plane(
        world_size=4,
        pairs=((0, 1), (2, 3)),
        extra_config={
            "enable_route_cache": True,
            "route_load_weight": 0.03,
            "route_load_bypass_threshold": 256,
            "route_cache_queue_slack": 256,
        },
    )
    try:
        bm = BlockManager(num_blocks=8, block_size=2)
        client = ControlPlaneClient(-1, request_queue, response_queues[-1])
        client.block_manager = bm

        prefix_tokens = [11, 22]
        prefix_hash = bm.compute_hash(prefix_tokens, -1)
        for rank in range(4):
            request_queue.put({
                "type": "block_state",
                "rank": rank,
                "free_blocks": 64,
                "block_hashes": {0: prefix_hash},
                "waiting_sequences": 0,
                "running_sequences": 0,
                "waiting_tokens": 0,
                "running_tokens": 0,
            })
        time.sleep(0.2)

        targets = [
            client.route_sequence(
                Sequence(token_ids=prefix_tokens + [idx] * 512, block_size=2),
                return_meta=True,
            )["target_rank"]
            for idx in range(12)
        ]

        assert len(set(targets)) > 1
        assert max(targets.count(rank) for rank in set(targets)) < 10
    finally:
        _stop_control_plane(request_queue, thread)


def test_route_cache_balances_across_prefix_owners():
    config, request_queue, response_queues, thread = _start_control_plane(
        world_size=4,
        pairs=((0, 1), (2, 3)),
        extra_config={
            "enable_route_cache": True,
            "route_load_weight": 0.03,
            "route_cache_queue_slack": 256,
        },
    )
    try:
        bm = BlockManager(num_blocks=8, block_size=2)
        client = ControlPlaneClient(-1, request_queue, response_queues[-1])
        client.block_manager = bm

        prefix_tokens = [11, 22]
        prefix_hash = bm.compute_hash(prefix_tokens, -1)
        for rank in range(4):
            request_queue.put({
                "type": "block_state",
                "rank": rank,
                "free_blocks": 64,
                "block_hashes": {0: prefix_hash},
                "waiting_sequences": 0,
                "running_sequences": 0,
                "waiting_tokens": 4096 if rank == 0 else 0,
                "running_tokens": 0,
            })
        time.sleep(0.2)

        first = client.route_sequence(
            Sequence(token_ids=prefix_tokens + [33], block_size=2),
            return_meta=True,
        )
        second = client.route_sequence(
            Sequence(token_ids=prefix_tokens + [44], block_size=2),
            return_meta=True,
        )

        assert first["target_rank"] != 0
        assert second["target_rank"] != 0
        assert second["route_info"]["reason"] == "route_cache"
    finally:
        _stop_control_plane(request_queue, thread)


def test_rebalance_round_trip_executes_on_source_and_target():
    config, request_queue, response_queues, control_thread = _start_control_plane()
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

        pump_thread = threading.Thread(target=pump_target, daemon=True)
        pump_thread.start()
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

            assert client0.rebalance(0, 1, allow_copy=True) is True
        finally:
            stop.set()
            pump_thread.join(timeout=5)

        ranks = {entry[0] for entry in seen}
        assert ranks == {0, 1}
        assert all(entry[2] == 0 for entry in seen)
        assert any(1 in entry[3] for entry in seen)
    finally:
        _stop_control_plane(request_queue, control_thread)


def test_rebalance_can_plan_copy_for_pinned_source_blocks():
    config, request_queue, response_queues, control_thread = _start_control_plane()
    try:
        client0 = ControlPlaneClient(0, request_queue, response_queues[0])
        client1 = ControlPlaneClient(1, request_queue, response_queues[1])
        modes = []

        def make_executor(rank):
            def _executor(plan):
                modes.append((rank, plan["_phase"], plan["mode"]))
                return {"success": True}

            return _executor

        client0.set_rebalance_executor(make_executor(0))
        client1.set_rebalance_executor(make_executor(1))

        stop = threading.Event()

        def pump_target():
            while not stop.is_set():
                client1.pump_async_messages()
                time.sleep(0.01)

        pump_thread = threading.Thread(target=pump_target, daemon=True)
        pump_thread.start()
        try:
            request_queue.put({
                "type": "block_state",
                "rank": 0,
                "free_blocks": 0,
                "block_hashes": {0: 101},
                "evictable_block_hashes": {},
                "pinned_block_hashes": {0: 101},
            })
            request_queue.put({
                "type": "block_state",
                "rank": 1,
                "free_blocks": 2,
                "block_hashes": {},
            })
            time.sleep(0.2)

            assert client0.rebalance(0, 1, allow_copy=True) is True
        finally:
            stop.set()
            pump_thread.join(timeout=5)

        assert any(mode == "copy" for _, _, mode in modes)
        assert (0, "execute", "copy") in modes
        assert (1, "execute", "copy") in modes
    finally:
        _stop_control_plane(request_queue, control_thread)


def test_route_schedules_background_copy_without_blocking_response():
    config, request_queue, response_queues, control_thread = _start_control_plane(
        extra_config={
            "enable_background_copy": True,
            "background_copy_max_blocks": 1,
            "background_copy_cooldown_s": 0.0,
            "background_copy_hot_threshold": 1,
        }
    )
    try:
        bm = BlockManager(num_blocks=8, block_size=2)
        client0 = ControlPlaneClient(0, request_queue, response_queues[0])
        client1 = ControlPlaneClient(1, request_queue, response_queues[1])
        client0.block_manager = bm

        seen = []

        def make_executor(rank):
            def _executor(plan):
                seen.append((rank, plan["_phase"], plan["mode"], bool(plan.get("background"))))
                return {"success": True}

            return _executor

        client0.set_rebalance_executor(make_executor(0))
        client1.set_rebalance_executor(make_executor(1))

        prefix_tokens = [11, 22]
        prefix_hash = bm.compute_hash(prefix_tokens, -1)
        request_queue.put({
            "type": "block_state",
            "rank": 0,
            "free_blocks": 4,
            "block_hashes": {0: prefix_hash},
            "evictable_block_hashes": {},
            "pinned_block_hashes": {0: prefix_hash},
        })
        request_queue.put({
            "type": "block_state",
            "rank": 1,
            "free_blocks": 4,
            "block_hashes": {},
        })
        time.sleep(0.2)

        result = client0.route_sequence(
            Sequence(token_ids=prefix_tokens + [33], block_size=2),
            return_meta=True,
        )
        assert result["target_rank"] == 0

        deadline = time.time() + 5
        while time.time() < deadline:
            client0.pump_async_messages()
            client1.pump_async_messages()
            if (0, "execute", "copy", True) in seen and (1, "execute", "copy", True) in seen:
                break
            time.sleep(0.01)

        assert (0, "execute", "copy", True) in seen
        assert (1, "execute", "copy", True) in seen
    finally:
        _stop_control_plane(request_queue, control_thread)


def test_background_copy_uses_ordered_prefix_hash_chain():
    config, request_queue, response_queues, control_thread = _start_control_plane(
        extra_config={
            "enable_background_copy": True,
            "background_copy_max_blocks": 2,
            "background_copy_cooldown_s": 0.0,
            "background_copy_hot_threshold": 1,
        }
    )
    try:
        bm = BlockManager(num_blocks=8, block_size=2)
        client0 = ControlPlaneClient(0, request_queue, response_queues[0])
        client1 = ControlPlaneClient(1, request_queue, response_queues[1])
        client0.block_manager = bm

        executed_plans = []

        def make_executor(rank):
            def _executor(plan):
                if plan["_phase"] == "execute":
                    executed_plans.append((rank, plan))
                return {"success": True}

            return _executor

        client0.set_rebalance_executor(make_executor(0))
        client1.set_rebalance_executor(make_executor(1))

        h0 = bm.compute_hash([1, 2], -1)
        h1 = bm.compute_hash([3, 4], h0)
        h2 = bm.compute_hash([5, 6], h1)
        request_queue.put({
            "type": "block_state",
            "rank": 0,
            "free_blocks": 4,
            "block_hashes": {0: h0, 1: h1, 2: h2},
            "evictable_block_hashes": {},
            "pinned_block_hashes": {0: h0, 1: h1, 2: h2},
        })
        request_queue.put({
            "type": "block_state",
            "rank": 1,
            "free_blocks": 4,
            "block_hashes": {},
        })
        time.sleep(0.2)

        result = client0.route_sequence(
            Sequence(token_ids=[1, 2, 3, 4, 5, 6, 7], block_size=2),
            return_meta=True,
        )
        assert result["target_rank"] == 0

        deadline = time.time() + 5
        while time.time() < deadline:
            client0.pump_async_messages()
            client1.pump_async_messages()
            if len([entry for entry in executed_plans if entry[0] == 0]) >= 1:
                break
            time.sleep(0.01)

        source_plans = [plan for rank, plan in executed_plans if rank == 0]
        assert source_plans
        transfer = source_plans[0]["transfers"][0]
        assert transfer["src_blocks"] == [0, 1]
        assert transfer["hashes"] == [h0, h1]
    finally:
        _stop_control_plane(request_queue, control_thread)


def test_background_copy_waits_for_hot_prefix_threshold():
    config, request_queue, response_queues, control_thread = _start_control_plane(
        extra_config={
            "enable_background_copy": True,
            "background_copy_max_blocks": 1,
            "background_copy_cooldown_s": 0.0,
            "background_copy_hot_threshold": 2,
        }
    )
    try:
        bm = BlockManager(num_blocks=8, block_size=2)
        client0 = ControlPlaneClient(0, request_queue, response_queues[0])
        client1 = ControlPlaneClient(1, request_queue, response_queues[1])
        client0.block_manager = bm
        seen = []

        def make_executor(rank):
            def _executor(plan):
                seen.append((rank, plan["_phase"], plan["mode"], bool(plan.get("background"))))
                return {"success": True}

            return _executor

        client0.set_rebalance_executor(make_executor(0))
        client1.set_rebalance_executor(make_executor(1))

        prefix_tokens = [11, 22]
        prefix_hash = bm.compute_hash(prefix_tokens, -1)
        request_queue.put({
            "type": "block_state",
            "rank": 0,
            "free_blocks": 4,
            "block_hashes": {0: prefix_hash},
            "evictable_block_hashes": {},
            "pinned_block_hashes": {0: prefix_hash},
        })
        request_queue.put({
            "type": "block_state",
            "rank": 1,
            "free_blocks": 4,
            "block_hashes": {},
        })
        time.sleep(0.2)

        client0.route_sequence(Sequence(token_ids=prefix_tokens + [33], block_size=2))
        client0.pump_async_messages()
        client1.pump_async_messages()
        assert not seen

        client0.route_sequence(Sequence(token_ids=prefix_tokens + [44], block_size=2))
        deadline = time.time() + 5
        while time.time() < deadline:
            client0.pump_async_messages()
            client1.pump_async_messages()
            if (0, "execute", "copy", True) in seen and (1, "execute", "copy", True) in seen:
                break
            time.sleep(0.01)

        assert (0, "execute", "copy", True) in seen
        assert (1, "execute", "copy", True) in seen
    finally:
        _stop_control_plane(request_queue, control_thread)


def test_rebalance_prepare_failure_does_not_execute_source():
    config, request_queue, response_queues, control_thread = _start_control_plane()
    try:
        client0 = ControlPlaneClient(0, request_queue, response_queues[0])
        client1 = ControlPlaneClient(1, request_queue, response_queues[1])

        seen = []

        def source_executor(plan):
            seen.append(("source", plan["_phase"]))
            return True

        def target_executor(plan):
            seen.append(("target", plan["_phase"]))
            if plan["_phase"] == "prepare":
                return {"success": False, "error": "not enough free blocks"}
            return {"success": True}

        client0.set_rebalance_executor(source_executor)
        client1.set_rebalance_executor(target_executor)

        stop = threading.Event()

        def pump_target():
            while not stop.is_set():
                client1.pump_async_messages()
                time.sleep(0.01)

        pump_thread = threading.Thread(target=pump_target, daemon=True)
        pump_thread.start()
        try:
            request_queue.put({
                "type": "block_state",
                "rank": 0,
                "free_blocks": 0,
                "block_hashes": {i: 100 + i for i in range(5)},
            })
            request_queue.put({
                "type": "block_state",
                "rank": 1,
                "free_blocks": 5,
                "block_hashes": {},
            })
            time.sleep(0.2)

            assert client0.rebalance(0, 5) is False
        finally:
            stop.set()
            pump_thread.join(timeout=5)

        assert ("target", "prepare") in seen
        assert ("source", "execute") not in seen
    finally:
        _stop_control_plane(request_queue, control_thread)


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


def test_ready_blocks_register_into_global_page_table_after_prefill():
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
    assert gbm.global_page_table == {}

    bm.mark_kv_ready([seq])
    hashes = list(gbm.global_page_table.keys())
    assert len(hashes) == 2
