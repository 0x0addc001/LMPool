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
        # Most control-plane tests validate two-phase transfer protocol rather
        # than admission economics; dedicated scheduler tests cover the
        # production benefit threshold.
        "foreground_transfer_min_benefit_ratio": 0.0,
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


def _get_message_type(response_queue, message_type, timeout=5):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise queue.Empty(f"timed out waiting for {message_type}")
        message = response_queue.get(timeout=remaining)
        if message.get("type") == message_type:
            return message


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


def test_route_uses_hash_chain_and_reserves_only_uncached_blocks():
    config, request_queue, response_queues, thread = _start_control_plane()
    try:
        bm = BlockManager(num_blocks=8, block_size=2)
        client = ControlPlaneClient(-1, request_queue, response_queues[-1])
        client.block_manager = bm
        seq = Sequence(token_ids=[1, 2, 3, 4, 5, 6, 7], block_size=2)
        hashes = client._compute_prefix_hashes(seq)

        request_queue.put({
            "type": "block_state",
            "rank": 0,
            "free_blocks": 8,
            "block_hashes": {},
        })
        request_queue.put({
            "type": "block_state",
            "rank": 1,
            "free_blocks": 2,
            "block_hashes": {0: hashes[0], 1: hashes[1], 2: hashes[2]},
        })
        time.sleep(0.2)

        first = client.route_sequence(seq, return_meta=True)
        request_queue.put({
            "type": "route_admitted",
            "rank": first["target_rank"],
            "seq_id": seq.seq_id,
            "num_tokens": seq.num_tokens,
        })
        time.sleep(0.05)
        second = client.route_sequence(
            Sequence(token_ids=[1, 2, 3, 4, 5, 6, 8], block_size=2),
            return_meta=True,
        )

        assert first["target_rank"] == 1
        assert first["route_info"]["matched_prefix_blocks"] == 3
        assert first["route_info"]["required_new_blocks"] == 1
        assert seq.routed_prefix_hashes == hashes[:3]
        assert second["target_rank"] == 1
        assert second["route_info"]["required_new_blocks"] == 1
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


def test_concurrent_rebalance_plans_serialize_on_nvlink_pair():
    config, request_queue, response_queues, control_thread = _start_control_plane()
    try:
        request_queue.put({
            "type": "block_state",
            "rank": 0,
            "free_blocks": 0,
            "block_hashes": {0: 101, 1: 202},
            "evictable_block_hashes": {0: 101, 1: 202},
            "pinned_block_hashes": {},
        })
        request_queue.put({
            "type": "block_state",
            "rank": 1,
            "free_blocks": 4,
            "block_hashes": {},
            "evictable_block_hashes": {},
            "pinned_block_hashes": {},
        })
        time.sleep(0.2)

        for request_id in ("first", "second"):
            request_queue.put({
                "type": "rebalance_request",
                "request_id": request_id,
                "reply_rank": -1,
                "gpu_id": 0,
                "needed_blocks": 1,
                "allow_copy": False,
            })

        first = _get_message_type(response_queues[0], "rebalance_prepare")
        second = _get_message_type(response_queues[-1], "rebalance_response")

        assert first["plan"]["transfers"][0]["src_blocks"]
        assert second["request_id"] == "second"
        assert second["success"] is False
        assert second["reason"] == "pair_busy"
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


def test_hot_block_state_schedules_background_copy_without_blocking_route():
    config, request_queue, response_queues, control_thread = _start_control_plane(
        extra_config={
            "enable_background_copy": True,
            "background_copy_max_blocks": 1,
            "background_copy_cooldown_s": 0.0,
            "background_copy_hot_threshold": 1,
            "background_copy_min_load_skew": 0.0,
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
            "block_access_stats": {
                0: {"last_access_time": 1.0, "access_count": 2},
            },
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
            "background_copy_min_load_skew": 0.0,
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
            "block_parent_hashes": {0: -1, 1: h0, 2: h1},
            "block_access_stats": {
                block_id: {"last_access_time": 1.0, "access_count": 2}
                for block_id in range(3)
            },
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
            "background_copy_min_load_skew": 0.0,
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
            "block_access_stats": {
                0: {"last_access_time": 1.0, "access_count": 1},
            },
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

        request_queue.put({
            "type": "block_state",
            "rank": 0,
            "free_blocks": 4,
            "block_hashes": {0: prefix_hash},
            "evictable_block_hashes": {},
            "pinned_block_hashes": {0: prefix_hash},
            "block_access_stats": {
                0: {"last_access_time": 2.0, "access_count": 2},
            },
        })
        # Block-state updates are authoritative snapshots, not planner scan
        # triggers. The next ingress decision evaluates the changed hot state.
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


def test_ingress_forecast_flushes_hot_prefix_copy_before_future_requests():
    config, request_queue, response_queues, control_thread = _start_control_plane(
        extra_config={
            "enable_background_copy": True,
            "background_copy_max_blocks": 2,
            "background_copy_cooldown_s": 0.0,
            "background_copy_hot_threshold": 1,
            "background_copy_expected_reuses": 4.0,
            "background_copy_idle_pressure_threshold": 2.0,
            "block_size": 2,
            "foreground_prefill_token_time_ms": 1.0,
            "foreground_transfer_min_benefit_ratio": 1.0,
            "foreground_transfer_bandwidth_gib_s": 10.0,
            "foreground_transfer_fixed_latency_ms": 0.0,
            "foreground_transfer_interference_multiplier": 1.0,
        }
    )
    try:
        bm = BlockManager(num_blocks=8, block_size=2)
        worker0 = ControlPlaneClient(0, request_queue, response_queues[0])
        worker1 = ControlPlaneClient(1, request_queue, response_queues[1])
        ingress = ControlPlaneClient(-1, request_queue, response_queues[-1])
        executed = []

        def make_executor(rank):
            def _executor(plan):
                if plan["_phase"] == "execute":
                    executed.append((rank, plan))
                return {"success": True}

            return _executor

        worker0.set_rebalance_executor(make_executor(0))
        worker1.set_rebalance_executor(make_executor(1))
        h0 = bm.compute_hash([1, 2], -1)
        h1 = bm.compute_hash([3, 4], h0)
        request_queue.put({
            "type": "block_state",
            "rank": 0,
            "free_blocks": 4,
            "block_hashes": {0: h0, 1: h1},
            "block_parent_hashes": {0: -1, 1: h0},
            "block_access_stats": {
                0: {"last_access_time": 1.0, "access_count": 1},
                1: {"last_access_time": 1.0, "access_count": 1},
            },
        })
        request_queue.put({
            "type": "block_state",
            "rank": 1,
            "free_blocks": 4,
            "block_hashes": {},
        })

        stop = threading.Event()

        def pump_workers():
            while not stop.is_set():
                worker0.pump_async_messages()
                worker1.pump_async_messages()
                time.sleep(0.005)

        pump_thread = threading.Thread(target=pump_workers, daemon=True)
        pump_thread.start()
        try:
            result = ingress.flush_background_copies({h0: 4, h1: 4}, timeout_s=5)
        finally:
            stop.set()
            pump_thread.join(timeout=5)

        assert result["success"] is True
        assert result["placement_stats"]["dispatched"] == 1
        assert result["placement_stats"]["completed"] == 1
        source_plan = next(plan for rank, plan in executed if rank == 0)
        assert source_plan["background_trigger"] == "ingress_forecast"
        assert source_plan["estimated_future_reuses"] == 4
        assert source_plan["estimated_saved_prefill_ms"] == 4.0
        assert source_plan["transfers"][0]["hashes"] == [h0, h1]
    finally:
        _stop_control_plane(request_queue, control_thread)


def test_completed_forecast_copy_leases_future_routes_to_replica():
    config, request_queue, response_queues, control_thread = _start_control_plane(
        extra_config={
            "enable_background_copy": True,
            "background_copy_max_blocks": 2,
            "background_copy_cooldown_s": 0.0,
            "background_copy_expected_reuses": 4.0,
            "foreground_transfer_min_benefit_ratio": 0.0,
        }
    )
    try:
        bm = BlockManager(num_blocks=8, block_size=2)
        worker0 = ControlPlaneClient(0, request_queue, response_queues[0])
        worker1 = ControlPlaneClient(1, request_queue, response_queues[1])
        ingress = ControlPlaneClient(-1, request_queue, response_queues[-1])
        ingress.block_manager = bm
        worker0.set_rebalance_executor(lambda _plan: {"success": True})
        worker1.set_rebalance_executor(lambda _plan: {"success": True})

        h0 = bm.compute_hash([1, 2], -1)
        h1 = bm.compute_hash([3, 4], h0)
        request_queue.put({
            "type": "block_state",
            "rank": 0,
            "free_blocks": 4,
            "block_hashes": {0: h0, 1: h1},
            "block_parent_hashes": {0: -1, 1: h0},
            "block_access_stats": {
                0: {"last_access_time": 1.0, "access_count": 1},
                1: {"last_access_time": 1.0, "access_count": 1},
            },
        })
        request_queue.put({
            "type": "block_state",
            "rank": 1,
            "free_blocks": 4,
            "block_hashes": {},
        })

        stop = threading.Event()

        def pump_workers():
            while not stop.is_set():
                worker0.pump_async_messages()
                worker1.pump_async_messages()
                time.sleep(0.005)

        pump_thread = threading.Thread(target=pump_workers, daemon=True)
        pump_thread.start()
        try:
            result = ingress.flush_background_copies({h0: 4, h1: 4}, timeout_s=5)
        finally:
            stop.set()
            pump_thread.join(timeout=5)
        assert result["placement_stats"]["completed"] == 1

        # The test executor has no real local BlockManager, so publish the
        # destination snapshot that a data-plane commit normally sends.
        request_queue.put({
            "type": "block_state",
            "rank": 1,
            "free_blocks": 2,
            "block_hashes": {4: h0, 5: h1},
            "block_parent_hashes": {4: -1, 5: h0},
        })
        time.sleep(0.1)

        for _ in range(4):
            routed = ingress.route_sequence(
                Sequence(token_ids=[1, 2, 3, 4], block_size=2),
                return_meta=True,
            )
            assert routed["target_rank"] == 1
            assert routed["route_info"]["reason"] == "placement_lease"
            assert routed["route_info"]["matched_prefix_blocks"] == 2
    finally:
        _stop_control_plane(request_queue, control_thread)


def test_ingress_forecast_batches_prefixes_for_same_directed_pair():
    config, request_queue, response_queues, control_thread = _start_control_plane(
        extra_config={
            "enable_background_copy": True,
            "background_copy_max_blocks": 2,
            "background_copy_batch_max_candidates": 8,
            "background_copy_batch_max_blocks": 8,
            "background_copy_cooldown_s": 0.0,
            "foreground_transfer_min_benefit_ratio": 0.0,
        }
    )
    try:
        bm = BlockManager(num_blocks=8, block_size=2)
        worker0 = ControlPlaneClient(0, request_queue, response_queues[0])
        worker1 = ControlPlaneClient(1, request_queue, response_queues[1])
        ingress = ControlPlaneClient(-1, request_queue, response_queues[-1])
        executed = []

        def make_executor(rank):
            def _executor(plan):
                if plan["_phase"] == "execute":
                    executed.append((rank, plan))
                return {"success": True}

            return _executor

        worker0.set_rebalance_executor(make_executor(0))
        worker1.set_rebalance_executor(make_executor(1))
        h0 = bm.compute_hash([1, 2], -1)
        h1 = bm.compute_hash([3, 4], h0)
        h2 = bm.compute_hash([5, 6], -1)
        h3 = bm.compute_hash([7, 8], h2)
        request_queue.put({
            "type": "block_state",
            "rank": 0,
            "free_blocks": 4,
            "block_hashes": {0: h0, 1: h1, 2: h2, 3: h3},
            "block_parent_hashes": {0: -1, 1: h0, 2: -1, 3: h2},
            "block_access_stats": {
                block_id: {"last_access_time": 1.0, "access_count": 1}
                for block_id in range(4)
            },
        })
        request_queue.put({
            "type": "block_state",
            "rank": 1,
            "free_blocks": 8,
            "block_hashes": {},
        })

        stop = threading.Event()

        def pump_workers():
            while not stop.is_set():
                worker0.pump_async_messages()
                worker1.pump_async_messages()
                time.sleep(0.005)

        pump_thread = threading.Thread(target=pump_workers, daemon=True)
        pump_thread.start()
        try:
            result = ingress.flush_background_copies(
                {h0: 2, h1: 2, h2: 2, h3: 2},
                timeout_s=5,
            )
        finally:
            stop.set()
            pump_thread.join(timeout=5)

        source_plans = [plan for rank, plan in executed if rank == 0]
        assert len(source_plans) == 1
        assert set(source_plans[0]["transfers"][0]["hashes"]) == {h0, h1, h2, h3}
        assert len(source_plans[0]["background_candidate_keys"]) == 2
        assert result["placement_stats"]["dispatched"] == 2
        assert result["placement_stats"]["completed"] == 2
        assert result["placement_stats"]["plans_dispatched"] == 1
        assert result["placement_stats"]["plans_completed"] == 1
    finally:
        _stop_control_plane(request_queue, control_thread)


def test_background_copy_negative_cache_suppresses_identical_rejections():
    config, request_queue, response_queues, control_thread = _start_control_plane(
        extra_config={
            "enable_background_copy": True,
            "background_copy_max_blocks": 1,
            "background_copy_cooldown_s": 0.0,
            "background_copy_hot_threshold": 1,
            "background_copy_min_load_skew": 0.0,
            "foreground_transfer_min_benefit_ratio": 1000.0,
        }
    )
    try:
        bm = BlockManager(num_blocks=8, block_size=2)
        ingress = ControlPlaneClient(-1, request_queue, response_queues[-1])
        prefix_hash = bm.compute_hash([11, 22], -1)
        target_state = {
            "type": "block_state",
            "rank": 1,
            "free_blocks": 4,
            "block_hashes": {},
        }
        source_state = {
            "type": "block_state",
            "rank": 0,
            "free_blocks": 4,
            "block_hashes": {0: prefix_hash},
            "block_access_stats": {
                0: {"last_access_time": 1.0, "access_count": 2},
            },
        }
        request_queue.put(target_state)
        request_queue.put(source_state)

        result = None
        for _ in range(5):
            result = ingress.flush_background_copies({prefix_hash: 4}, timeout_s=5)

        stats = result["placement_stats"]
        pair_stats = result["placement_pair_stats"]["0-1"]
        assert stats["queued"] == 1
        assert stats["evaluated"] == 1
        assert stats["dropped_low_benefit"] == 1
        assert stats["skipped_negative_cache"] >= 4
        assert pair_stats["queued"] == 1
        assert pair_stats["dropped_low_benefit"] == 1
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
