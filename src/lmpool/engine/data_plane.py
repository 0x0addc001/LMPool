import logging
import os
import random
import sys
import time
from multiprocessing import Queue
from multiprocessing.connection import wait as wait_for_connections
from queue import Empty

import torch
import numpy as np

from lmpool.engine.control_plane import ControlPlaneClient
from lmpool.engine.model_runner import ModelRunner
from lmpool.engine.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _wait_for_worker_events(
    recv_queue: Queue,
    control_queue,
    timeout: float,
) -> tuple[bool, bool] | None:
    """Wait for ingress or control work without polling one queue first.

    ``multiprocessing.Queue`` exposes its receive connection through ``_reader``.
    Keep a fallback for queue-compatible test doubles and alternate queue
    implementations that do not expose a connection.
    """
    recv_reader = getattr(recv_queue, "_reader", None)
    control_reader = getattr(control_queue, "_reader", None)
    if recv_reader is None or control_reader is None:
        return None
    ready = set(wait_for_connections([recv_reader, control_reader], timeout))
    return recv_reader in ready, control_reader in ready


def _as_token_list(tokens):
    return [t.item() if isinstance(t, torch.Tensor) else t for t in tokens]


def _configure_logging(config: dict):
    level_name = str(config.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def data_plane_process(
    config,
    rank,
    recv_queue: Queue,
    send_queue: Queue,
    global_request_queue: Queue = None,
    global_response_queue: Queue = None,
):
    """Per-rank data-plane process that initializes ModelRunner and enters loop."""
    _configure_logging(config)
    seed = int(config.get("random_seed", 0)) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

    model_runner = ModelRunner(config, rank, gbm=None)
    prewarm_observations = []
    if config.get("enable_kv_transfer_prewarm", True):
        from lmpool.engine.kv_transfer import prewarm_p2p_pairs

        prewarm_observations = prewarm_p2p_pairs(
            config.get("nvlink_topo", {}).get("pairs") or [],
            num_layers=int(config.get("num_layers", 1)),
            block_size=int(config.get("block_size", 256)),
            num_kv_heads=int(config.get("num_kv_heads", 1)),
            head_dim=int(config.get("head_dim", config.get("hidden_size", 1) // config.get("num_heads", 1))),
            num_blocks=max(1, int(config.get("kv_transfer_prewarm_blocks", 2))),
        )
    control_plane_client = None
    if config.get("enable_global_pool", False) and config.get("use_control_plane_process", True):
        control_plane_client = ControlPlaneClient(
            rank,
            global_request_queue,
            global_response_queue,
            heartbeat_interval=float(config.get("heartbeat_interval", 1.0)),
            heartbeat_timeout=float(config.get("heartbeat_timeout", 3.0)),
        )
        for observation in prewarm_observations:
            control_plane_client.report_transfer_observation(**observation)
    scheduler = Scheduler(
        max_num_sequences=config.get("max_num_sequences", 16),
        max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
        max_cached_blocks=config.get("max_cached_blocks", 1024),
        block_size=config.get("block_size", 256),
        eos=config.get("eos", 50256),
        global_scheduler=control_plane_client,
    )
    scheduler.enable_foreground_rebalance = bool(config.get("enable_foreground_rebalance", True))
    scheduler.preserve_cache_via_transfer = bool(config.get("preserve_cache_via_transfer", False))
    scheduler.foreground_transfer_min_blocks = max(
        1,
        int(config.get("foreground_transfer_min_blocks", getattr(scheduler, "foreground_transfer_min_blocks", 1))),
    )
    scheduler._rebalance_cooldown_s = max(
        0.0,
        float(config.get("foreground_transfer_fail_cooldown_s", getattr(scheduler, "_rebalance_cooldown_s", 0.25))),
    )
    scheduler._rebalance_cooldown_max_s = max(
        scheduler._rebalance_cooldown_s,
        float(config.get(
            "foreground_transfer_fail_cooldown_max_s",
            getattr(scheduler, "_rebalance_cooldown_max_s", 30.0),
        )),
    )
    prepared_rebalance_blocks: dict[str, dict[tuple[int, tuple[int, ...]], list[int]]] = {}

    def execute_rebalance_plan(plan: dict):
        if plan is None:
            return {"success": False}
        phase = plan.get("_phase", "execute")
        plan_id = plan.get("plan_id")
        transfers = plan.get("transfers", [])

        if phase == "abort":
            prepared = prepared_rebalance_blocks.pop(plan_id, {})
            for local_target_blocks in prepared.values():
                scheduler.block_manager.release_reserved_blocks(local_target_blocks)
            send_block_state()
            return {"success": True, "rank": rank}

        if phase == "prepare":
            prepared: dict[tuple[int, tuple[int, ...]], list[int]] = {}
            is_background = bool(plan.get("background"))
            all_source_blocks = [
                block_id
                for transfer in transfers
                if rank == transfer["src_gpu"]
                for block_id in (
                    list(transfer["src_blocks"])
                    + list(transfer.get("release_source_blocks", []))
                )
            ]
            all_source_blocks = list(dict.fromkeys(all_source_blocks))
            stale_source_blocks = [
                block_id
                for block_id in all_source_blocks
                if block_id not in scheduler.block_manager.used_block_ids
            ]
            if stale_source_blocks:
                send_block_state()
                return {
                    "success": False,
                    "rank": rank,
                    "reason": "stale_source",
                    "error": f"source blocks are no longer allocated: {stale_source_blocks}",
                }
            source_blocks = [
                block_id
                for transfer in transfers
                if rank == transfer["src_gpu"]
                for block_id in transfer.get(
                    "release_source_blocks",
                    [] if transfer.get("mode", "move") == "copy" else transfer["src_blocks"],
                )
            ]
            pinned_blocks = [
                block_id
                for block_id in source_blocks
                if scheduler.block_manager.blocks[block_id].ref_count != 0
            ]
            if pinned_blocks:
                if is_background:
                    send_queue.put({
                        "type": "runtime_stats",
                        "rank": rank,
                        "data": {
                            "background_copy_fail": 1,
                            "background_copy_fail_reasons": {"pinned_source": 1},
                        },
                    })
                send_block_state()
                return {
                    "success": False,
                    "rank": rank,
                    "reason": "pinned_source",
                    "error": f"source blocks still referenced: {pinned_blocks}",
                }

            needed = sum(
                len(transfer["src_blocks"])
                for transfer in transfers
                if rank == transfer["dst_gpu"]
            )
            if len(scheduler.block_manager.free_block_ids) < needed:
                if is_background:
                    send_queue.put({
                        "type": "runtime_stats",
                        "rank": rank,
                        "data": {
                            "background_copy_fail": 1,
                            "background_copy_fail_reasons": {"no_target_space": 1},
                        },
                    })
                send_block_state()
                return {
                    "success": False,
                    "rank": rank,
                    "reason": "no_target_space",
                    "error": (
                        f"not enough free blocks to reserve: need {needed}, "
                        f"have {len(scheduler.block_manager.free_block_ids)}"
                    ),
                }

            for transfer in transfers:
                if rank != transfer["dst_gpu"]:
                    continue
                src_gpu = transfer["src_gpu"]
                src_blocks = transfer["src_blocks"]
                local_target_blocks = scheduler.block_manager.reserve_free_blocks(len(src_blocks))
                prepared[(src_gpu, tuple(src_blocks))] = local_target_blocks

            if plan_id is not None:
                prepared_rebalance_blocks[plan_id] = prepared
            send_block_state()
            return {"success": True, "rank": rank}

        source_transfers = [
            transfer for transfer in transfers if rank == transfer["src_gpu"]
        ]
        execution_stats = {}
        if source_transfers:
            transfer_started = time.perf_counter()
            transfer_observations = []
            for transfer in source_transfers:
                operation_started = time.perf_counter()
                model_runner.execute_swap_out(
                    transfer["src_blocks"],
                    transfer["dst_gpu"],
                )
                operation_time_s = time.perf_counter() - operation_started
                transfer_observations.append({
                    "src_gpu": int(transfer["src_gpu"]),
                    "dst_gpu": int(transfer["dst_gpu"]),
                    "transfer_bytes": model_runner.kv_transfer_bytes(
                        len(transfer["src_blocks"])
                    ),
                    "transfer_time_s": operation_time_s,
                })
            transfer_time_s = time.perf_counter() - transfer_started
            sent_blocks = {
                block_id
                for transfer in source_transfers
                for block_id in transfer["src_blocks"]
            }
            release_source_blocks = list(dict.fromkeys(
                block_id
                for transfer in source_transfers
                for block_id in transfer.get(
                    "release_source_blocks",
                    [] if transfer.get("mode", plan.get("mode", "move")) == "copy"
                    else transfer["src_blocks"],
                )
            ))
            if release_source_blocks:
                scheduler.block_manager.release_blocks(release_source_blocks)
            stats = {
                "transfer_count": len(sent_blocks),
                "swap_count": len(sent_blocks),
                "transfer_hashes": list(dict.fromkeys(
                    block_hash
                    for transfer in source_transfers
                    for block_hash in transfer.get("hashes", [])
                    if block_hash != -1
                )),
                "transfer_copy_count": len(sent_blocks - set(release_source_blocks)),
                "transfer_release_count": len(release_source_blocks),
                "chain_transfer_count": sum(
                    transfer.get("mode", plan.get("mode", "move")) == "chain_move"
                    for transfer in source_transfers
                ),
                "transfer_bytes": sum(
                    model_runner.kv_transfer_bytes(len(transfer["src_blocks"]))
                    for transfer in source_transfers
                ),
                "transfer_time_s": transfer_time_s,
                "transfer_source_time_s": transfer_time_s,
                "estimated_transfer_cost_ms": float(
                    plan.get("estimated_transfer_cost_ms", 0.0)
                ),
                "estimated_saved_prefill_ms": float(
                    plan.get("estimated_saved_prefill_ms", 0.0)
                ),
            }
            execution_stats = {
                "transfer_bytes": stats["transfer_bytes"],
                "transfer_time_s": stats["transfer_time_s"],
                "transfer_observations": transfer_observations,
            }
            if plan.get("background") and all(
                transfer.get("mode", plan.get("mode", "move")) == "copy"
                for transfer in source_transfers
            ):
                stats["background_copy_success"] = 1
            send_queue.put({
                "type": "runtime_stats",
                "rank": rank,
                "data": stats,
            })

        for transfer in transfers:
            src_gpu = transfer["src_gpu"]
            dst_gpu = transfer["dst_gpu"]
            src_blocks = transfer["src_blocks"]
            hashes = transfer.get("hashes", [-1] * len(src_blocks))
            parent_hashes = transfer.get("parent_hashes")
            access_counts = transfer.get("access_counts")
            mode = transfer.get("mode", plan.get("mode", "move"))
            if rank == dst_gpu:
                prepared = prepared_rebalance_blocks.get(plan_id, {})
                local_target_blocks = prepared.get((src_gpu, tuple(src_blocks)))
                if local_target_blocks is None:
                    local_target_blocks = scheduler.block_manager.reserve_free_blocks(len(src_blocks))
                target_started = time.perf_counter()
                model_runner.execute_swap_in(
                    src_gpu,
                    src_blocks,
                    local_target_blocks=local_target_blocks,
                )
                scheduler.block_manager.register_swap_in_blocks(
                    local_target_blocks,
                    hashes,
                    parent_hashes=parent_hashes,
                    access_counts=access_counts,
                )
                send_queue.put({
                    "type": "runtime_stats",
                    "rank": rank,
                    "data": {
                        "transfer_target_time_s": time.perf_counter() - target_started,
                    },
                })

        if plan_id is not None:
            prepared_rebalance_blocks.pop(plan_id, None)
        send_block_state()
        return {"success": True, "rank": rank, **execution_stats}

    if control_plane_client is not None:
        control_plane_client.set_rebalance_executor(execute_rebalance_plan)

    def handle_message(msg) -> bool:
        if msg.get("type") == "exit":
            if control_plane_client is not None:
                control_plane_client.close()
            model_runner.exit()
            return False
        if msg.get("type") == "sequence":
            seq = msg["seq"]
            if seq.remote_gpu_id == rank:
                seq.remote_gpu_id = -1
            scheduler.add_sequence(seq)
            return True
        return True

    def send_block_state():
        if control_plane_client is None:
            return
        decode_weight = max(0.0, float(config.get("route_decode_token_weight", 8.0)))

        def estimated_work_tokens(seq):
            remaining_decode = max(0, int(seq.max_tokens) - int(seq.num_completion_tokens))
            return int(len(seq) + decode_weight * remaining_decode)

        control_plane_client.report_block_state(
            len(scheduler.block_manager.free_block_ids),
            scheduler.block_manager.get_local_block_hashes(),
            scheduler.block_manager.get_evictable_block_hashes(),
            scheduler.block_manager.get_pinned_block_hashes(),
            len(scheduler.waiting),
            len(scheduler.running),
            sum(estimated_work_tokens(seq) for seq in scheduler.waiting),
            sum(estimated_work_tokens(seq) for seq in scheduler.running),
            scheduler.block_manager.get_local_block_parent_hashes(),
            scheduler.block_manager.get_local_block_access_stats(),
        )

    send_block_state()
    send_queue.put({
        "type": "runtime_stats",
        "rank": rank,
        "data": {"max_cached_blocks": int(config["max_cached_blocks"])},
    })
    idle_sent = False
    poll_timeout = float(config.get("worker_queue_poll_timeout", 0.01))
    heartbeat_interval = float(config.get("heartbeat_interval", 1.0))
    last_worker_heartbeat_sent = 0.0
    control_down_reported = False
    last_rebalance_success_count = 0
    last_rebalance_fail_count = 0
    last_rebalance_fail_reasons: dict[str, int] = {}
    last_preemption_count = 0

    def maybe_send_worker_heartbeat():
        nonlocal last_worker_heartbeat_sent
        if control_plane_client is None:
            return
        now = time.monotonic()
        if now - last_worker_heartbeat_sent >= heartbeat_interval:
            control_plane_client.send_heartbeat()
            last_worker_heartbeat_sent = now

    def check_control_health():
        if control_plane_client is None:
            return True
        return control_plane_client.check_control_health()

    while True:
        received_sequences = []
        try:
            if control_plane_client is not None:
                control_plane_client.pump_async_messages()
                maybe_send_worker_heartbeat()
            events = _wait_for_worker_events(
                recv_queue,
                global_response_queue if control_plane_client is not None else None,
                poll_timeout,
            )
            ingress_ready = True
            if events is not None:
                ingress_ready, control_ready = events
                if control_ready:
                    control_plane_client.pump_async_messages()
            if ingress_ready:
                msg = recv_queue.get(timeout=poll_timeout if events is None else 0)
                if not handle_message(msg):
                    return
                if msg.get("type") == "sequence":
                    received_sequences.append(msg["seq"])

                while True:
                    msg = recv_queue.get_nowait()
                    if not handle_message(msg):
                        return
                    if msg.get("type") == "sequence":
                        received_sequences.append(msg["seq"])
        except Exception as e:
            if isinstance(e, Empty):
                pass
            else:
                logger.warning("rank %s recv error: %s", rank, e)

        if received_sequences:
            # Publish the admitted waiting load before clearing optimistic
            # route reservations. Messages from this worker share one FIFO
            # queue, so the control plane never observes an idle snapshot in
            # between admission and acknowledgement.
            send_block_state()
            if control_plane_client is not None:
                for seq in received_sequences:
                    control_plane_client.acknowledge_route_admission(seq.seq_id, seq.num_tokens)
            received_seq_ids = [seq.seq_id for seq in received_sequences]
            logger.info("rank %s received %s seqs: %s", rank, len(received_seq_ids), received_seq_ids)
            idle_sent = False

        if control_plane_client is not None:
            control_plane_client.pump_async_messages()
            maybe_send_worker_heartbeat()
            if not check_control_health() and not control_down_reported:
                logger.error("rank %s detected control plane timeout", rank)
                control_down_reported = True

        scheduled, is_prefill = scheduler.schedule()
        scheduler_preemptions = int(getattr(scheduler, "preemption_count", 0))
        preemption_delta = scheduler_preemptions - last_preemption_count
        if preemption_delta:
            send_queue.put({
                "type": "runtime_stats",
                "rank": rank,
                "data": {"preemption_count": preemption_delta},
            })
            last_preemption_count = scheduler_preemptions
        if control_plane_client is not None:
            success_delta = control_plane_client.rebalance_success_count - last_rebalance_success_count
            fail_delta = control_plane_client.rebalance_fail_count - last_rebalance_fail_count
            reason_deltas = {}
            for reason, count in control_plane_client.rebalance_fail_reasons.items():
                delta = count - last_rebalance_fail_reasons.get(reason, 0)
                if delta:
                    reason_deltas[reason] = delta
            if success_delta or fail_delta or reason_deltas:
                send_queue.put({
                    "type": "runtime_stats",
                    "rank": rank,
                    "data": {
                        "rebalance_success": success_delta,
                        "rebalance_fail": fail_delta,
                        "rebalance_fail_reasons": reason_deltas,
                    },
                })
                last_rebalance_success_count = control_plane_client.rebalance_success_count
                last_rebalance_fail_count = control_plane_client.rebalance_fail_count
                last_rebalance_fail_reasons = dict(control_plane_client.rebalance_fail_reasons)
        if not scheduled:
            if scheduler.is_finished() and recv_queue.empty() and not idle_sent:
                try:
                    send_queue.put_nowait({"type": "idle", "rank": rank})
                    idle_sent = True
                except Exception as e:
                    logger.debug("rank %s failed to send idle: %s", rank, e)
            continue
        idle_sent = False

        local_rank = rank
        local_seqs = [s for s in scheduled if s.remote_gpu_id in (-1, local_rank)]
        remote_seqs = [s for s in scheduled if s.remote_gpu_id not in (-1, local_rank)]

        for seq in remote_seqs:
            send_queue.put({"type": "sequence", "target": seq.remote_gpu_id, "seq": seq})

        if local_seqs:
            run_tokens = sum(len(seq) for seq in local_seqs) if is_prefill else len(local_seqs)
            prefill_cached_tokens = (
                sum(min(len(seq), int(seq.num_cached_tokens)) for seq in local_seqs)
                if is_prefill
                else 0
            )
            prefill_uncached_tokens = (
                max(0, run_tokens - prefill_cached_tokens) if is_prefill else 0
            )
            first_token_count = sum(
                1 for seq in local_seqs if seq.num_completion_tokens == 0
            )
            committed_route_seq_ids = []
            if is_prefill:
                missing_blocks = [
                    (seq.seq_id, seq.remote_gpu_id, seq.num_blocks, list(seq.block_table))
                    for seq in local_seqs
                    if len(seq.block_table) < seq.num_blocks
                ]
                if missing_blocks:
                    raise RuntimeError(
                        f"rank {rank} scheduled local prefill without enough blocks: "
                        f"{missing_blocks}"
                    )
                prefill_stats = []
                for seq in local_seqs:
                    is_initial_prefill = seq.prefill_attempts == 0
                    if is_initial_prefill:
                        committed_route_seq_ids.append(seq.seq_id)
                    seq.prefill_attempts += 1
                    prefill_stats.append({
                        "seq_id": seq.seq_id,
                        "prefix_hit": seq.num_cached_tokens > 0,
                        "num_cached_tokens": seq.num_cached_tokens,
                        "num_prompt_tokens": seq.num_prompt_tokens,
                        "prefill_attempt": seq.prefill_attempts,
                        "is_initial_prefill": is_initial_prefill,
                        "preemption_count": seq.preemption_count,
                    })
                if prefill_stats:
                    send_queue.put({"type": "prefill_stats", "rank": rank, "data": prefill_stats})
            run_started = time.perf_counter()
            outputs = model_runner.run(local_seqs, is_prefill)
            scheduler.block_manager.mark_kv_ready(local_seqs)
            scheduler.postprocess(local_seqs, outputs)
            # postprocess consumes sampled CUDA values, so this wall time also
            # includes completion synchronization rather than only enqueue time.
            run_elapsed = time.perf_counter() - run_started
            send_queue.put({
                "type": "runtime_stats",
                "rank": rank,
                "data": {
                    "prefill_tokens": run_tokens if is_prefill else 0,
                    "prefill_prompt_tokens": run_tokens if is_prefill else 0,
                    "prefill_cached_tokens": prefill_cached_tokens,
                    "prefill_uncached_tokens": prefill_uncached_tokens,
                    "decode_tokens": 0 if is_prefill else run_tokens,
                    "prefill_time_s": run_elapsed if is_prefill else 0.0,
                    "decode_time_s": 0.0 if is_prefill else run_elapsed,
                    "first_tokens": first_token_count,
                },
            })
            if is_prefill and control_plane_client is not None:
                control_plane_client.report_prefill_observation(
                    prefill_uncached_tokens,
                    run_elapsed,
                )
            first_tokens = [
                (seq.seq_id, _as_token_list([seq.completion_token_ids[-1]])[0])
                for seq in local_seqs
                if seq.num_completion_tokens == 1
            ]
            if first_tokens:
                send_queue.put({"type": "first_token", "rank": rank, "data": first_tokens})
            send_block_state()
            if control_plane_client is not None and committed_route_seq_ids:
                control_plane_client.acknowledge_route_blocks(committed_route_seq_ids)

        finished = [
            (s.seq_id, _as_token_list(s.completion_token_ids))
            for s in scheduled
            if s.is_finished
        ]
        if finished:
            logger.info("rank %s finished seqs: %s", rank, [seq_id for seq_id, _ in finished])
            send_queue.put({
                "type": "runtime_stats",
                "rank": rank,
                "data": {
                    "finished": len(finished),
                    "output_tokens": sum(len(tokens) for _, tokens in finished),
                },
            })
            send_queue.put({"type": "finished", "data": finished})
