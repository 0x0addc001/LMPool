import logging
import os
import random
import sys
import time
from multiprocessing import Queue
from queue import Empty

import torch
import numpy as np

from lmpool.engine.control_plane import ControlPlaneClient
from lmpool.engine.model_runner import ModelRunner
from lmpool.engine.scheduler import Scheduler

logger = logging.getLogger(__name__)


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
    control_plane_client = None
    if config.get("enable_global_pool", False) and config.get("use_control_plane_process", True):
        control_plane_client = ControlPlaneClient(
            rank,
            global_request_queue,
            global_response_queue,
            heartbeat_interval=float(config.get("heartbeat_interval", 1.0)),
            heartbeat_timeout=float(config.get("heartbeat_timeout", 3.0)),
        )
    scheduler = Scheduler(
        max_num_sequences=config.get("max_num_sequences", 16),
        max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
        max_cached_blocks=config.get("max_cached_blocks", 1024),
        block_size=config.get("block_size", 256),
        eos=config.get("eos", 50256),
        global_scheduler=control_plane_client,
    )
    scheduler.enable_foreground_rebalance = bool(config.get("enable_foreground_rebalance", True))
    scheduler.foreground_transfer_min_blocks = max(
        1,
        int(config.get("foreground_transfer_min_blocks", getattr(scheduler, "foreground_transfer_min_blocks", 1))),
    )
    scheduler._rebalance_cooldown_s = max(
        0.0,
        float(config.get("foreground_transfer_fail_cooldown_s", getattr(scheduler, "_rebalance_cooldown_s", 0.25))),
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
                for block_id in transfer["src_blocks"]
            ]
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
                if rank == transfer["src_gpu"] and transfer.get("mode", "move") != "copy"
                for block_id in transfer["src_blocks"]
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

        for transfer in transfers:
            src_gpu = transfer["src_gpu"]
            dst_gpu = transfer["dst_gpu"]
            src_blocks = transfer["src_blocks"]
            hashes = transfer.get("hashes", [-1] * len(src_blocks))
            mode = transfer.get("mode", plan.get("mode", "move"))

            if rank == src_gpu:
                model_runner.execute_swap_out(src_blocks, dst_gpu)
                if mode != "copy":
                    scheduler.block_manager.release_blocks(src_blocks)
                stats = {
                    "transfer_count": len(src_blocks),
                    "swap_count": len(src_blocks),
                    "transfer_copy_count": len(src_blocks) if mode == "copy" else 0,
                }
                if plan.get("background") and mode == "copy":
                    stats["background_copy_success"] = 1
                send_queue.put({
                    "type": "runtime_stats",
                    "rank": rank,
                    "data": stats,
                })
            elif rank == dst_gpu:
                prepared = prepared_rebalance_blocks.get(plan_id, {})
                local_target_blocks = prepared.get((src_gpu, tuple(src_blocks)))
                if local_target_blocks is None:
                    local_target_blocks = scheduler.block_manager.reserve_free_blocks(len(src_blocks))
                model_runner.execute_swap_in(
                    src_gpu,
                    src_blocks,
                    local_target_blocks=local_target_blocks,
                )
                scheduler.block_manager.register_swap_in_blocks(local_target_blocks, hashes)

        if plan_id is not None:
            prepared_rebalance_blocks.pop(plan_id, None)
        send_block_state()
        return {"success": True, "rank": rank}

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
        control_plane_client.report_block_state(
            len(scheduler.block_manager.free_block_ids),
            scheduler.block_manager.get_local_block_hashes(),
            scheduler.block_manager.get_evictable_block_hashes(),
            scheduler.block_manager.get_pinned_block_hashes(),
            len(scheduler.waiting),
            len(scheduler.running),
            sum(len(seq) for seq in scheduler.waiting),
            sum(len(seq) for seq in scheduler.running),
        )

    send_block_state()
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
            timeout = 0.05 if scheduler.is_finished() else poll_timeout
            msg = recv_queue.get(timeout=timeout)
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
            run_elapsed = time.perf_counter() - run_started
            scheduler.block_manager.mark_kv_ready(local_seqs)
            send_queue.put({
                "type": "runtime_stats",
                "rank": rank,
                "data": {
                    "prefill_tokens": run_tokens if is_prefill else 0,
                    "decode_tokens": 0 if is_prefill else run_tokens,
                    "prefill_time_s": run_elapsed if is_prefill else 0.0,
                    "decode_time_s": 0.0 if is_prefill else run_elapsed,
                    "first_tokens": sum(1 for seq in local_seqs if seq.num_completion_tokens == 0),
                },
            })
            scheduler.postprocess(local_seqs, outputs)
            first_tokens = [
                (seq.seq_id, _as_token_list([seq.completion_token_ids[-1]])[0])
                for seq in local_seqs
                if seq.num_completion_tokens == 1
            ]
            if first_tokens:
                send_queue.put({"type": "first_token", "rank": rank, "data": first_tokens})
            send_block_state()

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
