import logging
import os
import sys
import time
from multiprocessing import Queue
from queue import Empty

import torch

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
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

    model_runner = ModelRunner(config, rank, gbm=None)
    control_plane_client = None
    if config.get("enable_global_pool", False) and config.get("use_control_plane_process", True):
        control_plane_client = ControlPlaneClient(rank, global_request_queue, global_response_queue)
    scheduler = Scheduler(
        max_num_sequences=config.get("max_num_sequences", 16),
        max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
        max_cached_blocks=config.get("max_cached_blocks", 1024),
        block_size=config.get("block_size", 256),
        eos=config.get("eos", 50256),
        global_scheduler=control_plane_client,
    )

    def execute_rebalance_plan(plan: dict):
        if plan is None:
            return {"success": False}
        transfers = plan.get("transfers", [])
        for transfer in transfers:
            src_gpu = transfer["src_gpu"]
            dst_gpu = transfer["dst_gpu"]
            src_blocks = transfer["src_blocks"]
            hashes = transfer.get("hashes", [-1] * len(src_blocks))

            if rank == src_gpu:
                model_runner.execute_swap_out(src_blocks, dst_gpu)
                scheduler.block_manager.release_blocks(src_blocks)
            elif rank == dst_gpu:
                local_target_blocks = scheduler.block_manager.reserve_free_blocks(len(src_blocks))
                model_runner.execute_swap_in(
                    src_gpu,
                    src_blocks,
                    local_target_blocks=local_target_blocks,
                )
                scheduler.block_manager.register_swap_in_blocks(local_target_blocks, hashes)

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
            scheduler.add_sequence(msg["seq"])
            return True
        return True

    def send_block_state():
        if control_plane_client is None:
            return
        control_plane_client.report_block_state(
            len(scheduler.block_manager.free_block_ids),
            scheduler.block_manager.get_local_block_hashes(),
        )

    send_block_state()
    idle_sent = False
    poll_timeout = float(config.get("worker_queue_poll_timeout", 0.01))
    heartbeat_interval = float(config.get("heartbeat_interval", 1.0))
    last_worker_heartbeat_sent = 0.0
    control_down_reported = False

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
        received_seq_ids = []
        try:
            if control_plane_client is not None:
                control_plane_client.pump_async_messages()
                maybe_send_worker_heartbeat()
            timeout = 0.05 if scheduler.is_finished() else poll_timeout
            msg = recv_queue.get(timeout=timeout)
            if not handle_message(msg):
                return
            if msg.get("type") == "sequence":
                received_seq_ids.append(msg["seq"].seq_id)

            while True:
                msg = recv_queue.get_nowait()
                if not handle_message(msg):
                    return
                if msg.get("type") == "sequence":
                    received_seq_ids.append(msg["seq"].seq_id)
        except Exception as e:
            if isinstance(e, Empty):
                pass
            else:
                logger.warning("rank %s recv error: %s", rank, e)

        if received_seq_ids:
            logger.info("rank %s received %s seqs: %s", rank, len(received_seq_ids), received_seq_ids)
            idle_sent = False

        if control_plane_client is not None:
            control_plane_client.pump_async_messages()
            maybe_send_worker_heartbeat()
            if not check_control_health() and not control_down_reported:
                logger.error("rank %s detected control plane timeout", rank)
                control_down_reported = True

        scheduled, is_prefill = scheduler.schedule()
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
            outputs = model_runner.run(local_seqs, is_prefill)
            scheduler.postprocess(local_seqs, outputs)
            send_block_state()

        finished = [
            (s.seq_id, _as_token_list(s.completion_token_ids))
            for s in scheduled
            if s.is_finished
        ]
        if finished:
            logger.info("rank %s finished seqs: %s", rank, [seq_id for seq_id, _ in finished])
            send_queue.put({"type": "finished", "data": finished})
