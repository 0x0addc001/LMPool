import logging
import queue
import time
import uuid
from collections import deque
from multiprocessing import Queue
from typing import Optional

import numpy as np
import xxhash

from lmpool.engine.global_block_manager import GlobalBlockManager
from lmpool.engine.global_scheduler import GlobalScheduler
from lmpool.engine.sequence import Sequence

logger = logging.getLogger(__name__)


class ControlPlaneClient:
    """
    Thin RPC client used by rank workers and the master ingress path.

    The client keeps no authoritative global state. It computes local prefix
    hashes because that requires the local BlockManager hash function, then asks
    the control plane process to make the route decision against the global page
    table.
    """

    def __init__(self, rank: int, request_queue: Queue, response_queue: Queue):
        self.rank = rank
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.block_manager = None
        self.gbm = None
        self.rebalance_executor = None
        self._stashed_messages = deque()
        self.heartbeat_interval = 1.0
        self.heartbeat_timeout = 3.0
        self._last_control_heartbeat = time.monotonic()
        self._last_worker_heartbeat = 0.0
        self._control_down_reported = False

    def route_sequence(self, seq: Sequence, return_meta: bool = False) -> int | dict:
        request_id = uuid.uuid4().hex
        prefix_hash = self._compute_prefix_hash(seq)
        self.request_queue.put({
            "type": "route_request",
            "request_id": request_id,
            "reply_rank": self.rank,
            "requester_rank": self.rank,
            "seq_id": seq.seq_id,
            "num_tokens": seq.num_tokens,
            "num_blocks": seq.num_blocks,
            "prefix_hash": prefix_hash,
        })
        while True:
            msg = self._next_response()
            if msg.get("request_id") != request_id:
                if self._handle_async_message(msg):
                    continue
                self._stashed_messages.append(msg)
                continue
            if msg.get("type") == "route_response":
                if return_meta:
                    return {
                        "target_rank": msg["target_rank"],
                        "route_info": msg.get("route_info", {}),
                    }
                return msg["target_rank"]
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("error", "control plane request failed"))

    def report_block_state(self, free_blocks: int, block_hashes: dict[int, int]):
        if self.rank < 0:
            return
        self.request_queue.put({
            "type": "block_state",
            "rank": self.rank,
            "free_blocks": free_blocks,
            "block_hashes": block_hashes,
        })

    def rebalance(self, gpu_id: int, needed_blocks: int) -> bool:
        request_id = uuid.uuid4().hex
        self.request_queue.put({
            "type": "rebalance_request",
            "request_id": request_id,
            "reply_rank": self.rank,
            "gpu_id": gpu_id,
            "needed_blocks": needed_blocks,
        })
        while True:
            msg = self._next_response()
            if msg.get("request_id") != request_id:
                if self._handle_async_message(msg):
                    continue
                self._stashed_messages.append(msg)
                continue
            if msg.get("type") == "rebalance_response":
                return bool(msg.get("success", False))
            if msg.get("type") == "error":
                logger.error(
                    "rank %s rebalance error for gpu=%s needed=%s: %s",
                    self.rank,
                    gpu_id,
                    needed_blocks,
                    msg.get("error", "unknown error"),
                )
                return False

    def close(self):
        if self.rank < 0:
            return
        self.request_queue.put({
            "type": "client_exit",
            "rank": self.rank,
        })

    def send_heartbeat(self):
        if self.rank < 0:
            return
        self._last_worker_heartbeat = time.monotonic()
        self.request_queue.put({
            "type": "heartbeat",
            "rank": self.rank,
            "ts": time.time(),
        })

    def note_control_heartbeat(self, ts: Optional[float] = None):
        self._last_control_heartbeat = time.monotonic()
        self._control_down_reported = False

    def check_control_health(self) -> bool:
        if self.rank < 0:
            return True
        if time.monotonic() - self._last_control_heartbeat <= self.heartbeat_timeout:
            return True
        if not self._control_down_reported:
            logger.error(
                "rank %s control heartbeat timeout: no heartbeat for %.2fs",
                self.rank,
                time.monotonic() - self._last_control_heartbeat,
            )
            self._control_down_reported = True
        return False

    def set_rebalance_executor(self, executor):
        self.rebalance_executor = executor

    def pump_async_messages(self) -> None:
        while True:
            try:
                msg = self.response_queue.get_nowait()
            except queue.Empty:
                break
            if msg.get("type") in {"route_response", "rebalance_response", "error"}:
                self._stashed_messages.appendleft(msg)
                break
            self._handle_async_message(msg)

    def _compute_prefix_hash(self, seq: Sequence) -> Optional[int]:
        full_blocks = int(seq.num_tokens // seq.block_size)
        if full_blocks == 0:
            return None
        if self.block_manager is None:
            hash_val = -1
            for i in range(full_blocks):
                block_tokens = seq.token_ids[i * seq.block_size : (i + 1) * seq.block_size]
                hasher = xxhash.xxh64()
                if hash_val != -1:
                    hasher.update(hash_val.to_bytes(8, "little"))
                if isinstance(block_tokens, tuple):
                    block_tokens = list(block_tokens)
                block_tokens = [t.item() if hasattr(t, "item") else t for t in block_tokens]
                hasher.update(np.array(block_tokens, dtype=np.int32).tobytes())
                hash_val = hasher.intdigest()
            return hash_val

        hash_val = -1
        for i in range(full_blocks):
            block_tokens = seq.token_ids[i * seq.block_size : (i + 1) * seq.block_size]
            hash_val = self.block_manager.compute_hash(block_tokens, hash_val)
        return hash_val

    def _next_response(self):
        if self._stashed_messages:
            return self._stashed_messages.popleft()
        return self.response_queue.get()

    def _handle_async_message(self, msg: dict) -> bool:
        if msg.get("type") == "heartbeat":
            self.note_control_heartbeat(msg.get("ts"))
            return True
        if msg.get("type") != "rebalance_execute":
            return False
        if self.rebalance_executor is None:
            raise RuntimeError("rebalance_executor is not installed on ControlPlaneClient")
        plan = msg["plan"]
        result = self.rebalance_executor(plan)
        self.request_queue.put({
            "type": "rebalance_done",
            "plan_id": msg["plan_id"],
            "rank": self.rank,
            "role": msg.get("role"),
            "result": result,
        })
        return True


def control_plane_process(config: dict, request_queue: Queue, response_queues: dict[int, Queue]):
    level_name = str(config.get("log_level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    world_size = config["world_size"]
    heartbeat_interval = float(config.get("heartbeat_interval", 1.0))
    heartbeat_timeout = float(config.get("heartbeat_timeout", 3.0))
    gbm = GlobalBlockManager(
        rank=-1,
        world_size=world_size,
        num_blocks_per_gpu=config["max_cached_blocks"],
        master_rank=-1,
        nvlink_pairs=config.get("nvlink_topo", {}).get("pairs"),
    )
    scheduler = GlobalScheduler(gbm=gbm, block_manager=None)
    logger.info("control plane process started")

    pending_rebalances: dict[str, dict] = {}
    worker_last_heartbeat: dict[int, float] = {rank: time.monotonic() for rank in range(world_size)}
    worker_down: set[int] = set()
    last_control_heartbeat_sent = 0.0

    def broadcast_control_heartbeat():
        now = time.time()
        for reply_rank, q in response_queues.items():
            if reply_rank < 0:
                continue
            q.put({
                "type": "heartbeat",
                "rank": -1,
                "ts": now,
            })

    def mark_worker_down(rank: int):
        if rank in worker_down:
            return
        worker_down.add(rank)
        logger.error("worker rank %s heartbeat timeout", rank)
        for plan_id, plan in list(pending_rebalances.items()):
            if rank not in plan["pending_ranks"]:
                continue
            plan["pending_ranks"].discard(rank)
            if not plan["pending_ranks"]:
                response_queues[plan["reply_rank"]].put({
                    "type": "rebalance_response",
                    "request_id": plan["request_id"],
                    "success": False,
                    "error": f"worker {rank} down",
                    "plan_id": plan_id,
                })
                del pending_rebalances[plan_id]

    def service_heartbeats(force: bool = False):
        nonlocal last_control_heartbeat_sent
        now = time.monotonic()
        if force or (now - last_control_heartbeat_sent) >= heartbeat_interval:
            broadcast_control_heartbeat()
            last_control_heartbeat_sent = now
        for worker_rank, last_seen in list(worker_last_heartbeat.items()):
            if now - last_seen > heartbeat_timeout:
                mark_worker_down(worker_rank)

    service_heartbeats(force=True)
    while True:
        try:
            msg = request_queue.get(timeout=0.1)
        except queue.Empty:
            service_heartbeats()
            continue

        msg_type = msg.get("type")
        if msg_type == "shutdown":
            logger.info("control plane process shutting down")
            return
        if msg_type == "client_exit":
            continue
        if msg_type == "heartbeat":
            worker_rank = msg["rank"]
            worker_last_heartbeat[worker_rank] = time.monotonic()
            if worker_rank in worker_down:
                worker_down.discard(worker_rank)
                logger.info("worker rank %s heartbeat recovered", worker_rank)
            service_heartbeats()
            continue
        if msg_type == "block_state":
            gbm.update_gpu_state(
                msg["rank"],
                msg["free_blocks"],
                msg["block_hashes"],
            )
            service_heartbeats()
            continue
        if msg_type == "rebalance_done":
            plan_id = msg["plan_id"]
            plan = pending_rebalances.get(plan_id)
            if plan is None:
                service_heartbeats()
                continue
            plan["pending_ranks"].discard(msg["rank"])
            if not plan["pending_ranks"]:
                response_queues[plan["reply_rank"]].put({
                    "type": "rebalance_response",
                    "request_id": plan["request_id"],
                    "success": True,
                    "plan_id": plan_id,
                })
                del pending_rebalances[plan_id]
            service_heartbeats()
            continue
        if msg_type == "route_request":
            try:
                target_rank = scheduler.route_sequence_meta(
                    requester_rank=msg["requester_rank"],
                    seq_id=msg["seq_id"],
                    num_tokens=msg["num_tokens"],
                    num_blocks=msg["num_blocks"],
                    prefix_hash=msg["prefix_hash"],
                    return_info=True,
                )
                if isinstance(target_rank, tuple):
                    target_rank, route_info = target_rank
                else:
                    route_info = {}
                gbm.reserve_blocks(target_rank, msg["num_blocks"])
                response_queues[msg["reply_rank"]].put({
                    "type": "route_response",
                    "request_id": msg["request_id"],
                    "target_rank": target_rank,
                    "route_info": route_info,
                })
            except Exception as exc:
                response_queues[msg["reply_rank"]].put({
                    "type": "error",
                    "request_id": msg["request_id"],
                    "error": str(exc),
                })
            service_heartbeats()
            continue
        if msg_type == "rebalance_request":
            try:
                plan = scheduler.plan_rebalance(
                    gpu_id=msg["gpu_id"],
                    needed_blocks=msg["needed_blocks"],
                )
                if plan is None:
                    response_queues[msg["reply_rank"]].put({
                        "type": "rebalance_response",
                        "request_id": msg["request_id"],
                        "success": False,
                    })
                    continue

                plan_id = uuid.uuid4().hex
                plan["plan_id"] = plan_id
                pending_rebalances[plan_id] = {
                    "request_id": msg["request_id"],
                    "reply_rank": msg["reply_rank"],
                    "pending_ranks": {
                        transfer["src_gpu"] for transfer in plan["transfers"]
                    } | {
                        transfer["dst_gpu"] for transfer in plan["transfers"]
                    },
                }

                source_ranks = {
                    transfer["src_gpu"] for transfer in plan["transfers"]
                }
                target_ranks = {
                    transfer["dst_gpu"] for transfer in plan["transfers"]
                }
                for src_rank in source_ranks:
                    response_queues[src_rank].put({
                        "type": "rebalance_execute",
                        "plan_id": plan_id,
                        "role": "source",
                        "plan": plan,
                    })
                for dst_rank in target_ranks:
                    if dst_rank in source_ranks:
                        continue
                    response_queues[dst_rank].put({
                        "type": "rebalance_execute",
                        "plan_id": plan_id,
                        "role": "target",
                        "plan": plan,
                    })
                # 立即继续，等待 rebalance_done
            except Exception as exc:
                response_queues[msg["reply_rank"]].put({
                    "type": "error",
                    "request_id": msg["request_id"],
                    "error": str(exc),
                })
            service_heartbeats()
            continue

        service_heartbeats()
        logger.warning("unknown control-plane message: %s", msg)
