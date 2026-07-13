import logging
import queue
import time
import uuid
from collections import defaultdict, deque
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

    def __init__(
        self,
        rank: int,
        request_queue: Queue,
        response_queue: Queue,
        heartbeat_interval: float = 1.0,
        heartbeat_timeout: float = 3.0,
    ):
        self.rank = rank
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.block_manager = None
        self.gbm = None
        self.rebalance_executor = None
        self._stashed_messages = deque()
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self._last_control_heartbeat = time.monotonic()
        self._last_worker_heartbeat = 0.0
        self._control_down_reported = False
        self.rebalance_success_count = 0
        self.rebalance_fail_count = 0
        self.rebalance_fail_reasons = defaultdict(int)

    def route_sequence(self, seq: Sequence, return_meta: bool = False) -> int | dict:
        request_id = uuid.uuid4().hex
        prefix_hashes = self._compute_prefix_hashes(seq)
        prefix_hash = prefix_hashes[-1] if prefix_hashes else None
        self.request_queue.put({
            "type": "route_request",
            "request_id": request_id,
            "reply_rank": self.rank,
            "requester_rank": self.rank,
            "seq_id": seq.seq_id,
            "num_tokens": seq.num_tokens,
            "num_blocks": seq.num_blocks,
            "prefix_hash": prefix_hash,
            "prefix_hashes": prefix_hashes,
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

    def report_block_state(
        self,
        free_blocks: int,
        block_hashes: dict[int, int],
        evictable_block_hashes: dict[int, int] | None = None,
        pinned_block_hashes: dict[int, int] | None = None,
        waiting_sequences: int = 0,
        running_sequences: int = 0,
        waiting_tokens: int = 0,
        running_tokens: int = 0,
    ):
        if self.rank < 0:
            return
        self.request_queue.put({
            "type": "block_state",
            "rank": self.rank,
            "free_blocks": free_blocks,
            "block_hashes": block_hashes,
            "evictable_block_hashes": evictable_block_hashes,
            "pinned_block_hashes": pinned_block_hashes,
            "waiting_sequences": waiting_sequences,
            "running_sequences": running_sequences,
            "waiting_tokens": waiting_tokens,
            "running_tokens": running_tokens,
        })

    def rebalance(self, gpu_id: int, needed_blocks: int, allow_copy: bool = False) -> bool:
        request_id = uuid.uuid4().hex
        self.request_queue.put({
            "type": "rebalance_request",
            "request_id": request_id,
            "reply_rank": self.rank,
            "gpu_id": gpu_id,
            "needed_blocks": needed_blocks,
            "allow_copy": allow_copy,
        })
        while True:
            msg = self._next_response()
            if msg.get("request_id") != request_id:
                if self._handle_async_message(msg):
                    continue
                self._stashed_messages.append(msg)
                continue
            if msg.get("type") == "rebalance_response":
                success = bool(msg.get("success", False))
                if success:
                    self.rebalance_success_count += 1
                else:
                    self.rebalance_fail_count += 1
                    self.rebalance_fail_reasons[msg.get("reason", "unknown")] += 1
                return success
            if msg.get("type") == "error":
                logger.error(
                    "rank %s rebalance error for gpu=%s needed=%s: %s",
                    self.rank,
                    gpu_id,
                    needed_blocks,
                    msg.get("error", "unknown error"),
                )
                self.rebalance_fail_count += 1
                self.rebalance_fail_reasons["error"] += 1
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
        prefix_hashes = self._compute_prefix_hashes(seq)
        return prefix_hashes[-1] if prefix_hashes else None

    def _compute_prefix_hashes(self, seq: Sequence) -> list[int]:
        full_blocks = int(seq.num_tokens // seq.block_size)
        if full_blocks == 0:
            return []
        hashes = []
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
                hashes.append(hash_val)
            return hashes

        hash_val = -1
        for i in range(full_blocks):
            block_tokens = seq.token_ids[i * seq.block_size : (i + 1) * seq.block_size]
            hash_val = self.block_manager.compute_hash(block_tokens, hash_val)
            hashes.append(hash_val)
        return hashes

    def _next_response(self):
        if self._stashed_messages:
            return self._stashed_messages.popleft()
        return self.response_queue.get()

    def _handle_async_message(self, msg: dict) -> bool:
        if msg.get("type") == "heartbeat":
            self.note_control_heartbeat(msg.get("ts"))
            return True
        if msg.get("type") not in {"rebalance_prepare", "rebalance_execute", "rebalance_abort"}:
            return False
        if self.rebalance_executor is None:
            raise RuntimeError("rebalance_executor is not installed on ControlPlaneClient")
        plan = dict(msg["plan"])
        if msg.get("type") == "rebalance_prepare":
            phase = "prepare"
        elif msg.get("type") == "rebalance_abort":
            phase = "abort"
        else:
            phase = "execute"
        plan["_phase"] = phase
        result = self.rebalance_executor(plan)
        self.request_queue.put({
            "type": "rebalance_done",
            "plan_id": msg["plan_id"],
            "rank": self.rank,
            "role": msg.get("role"),
            "phase": phase,
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
    scheduler.prefix_hit_weight = float(config.get("route_prefix_hit_weight", scheduler.prefix_hit_weight))
    scheduler.queue_pressure_weight = float(config.get("route_queue_pressure_weight", scheduler.queue_pressure_weight))
    scheduler.free_block_weight = float(config.get("route_free_block_weight", scheduler.free_block_weight))
    scheduler.load_weight = float(config.get("route_load_weight", scheduler.load_weight))
    scheduler.waiting_token_weight = float(config.get("route_waiting_token_weight", scheduler.waiting_token_weight))
    scheduler.running_token_weight = float(config.get("route_running_token_weight", scheduler.running_token_weight))
    scheduler.running_sequence_weight = float(config.get("route_running_sequence_weight", scheduler.running_sequence_weight))
    scheduler.load_bypass_threshold = float(config.get("route_load_bypass_threshold", scheduler.load_bypass_threshold))
    route_cache_queue_slack = float(config.get("route_cache_queue_slack", 2.0))
    enable_background_copy = bool(config.get("enable_background_copy", False))
    background_copy_max_blocks = max(1, int(config.get("background_copy_max_blocks", 1)))
    background_copy_cooldown_s = max(0.0, float(config.get("background_copy_cooldown_s", 2.0)))
    background_copy_hot_threshold = max(1, int(config.get("background_copy_hot_threshold", 3)))
    logger.info("control plane process started")

    pending_rebalances: dict[str, dict] = {}
    route_cache: dict[int, int] = {}
    background_copy_inflight_pairs: set[tuple[int, int]] = set()
    background_copy_recent: dict[tuple[int, int, int], float] = {}
    prefix_route_hits: dict[int, int] = defaultdict(int)
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
                if plan.get("reply_rank") is not None:
                    response_queues[plan["reply_rank"]].put({
                        "type": "rebalance_response",
                        "request_id": plan["request_id"],
                        "success": False,
                        "error": f"worker {rank} down",
                        "plan_id": plan_id,
                    })
                _release_background_copy_inflight(plan["plan"])
                del pending_rebalances[plan_id]

    def _send_rebalance_execute(plan_id: str, plan: dict):
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

    def _send_rebalance_abort(plan_id: str, plan: dict, ranks: set[int]):
        for rank in ranks:
            response_queues[rank].put({
                "type": "rebalance_abort",
                "plan_id": plan_id,
                "role": "abort",
                "plan": plan,
            })

    def _release_background_copy_inflight(plan: dict) -> None:
        if not plan.get("background"):
            return
        for transfer in plan.get("transfers", []):
            background_copy_inflight_pairs.discard((
                int(transfer["src_gpu"]),
                int(transfer["dst_gpu"]),
            ))

    def _enqueue_rebalance_plan(plan: dict, request_id: str | None, reply_rank: int | None) -> bool:
        plan_id = uuid.uuid4().hex
        plan["plan_id"] = plan_id
        source_ranks = {
            transfer["src_gpu"] for transfer in plan["transfers"]
        }
        target_ranks = {
            transfer["dst_gpu"] for transfer in plan["transfers"]
        }
        execute_ranks = source_ranks | target_ranks
        if not target_ranks:
            return False

        pending_rebalances[plan_id] = {
            "request_id": request_id,
            "reply_rank": reply_rank,
            "phase": "prepare",
            "pending_ranks": set(execute_ranks),
            "execute_ranks": execute_ranks,
            "plan": plan,
        }

        for prepare_rank in execute_ranks:
            role = "source" if prepare_rank in source_ranks else "target"
            response_queues[prepare_rank].put({
                "type": "rebalance_prepare",
                "plan_id": plan_id,
                "role": role,
                "plan": plan,
            })
        return True

    def _maybe_schedule_background_copy(route_info: dict) -> None:
        if not enable_background_copy or not route_info.get("prefix_hit"):
            return
        prefix_hash = route_info.get("prefix_hash")
        prefix_hashes = route_info.get("prefix_hashes") or ([prefix_hash] if prefix_hash is not None else [])
        hit_summary = route_info.get("hit_summary") or {}
        if prefix_hash is None or not hit_summary:
            return
        prefix_route_hits[int(prefix_hash)] += 1
        if prefix_route_hits[int(prefix_hash)] < background_copy_hot_threshold:
            return

        target_rank = route_info.get("target_rank")
        src_gpu = target_rank if target_rank in hit_summary else next(iter(hit_summary))
        dst_gpu = gbm._get_nvlink_partner(src_gpu)
        if dst_gpu is None or dst_gpu == src_gpu:
            return
        pair_key = (int(src_gpu), int(dst_gpu))
        if pair_key in background_copy_inflight_pairs:
            return
        if gbm.get_free_blocks_count(dst_gpu) <= 0:
            return

        cooldown_key = (int(prefix_hash), int(src_gpu), int(dst_gpu))
        now = time.monotonic()
        if now - background_copy_recent.get(cooldown_key, 0.0) < background_copy_cooldown_s:
            return

        src_blocks = []
        hashes = []
        for block_hash in prefix_hashes:
            if len(src_blocks) >= background_copy_max_blocks:
                break
            locations = [
                loc for loc in gbm.get_block_location(block_hash)
                if loc.gpu_id == src_gpu
            ]
            if not locations:
                continue
            if any(loc.gpu_id == dst_gpu for loc in gbm.get_block_location(block_hash)):
                continue
            block_id = locations[0].block_id
            src_blocks.append(block_id)
            hashes.append(block_hash)

        if not src_blocks or gbm.get_free_blocks_count(dst_gpu) < len(src_blocks):
            return

        plan = {
            "gpu_id": src_gpu,
            "needed_blocks": len(src_blocks),
            "mode": "copy",
            "background": True,
            "transfers": [{
                "src_gpu": src_gpu,
                "dst_gpu": dst_gpu,
                "src_blocks": src_blocks,
                "hashes": hashes,
                "mode": "copy",
            }],
        }
        if _enqueue_rebalance_plan(plan, request_id=None, reply_rank=None):
            background_copy_inflight_pairs.add(pair_key)
            background_copy_recent[cooldown_key] = now
            logger.info(
                "background transfer copy scheduled: prefix=%s src=%s dst=%s blocks=%s",
                prefix_hash,
                src_gpu,
                dst_gpu,
                src_blocks,
            )

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
                msg.get("evictable_block_hashes"),
                msg.get("pinned_block_hashes"),
                msg.get("waiting_sequences", 0),
                msg.get("running_sequences", 0),
                msg.get("waiting_tokens", 0),
                msg.get("running_tokens", 0),
            )
            service_heartbeats()
            continue
        if msg_type == "rebalance_done":
            plan_id = msg["plan_id"]
            plan = pending_rebalances.get(plan_id)
            if plan is None:
                service_heartbeats()
                continue
            result = msg.get("result")
            if isinstance(result, dict):
                result_success = bool(result.get("success", False))
                result_error = result.get("error", "rebalance failed")
                result_reason = result.get("reason", result_error)
            else:
                result_success = bool(result)
                result_error = "rebalance failed"
                result_reason = "unknown"
            if not result_success:
                prepared_ranks = set(plan.get("prepared_ranks", set()))
                if prepared_ranks:
                    _send_rebalance_abort(plan_id, plan["plan"], prepared_ranks)
                if plan.get("reply_rank") is not None:
                    response_queues[plan["reply_rank"]].put({
                        "type": "rebalance_response",
                        "request_id": plan["request_id"],
                        "success": False,
                        "error": result_error,
                        "reason": result_reason,
                        "plan_id": plan_id,
                    })
                _release_background_copy_inflight(plan["plan"])
                del pending_rebalances[plan_id]
                service_heartbeats()
                continue

            phase = msg.get("phase", plan.get("phase", "execute"))
            if phase == "prepare":
                plan.setdefault("prepared_ranks", set()).add(msg["rank"])
            plan["pending_ranks"].discard(msg["rank"])
            if not plan["pending_ranks"]:
                if phase == "prepare":
                    plan["phase"] = "execute"
                    plan["pending_ranks"] = set(plan["execute_ranks"])
                    _send_rebalance_execute(plan_id, plan["plan"])
                else:
                    if plan.get("reply_rank") is not None:
                        response_queues[plan["reply_rank"]].put({
                            "type": "rebalance_response",
                            "request_id": plan["request_id"],
                            "success": True,
                            "plan_id": plan_id,
                        })
                    _release_background_copy_inflight(plan["plan"])
                    del pending_rebalances[plan_id]
            service_heartbeats()
            continue
        if msg_type == "route_request":
            try:
                prefix_hash = msg["prefix_hash"]
                prefix_hashes = msg.get("prefix_hashes") or ([prefix_hash] if prefix_hash is not None else [])
                cached_target = route_cache.get(prefix_hash) if prefix_hash is not None else None
                target_rank = None
                route_info = {}
                if cached_target is not None and gbm.get_free_blocks_count(cached_target) >= msg["num_blocks"]:
                    cached_locations = gbm.lookup_prefix(prefix_hash, requester_rank=msg["requester_rank"])
                    candidates = scheduler._candidate_gpus(msg["requester_rank"])
                    cached_pressure = scheduler._load_score(cached_target)
                    min_candidate_pressure = min(scheduler._load_score(gpu_id) for gpu_id in candidates)
                    cache_owner_is_available = any(loc.gpu_id == cached_target for loc in cached_locations)
                    cache_owner_is_not_congested = (
                        cached_target in candidates
                        and cached_pressure <= min_candidate_pressure + route_cache_queue_slack
                    )
                    if cache_owner_is_available and cache_owner_is_not_congested:
                        target_rank = cached_target
                        route_info = {
                            "requester_rank": msg["requester_rank"],
                            "seq_id": msg["seq_id"],
                            "num_tokens": msg["num_tokens"],
                            "num_blocks": msg["num_blocks"],
                            "prefix_hash": prefix_hash,
                            "prefix_hashes": prefix_hashes,
                            "prefix_hit": True,
                            "reason": "route_cache",
                            "target_rank": target_rank,
                            "hit_summary": scheduler._hit_summary(cached_locations),
                            "load_score": scheduler._load_summary(candidates),
                        }
                if target_rank is None:
                    target_rank = scheduler.route_sequence_meta(
                        requester_rank=msg["requester_rank"],
                        seq_id=msg["seq_id"],
                        num_tokens=msg["num_tokens"],
                        num_blocks=msg["num_blocks"],
                        prefix_hash=prefix_hash,
                        return_info=True,
                    )
                    if isinstance(target_rank, tuple):
                        target_rank, route_info = target_rank
                    else:
                        route_info = {}
                    if route_info:
                        route_info["prefix_hashes"] = prefix_hashes
                    if prefix_hash is not None and route_info.get("prefix_hit"):
                        route_cache[prefix_hash] = target_rank
                gbm.reserve_blocks(target_rank, msg["num_blocks"])
                response_queues[msg["reply_rank"]].put({
                    "type": "route_response",
                    "request_id": msg["request_id"],
                    "target_rank": target_rank,
                    "route_info": route_info,
                })
                try:
                    _maybe_schedule_background_copy(route_info)
                except Exception:
                    logger.exception("background transfer copy scheduling failed")
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
                    allow_copy=bool(msg.get("allow_copy", False)),
                )
                if plan is None:
                    response_queues[msg["reply_rank"]].put({
                        "type": "rebalance_response",
                        "request_id": msg["request_id"],
                        "success": False,
                        "reason": scheduler.last_rebalance_fail_reason or "no_plan",
                    })
                    continue

                if not _enqueue_rebalance_plan(plan, msg["request_id"], msg["reply_rank"]):
                    response_queues[msg["reply_rank"]].put({
                        "type": "rebalance_response",
                        "request_id": msg["request_id"],
                        "success": False,
                        "error": "rebalance plan has no target ranks",
                        "reason": "no_plan",
                    })
                    continue
                # 先等待目标 rank 预留成功，再下发 execute，避免源端单边进入 NCCL send。
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
