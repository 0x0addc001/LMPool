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
        self.last_rebalance_fail_reason = ""

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
            "max_tokens": seq.max_tokens,
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
                route_info = msg.get("route_info", {})
                matched_blocks = max(0, int(route_info.get("matched_prefix_blocks", 0)))
                seq.routed_prefix_hashes = list(prefix_hashes[:matched_blocks])
                if return_meta:
                    return {
                        "target_rank": msg["target_rank"],
                        "route_info": route_info,
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
        block_parent_hashes: dict[int, int] | None = None,
        block_access_stats: dict[int, dict] | None = None,
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
            "block_parent_hashes": block_parent_hashes,
            "block_access_stats": block_access_stats,
        })

    def acknowledge_route_admission(self, seq_id: int, num_tokens: int) -> None:
        """Tell the control plane that this worker has admitted one routed request."""
        self.request_queue.put({
            "type": "route_admitted",
            "rank": self.rank,
            "seq_id": int(seq_id),
            "num_tokens": int(num_tokens),
        })

    def acknowledge_route_blocks(self, seq_ids: list[int]) -> None:
        """Release block reservations after routed requests commit first prefill."""
        if self.rank < 0 or not seq_ids:
            return
        self.request_queue.put({
            "type": "route_blocks_committed",
            "rank": self.rank,
            "seq_ids": [int(seq_id) for seq_id in seq_ids],
        })

    def report_transfer_observation(
        self,
        src_gpu: int,
        dst_gpu: int,
        transfer_bytes: int,
        transfer_time_s: float,
    ) -> None:
        """Seed the control-plane cost model with pre-serving P2P calibration."""
        self.request_queue.put({
            "type": "transfer_observation",
            "rank": self.rank,
            "src_gpu": int(src_gpu),
            "dst_gpu": int(dst_gpu),
            "transfer_bytes": int(transfer_bytes),
            "transfer_time_s": float(transfer_time_s),
        })

    def report_prefill_observation(
        self,
        uncached_tokens: int,
        elapsed_s: float,
    ) -> None:
        if self.rank < 0 or uncached_tokens <= 0 or elapsed_s <= 0:
            return
        self.request_queue.put({
            "type": "prefill_observation",
            "rank": self.rank,
            "uncached_tokens": int(uncached_tokens),
            "elapsed_s": float(elapsed_s),
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
                    self.last_rebalance_fail_reason = ""
                else:
                    self.last_rebalance_fail_reason = msg.get("reason", "unknown")
                    self.rebalance_fail_count += 1
                    self.rebalance_fail_reasons[self.last_rebalance_fail_reason] += 1
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
                self.last_rebalance_fail_reason = "error"
                self.rebalance_fail_reasons["error"] += 1
                return False

    def flush_background_copies(
        self,
        prefix_demands: dict[int, int] | None = None,
        timeout_s: float = 120.0,
    ) -> dict:
        """Ask the control plane to place queued hot prefixes, then wait.

        ``prefix_demands`` describes prefixes present in the ingress queue but
        not submitted yet.  Counts are therefore remaining reuse estimates,
        unlike the old fixed expected-reuse constant.  Waiting is explicit so
        benchmarks can include proactive placement time in end-to-end results.
        """
        request_id = uuid.uuid4().hex
        self.request_queue.put({
            "type": "background_copy_flush",
            "request_id": request_id,
            "reply_rank": self.rank,
            "prefix_demands": {
                int(block_hash): max(0, int(count))
                for block_hash, count in (prefix_demands or {}).items()
            },
        })
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while True:
            if self._stashed_messages:
                msg = self._stashed_messages.popleft()
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for background copy flush")
                msg = self.response_queue.get(timeout=remaining)
            if msg.get("request_id") != request_id:
                if self._handle_async_message(msg):
                    continue
                self._stashed_messages.append(msg)
                continue
            if msg.get("type") == "background_copy_flush_response":
                return msg
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("error", "background copy flush failed"))

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
            if msg.get("type") in {
                "route_response",
                "rebalance_response",
                "background_copy_flush_response",
                "error",
            }:
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
    scheduler.owner_spill_sequence_skew = max(
        0.0,
        float(config.get("route_owner_spill_sequence_skew", scheduler.owner_spill_sequence_skew)),
    )
    scheduler.owner_spill_max_extra_cost = max(
        0.0,
        float(config.get("route_owner_spill_max_extra_cost", scheduler.owner_spill_max_extra_cost)),
    )
    scheduler.block_size = max(1, int(config.get("block_size", scheduler.block_size)))
    scheduler.prefill_cost_weight = max(
        0.0,
        float(config.get("route_prefill_cost_weight", scheduler.prefill_cost_weight)),
    )
    scheduler.reclaim_cost_weight = max(
        0.0,
        float(config.get("route_reclaim_cost_weight", scheduler.reclaim_cost_weight)),
    )
    scheduler.transfer_cost_weight = max(
        0.0,
        float(config.get("foreground_transfer_cost_weight", scheduler.transfer_cost_weight)),
    )
    scheduler.foreground_transfer_min_benefit_ratio = max(
        0.0,
        float(config.get(
            "foreground_transfer_min_benefit_ratio",
            scheduler.foreground_transfer_min_benefit_ratio,
        )),
    )
    scheduler.num_layers = max(1, int(config.get("num_layers", scheduler.num_layers)))
    scheduler.num_kv_heads = max(1, int(config.get("num_kv_heads", scheduler.num_kv_heads)))
    scheduler.head_dim = max(1, int(config.get("head_dim", scheduler.head_dim)))
    scheduler.kv_dtype_bytes = max(1, int(config.get("kv_dtype_bytes", scheduler.kv_dtype_bytes)))
    scheduler.transfer_bandwidth_gib_s = max(
        1e-6,
        float(config.get("foreground_transfer_bandwidth_gib_s", scheduler.transfer_bandwidth_gib_s)),
    )
    scheduler.transfer_fixed_latency_ms = max(
        0.0,
        float(config.get("foreground_transfer_fixed_latency_ms", scheduler.transfer_fixed_latency_ms)),
    )
    scheduler.transfer_interference_multiplier = max(
        1.0,
        float(config.get(
            "foreground_transfer_interference_multiplier",
            scheduler.transfer_interference_multiplier,
        )),
    )
    scheduler.prefill_token_time_ms = max(
        0.0,
        float(config.get("foreground_prefill_token_time_ms", scheduler.prefill_token_time_ms)),
    )
    scheduler.prefill_observation_discount = min(
        1.0,
        max(0.0, float(config.get("prefill_observation_discount", 0.5))),
    )
    scheduler.prefill_observation_ewma_alpha = min(
        1.0,
        max(0.0, float(config.get("prefill_observation_ewma_alpha", 0.2))),
    )
    scheduler.future_reuse_discount = min(
        1.0,
        max(0.0, float(config.get("foreground_future_reuse_discount", scheduler.future_reuse_discount))),
    )
    scheduler.transfer_cost_ewma_alpha = min(
        1.0,
        max(0.0, float(config.get(
            "foreground_transfer_ewma_alpha",
            scheduler.transfer_cost_ewma_alpha,
        ))),
    )
    enable_route_cache = bool(config.get("enable_route_cache", False))
    route_cache_queue_slack = float(config.get("route_cache_queue_slack", 2.0))
    enable_background_copy = bool(config.get("enable_background_copy", False))
    background_copy_max_blocks = max(1, int(config.get("background_copy_max_blocks", 1)))
    background_copy_batch_max_candidates = max(
        1,
        int(config.get("background_copy_batch_max_candidates", 16)),
    )
    background_copy_batch_max_blocks = max(
        background_copy_max_blocks,
        int(config.get(
            "background_copy_batch_max_blocks",
            background_copy_max_blocks * background_copy_batch_max_candidates,
        )),
    )
    background_copy_cooldown_s = max(0.0, float(config.get("background_copy_cooldown_s", 2.0)))
    background_copy_hot_threshold = max(1, int(config.get("background_copy_hot_threshold", 3)))
    background_copy_min_load_skew = max(
        0.0,
        float(config.get("background_copy_min_load_skew", scheduler.owner_spill_sequence_skew)),
    )
    background_copy_idle_pressure_threshold = max(
        0.0,
        float(config.get("background_copy_idle_pressure_threshold", 2.0)),
    )
    route_decode_token_weight = max(0.0, float(config.get("route_decode_token_weight", 8.0)))
    background_copy_expected_reuses = max(
        1.0,
        float(config.get("background_copy_expected_reuses", 4.0)),
    )
    # Proactive placement is driven by completed block access observations and
    # ingress demand forecasts.  Do not hold the current request on a hot owner
    # in anticipation of a route-triggered copy that has not completed yet.
    scheduler.enable_routing_guided_copy = False
    scheduler.routing_guided_copy_expected_reuses = background_copy_expected_reuses
    logger.info("control plane process started")

    pending_rebalances: dict[str, dict] = {}
    rebalance_source_blocks_inflight: set[tuple[int, int]] = set()
    transfer_inflight_pairs: set[tuple[int, int]] = set()
    route_cache: dict[int, int] = {}
    background_copy_inflight_pairs: set[tuple[int, int]] = set()
    background_copy_recent: dict[tuple[int, int, int], float] = {}
    background_copy_pair_recent: dict[tuple[int, tuple[int, int]], float] = {}
    background_copy_queues: dict[tuple[int, int], deque] = defaultdict(deque)
    background_copy_candidates: dict[tuple[int, int, int], dict] = {}
    background_copy_rejections: dict[tuple[int, int, int], dict] = {}
    background_future_demands: dict[int, int] = {}
    background_flush_waiters: list[dict] = []
    background_placement_stats = defaultdict(int)
    background_placement_pair_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    placement_leases: dict[int, dict] = {}
    prefix_route_hits: dict[int, int] = defaultdict(int)
    worker_last_heartbeat: dict[int, float] = {rank: time.monotonic() for rank in range(world_size)}
    worker_down: set[int] = set()
    last_control_heartbeat_sent = 0.0

    def _pair_label(pair: tuple[int, int]) -> str:
        return f"{min(pair)}-{max(pair)}"

    def _record_placement_stat(
        metric: str,
        pair: tuple[int, int] | None = None,
        amount: int = 1,
    ) -> None:
        background_placement_stats[metric] += int(amount)
        if pair is not None:
            background_placement_pair_stats[_pair_label(pair)][metric] += int(amount)

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
            if rank not in plan["execute_ranks"]:
                continue
            prepared_ranks = set(plan.get("prepared_ranks", set())) - {rank}
            if prepared_ranks:
                _send_rebalance_abort(plan_id, plan["plan"], prepared_ranks)
            if plan.get("reply_rank") is not None:
                response_queues[plan["reply_rank"]].put({
                    "type": "rebalance_response",
                    "request_id": plan["request_id"],
                    "success": False,
                    "error": f"worker {rank} down",
                    "reason": "worker_down",
                    "plan_id": plan_id,
                })
            _release_rebalance_inflight(plan["plan"])
            del pending_rebalances[plan_id]
        _service_background_placement()

    def _send_rebalance_execute(plan_id: str, plan: dict):
        source_ranks = {
            transfer["src_gpu"] for transfer in plan["transfers"]
        }
        target_ranks = {
            transfer["dst_gpu"] for transfer in plan["transfers"]
        }
        # Wake receivers first. Data-plane workers wait on the control queue, so
        # targets can enter NCCL recv before sources start the matching send.
        for dst_rank in target_ranks:
            if dst_rank in source_ranks:
                continue
            response_queues[dst_rank].put({
                "type": "rebalance_execute",
                "plan_id": plan_id,
                "role": "target",
                "plan": plan,
            })
        for src_rank in source_ranks:
            response_queues[src_rank].put({
                "type": "rebalance_execute",
                "plan_id": plan_id,
                "role": "source",
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

    def _source_block_keys(plan: dict) -> set[tuple[int, int]]:
        return {
            (int(transfer["src_gpu"]), int(block_id))
            for transfer in plan.get("transfers", [])
            for block_id in (
                list(transfer.get("src_blocks", []))
                + list(transfer.get("release_source_blocks", []))
            )
        }

    def _transfer_pairs(plan: dict) -> set[tuple[int, int]]:
        return {
            tuple(sorted((int(transfer["src_gpu"]), int(transfer["dst_gpu"]))))
            for transfer in plan.get("transfers", [])
        }

    def _release_rebalance_inflight(plan: dict, succeeded: bool = False) -> None:
        source_block_keys = _source_block_keys(plan)
        rebalance_source_blocks_inflight.difference_update(source_block_keys)
        gbm.release_transfer_blocks_inflight(source_block_keys)
        transfer_inflight_pairs.difference_update(_transfer_pairs(plan))
        if succeeded:
            elapsed_s = max(
                0.0,
                time.monotonic() - float(plan.get("placement_started_at", time.monotonic())),
            )
            for transfer in plan.get("transfers", []):
                scheduler.observe_placement(
                    scheduler._estimate_transfer_bytes(
                        len(transfer.get("src_blocks", []))
                    ),
                    elapsed_s,
                    int(transfer["src_gpu"]),
                    int(transfer["dst_gpu"]),
                )
        if not plan.get("background"):
            return
        candidate_keys = plan.get("background_candidate_keys")
        if candidate_keys is None:
            candidate_key = plan.get("background_candidate_key")
            candidate_keys = [] if candidate_key is None else [candidate_key]
        pair = tuple(plan.get("background_pair", ()))
        if len(pair) != 2:
            pair = next(iter(_transfer_pairs(plan)), None)
        for raw_candidate_key in candidate_keys:
            candidate_key = tuple(raw_candidate_key)
            candidate = background_copy_candidates.pop(candidate_key, None)
            _record_placement_stat(
                "completed" if succeeded else "failed",
                pair,
            )
            if not succeeded or pair is None:
                continue
            background_copy_rejections.pop(candidate_key, None)
            future_demand = max(
                0,
                int(background_future_demands.get(int(candidate_key[0]), 0)),
            )
            if candidate is not None and future_demand > 0:
                placement_leases[int(candidate_key[0])] = {
                    "src_gpu": int(candidate["src_gpu"]),
                    "target_gpu": int(candidate["dst_gpu"]),
                    # The source already served the completed warmup phase.
                    # Keep the forecast reuse phase on the replica as one
                    # batch, instead of fragmenting every prefix across both
                    # ranks through per-request load comparisons.
                    "remaining": future_demand,
                    "pair": pair,
                }
                _record_placement_stat("lease_created", pair)
        if candidate_keys:
            _record_placement_stat(
                "plans_completed" if succeeded else "plans_failed",
                pair,
            )
        for transfer in plan.get("transfers", []):
            background_copy_inflight_pairs.discard(tuple(sorted((
                int(transfer["src_gpu"]),
                int(transfer["dst_gpu"]),
            ))))

    def _enqueue_rebalance_plan(plan: dict, request_id: str | None, reply_rank: int | None) -> bool:
        source_block_keys = _source_block_keys(plan)
        plan_pairs = _transfer_pairs(plan)
        if not source_block_keys or source_block_keys & rebalance_source_blocks_inflight:
            plan["enqueue_fail_reason"] = "source_busy"
            return False
        if not plan_pairs or plan_pairs & transfer_inflight_pairs:
            plan["enqueue_fail_reason"] = "pair_busy"
            return False
        plan_id = uuid.uuid4().hex
        plan["plan_id"] = plan_id
        plan["placement_started_at"] = time.monotonic()
        source_ranks = {
            transfer["src_gpu"] for transfer in plan["transfers"]
        }
        target_ranks = {
            transfer["dst_gpu"] for transfer in plan["transfers"]
        }
        execute_ranks = source_ranks | target_ranks
        if not target_ranks:
            return False

        rebalance_source_blocks_inflight.update(source_block_keys)
        transfer_inflight_pairs.update(plan_pairs)
        gbm.mark_transfer_blocks_inflight(source_block_keys)

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

    def _drop_background_candidate(candidate_key: tuple[int, int, int], reason: str) -> None:
        candidate = background_copy_candidates.pop(candidate_key, None)
        pair = tuple(candidate["pair"]) if candidate is not None else None
        _record_placement_stat(f"dropped_{reason}", pair)
        if candidate is not None and reason in {
            "low_benefit",
            "no_target_space",
        }:
            background_copy_rejections[candidate_key] = {
                "reason": reason,
                "signature": candidate["signature"],
            }

    def _queue_background_candidate(
        src_gpu: int,
        dst_gpu: int,
        chain: dict,
        predicted_reuses: float,
        trigger: str,
    ) -> None:
        if predicted_reuses <= 0 or src_gpu == dst_gpu:
            return
        leaf_hash = int(chain["hashes"][-1])
        candidate_key = (leaf_hash, int(src_gpu), int(dst_gpu))
        pair_key = tuple(sorted((int(src_gpu), int(dst_gpu))))
        if any(
            location.gpu_id == dst_gpu
            for location in gbm.get_block_location(leaf_hash)
        ):
            return
        now = time.monotonic()
        if now - background_copy_recent.get(candidate_key, 0.0) < background_copy_cooldown_s:
            return
        canonical_recent_key = (leaf_hash, pair_key)
        if now - background_copy_pair_recent.get(canonical_recent_key, 0.0) < background_copy_cooldown_s:
            _record_placement_stat("skipped_pair_cooldown", pair_key)
            return
        signature = (
            tuple(int(block_hash) for block_hash in chain["hashes"]),
            round(float(predicted_reuses), 6),
            int(gbm.get_free_blocks_count(dst_gpu)),
        )
        rejection = background_copy_rejections.get(candidate_key)
        if rejection is not None and rejection.get("signature") == signature:
            _record_placement_stat("skipped_negative_cache", pair_key)
            return
        if rejection is not None:
            background_copy_rejections.pop(candidate_key, None)
        existing = background_copy_candidates.get(candidate_key)
        if existing is not None:
            existing["predicted_reuses"] = max(
                float(existing["predicted_reuses"]),
                float(predicted_reuses),
            )
            return
        background_copy_candidates[candidate_key] = {
            "key": candidate_key,
            "pair": pair_key,
            "src_gpu": int(src_gpu),
            "dst_gpu": int(dst_gpu),
            "hashes": list(chain["hashes"]),
            "predicted_reuses": float(predicted_reuses),
            "trigger": trigger,
            "signature": signature,
        }
        background_copy_queues[pair_key].append(candidate_key)
        _record_placement_stat("queued", pair_key)

    def _discover_background_candidates(trigger: str) -> None:
        if not enable_background_copy:
            return
        for first_gpu, second_gpu in sorted(gbm.nvlink_pairs):
            for src_gpu, dst_gpu in ((first_gpu, second_gpu), (second_gpu, first_gpu)):
                if (
                    trigger == "route"
                    and gbm.get_queue_pressure(src_gpu) - gbm.get_queue_pressure(dst_gpu)
                    < background_copy_min_load_skew
                ):
                    continue
                for chain in gbm.get_hot_prefix_chains(
                    src_gpu,
                    1 if trigger == "ingress_forecast" else background_copy_hot_threshold,
                    background_copy_max_blocks,
                ):
                    if trigger == "ingress_forecast":
                        demanded_indices = [
                            index
                            for index, block_hash in enumerate(chain["hashes"])
                            if int(background_future_demands.get(int(block_hash), 0)) > 0
                        ]
                        if not demanded_indices:
                            continue
                        demanded_end = demanded_indices[-1] + 1
                        chain = {
                            **chain,
                            "hashes": list(chain["hashes"][:demanded_end]),
                            "block_ids": list(chain["block_ids"][:demanded_end]),
                            "access_counts": list(chain["access_counts"][:demanded_end]),
                        }
                    leaf_hash = int(chain["hashes"][-1])
                    forecast_reuses = int(background_future_demands.get(leaf_hash, 0))
                    if forecast_reuses > 0:
                        predicted_reuses = min(
                            background_copy_expected_reuses,
                            float(forecast_reuses),
                        )
                    else:
                        observed_accesses = min(chain["access_counts"])
                        predicted_reuses = min(
                            background_copy_expected_reuses,
                            max(0.0, float(observed_accesses) - 1.0)
                            * scheduler.future_reuse_discount,
                        )
                    _queue_background_candidate(
                        src_gpu,
                        dst_gpu,
                        chain,
                        predicted_reuses,
                        trigger,
                    )

    def _build_background_transfer(candidate: dict) -> tuple[dict | None, str]:
        src_gpu = int(candidate["src_gpu"])
        dst_gpu = int(candidate["dst_gpu"])
        src_blocks = []
        hashes = []
        parent_hashes = []
        access_counts = []
        for block_hash in candidate["hashes"]:
            source_locations = [
                location
                for location in gbm.get_block_location(block_hash)
                if location.gpu_id == src_gpu
            ]
            if not source_locations:
                return None, "stale_source"
            if any(
                location.gpu_id == dst_gpu
                for location in gbm.get_block_location(block_hash)
            ):
                continue
            block_id = source_locations[0].block_id
            src_blocks.append(block_id)
            hashes.append(block_hash)
            parent_hashes.append(gbm.get_block_parent_hash(src_gpu, block_id))
            access_counts.append(gbm.block_access_count[src_gpu].get(block_id, 1))
        if not src_blocks:
            return None, "already_placed"
        return {
            "src_gpu": src_gpu,
            "dst_gpu": dst_gpu,
            "src_blocks": src_blocks,
            "hashes": hashes,
            "parent_hashes": parent_hashes,
            "access_counts": access_counts,
            "mode": "copy",
        }, ""

    def _build_background_batch_plan(
        candidates: list[dict],
    ) -> tuple[dict | None, list[tuple[tuple[int, int, int], str]]]:
        """Coalesce one directed NVLink pair into a single transfer payload."""
        if not candidates:
            return None, []
        src_gpu = int(candidates[0]["src_gpu"])
        dst_gpu = int(candidates[0]["dst_gpu"])
        selected_keys = []
        rejected = []
        block_metadata: dict[int, tuple[int, int, int]] = {}
        selected_candidates = []
        for candidate in candidates:
            transfer, reason = _build_background_transfer(candidate)
            candidate_key = tuple(candidate["key"])
            if transfer is None:
                rejected.append((candidate_key, reason))
                continue
            new_blocks = [
                block_id
                for block_id in transfer["src_blocks"]
                if block_id not in block_metadata
            ]
            if (
                selected_candidates
                and len(block_metadata) + len(new_blocks)
                > background_copy_batch_max_blocks
            ):
                continue
            for block_id, block_hash, parent_hash, access_count in zip(
                transfer["src_blocks"],
                transfer["hashes"],
                transfer["parent_hashes"],
                transfer["access_counts"],
            ):
                block_metadata.setdefault(
                    int(block_id),
                    (int(block_hash), int(parent_hash), int(access_count)),
                )
            selected_keys.append(candidate_key)
            selected_candidates.append(candidate)

        if not selected_candidates:
            return None, rejected
        if gbm.get_free_blocks_count(dst_gpu) < len(block_metadata):
            rejected.extend(
                (tuple(candidate["key"]), "no_target_space")
                for candidate in selected_candidates
            )
            return None, rejected

        src_blocks = list(block_metadata)
        hashes = [block_metadata[block_id][0] for block_id in src_blocks]
        parent_hashes = [block_metadata[block_id][1] for block_id in src_blocks]
        access_counts = [block_metadata[block_id][2] for block_id in src_blocks]
        transfer_bytes = scheduler._estimate_transfer_bytes(len(src_blocks))
        transfer_cost_ms = scheduler._estimate_transfer_cost_ms(
            transfer_bytes,
            src_gpu,
            dst_gpu,
        )
        # Without an intervening eviction, the target would self-warm after
        # its first miss. Count one avoidable cold prefill per copied block;
        # multiplying by every forecast reuse materially overstates benefit.
        saved_prefill_ms = (
            len(src_blocks)
            * scheduler.block_size
            * scheduler.estimate_prefill_token_time_ms(dst_gpu)
        )
        if saved_prefill_ms < (
            transfer_cost_ms * scheduler.foreground_transfer_min_benefit_ratio
        ):
            rejected.extend(
                (tuple(candidate["key"]), "low_benefit")
                for candidate in selected_candidates
            )
            return None, rejected
        return {
            "gpu_id": src_gpu,
            "needed_blocks": len(src_blocks),
            "mode": "copy",
            "background": True,
            "background_candidate_keys": selected_keys,
            "background_trigger": selected_candidates[0]["trigger"],
            "estimated_transfer_bytes": transfer_bytes,
            "estimated_transfer_cost_ms": transfer_cost_ms,
            "estimated_saved_prefill_ms": saved_prefill_ms,
            "estimated_future_reuses": sum(
                max(0.0, float(candidate["predicted_reuses"]))
                for candidate in selected_candidates
            ),
            "transfers": [{
                "src_gpu": src_gpu,
                "dst_gpu": dst_gpu,
                "src_blocks": src_blocks,
                "hashes": hashes,
                "parent_hashes": parent_hashes,
                "access_counts": access_counts,
                "mode": "copy",
            }],
        }, rejected

    def _dispatch_background_candidates() -> None:
        if not enable_background_copy:
            return
        for pair_key, candidate_queue in list(background_copy_queues.items()):
            if pair_key in background_copy_inflight_pairs or pair_key in transfer_inflight_pairs:
                continue
            if any(
                gbm.get_queue_pressure(gpu_id) > background_copy_idle_pressure_threshold
                for gpu_id in pair_key
            ):
                continue
            while candidate_queue:
                queued_keys = list(candidate_queue)
                candidate_queue.clear()
                batch_candidates = []
                deferred_keys = []
                direction = None
                for candidate_key in queued_keys:
                    candidate = background_copy_candidates.get(candidate_key)
                    if candidate is None:
                        continue
                    candidate_direction = (
                        int(candidate["src_gpu"]),
                        int(candidate["dst_gpu"]),
                    )
                    if direction is None:
                        direction = candidate_direction
                    if (
                        candidate_direction != direction
                        or len(batch_candidates) >= background_copy_batch_max_candidates
                    ):
                        deferred_keys.append(candidate_key)
                        continue
                    batch_candidates.append(candidate)
                candidate_queue.extend(deferred_keys)
                plan, rejected = _build_background_batch_plan(batch_candidates)
                _record_placement_stat("evaluated", pair_key, len(batch_candidates))
                for candidate_key, reason in rejected:
                    _drop_background_candidate(candidate_key, reason)
                selected_keys = (
                    set(plan.get("background_candidate_keys", []))
                    if plan else set()
                )
                rejected_keys = {key for key, _reason in rejected}
                for candidate in batch_candidates:
                    candidate_key = tuple(candidate["key"])
                    if (
                        candidate_key not in selected_keys
                        and candidate_key not in rejected_keys
                    ):
                        candidate_queue.append(candidate_key)
                if plan is None:
                    continue
                plan["background_pair"] = pair_key
                if not _enqueue_rebalance_plan(plan, request_id=None, reply_rank=None):
                    for candidate_key in reversed(plan["background_candidate_keys"]):
                        candidate_queue.appendleft(tuple(candidate_key))
                    break
                background_copy_inflight_pairs.add(pair_key)
                for candidate_key in plan["background_candidate_keys"]:
                    candidate_key = tuple(candidate_key)
                    background_copy_recent[candidate_key] = time.monotonic()
                    background_copy_pair_recent[(candidate_key[0], pair_key)] = time.monotonic()
                _record_placement_stat(
                    "dispatched", pair_key, len(plan["background_candidate_keys"])
                )
                _record_placement_stat("plans_dispatched", pair_key)
                logger.info(
                    "background transfer batch dispatched: candidates=%s src=%s dst=%s blocks=%s "
                    "trigger=%s predicted_reuses=%.2f",
                    len(plan["background_candidate_keys"]),
                    plan["transfers"][0]["src_gpu"],
                    plan["transfers"][0]["dst_gpu"],
                    plan["transfers"][0]["src_blocks"],
                    plan["background_trigger"],
                    plan["estimated_future_reuses"],
                )
                break

    def _background_work_pending() -> bool:
        return bool(
            background_copy_candidates
            or background_copy_inflight_pairs
            or any(
                pending["plan"].get("background")
                for pending in pending_rebalances.values()
            )
        )

    def _complete_background_flush_waiters() -> None:
        if _background_work_pending():
            return
        while background_flush_waiters:
            waiter = background_flush_waiters.pop(0)
            response_queues[waiter["reply_rank"]].put({
                "type": "background_copy_flush_response",
                "request_id": waiter["request_id"],
                "success": True,
                "placement_stats": dict(background_placement_stats),
                "placement_pair_stats": {
                    pair: dict(stats)
                    for pair, stats in background_placement_pair_stats.items()
                },
            })

    def _service_background_placement(discover: bool = False, trigger: str = "periodic") -> None:
        if discover:
            _discover_background_candidates(trigger)
        _dispatch_background_candidates()
        _complete_background_flush_waiters()

    def _maybe_schedule_background_copy(route_info: dict) -> None:
        if not enable_background_copy or not route_info.get("prefix_hit"):
            return
        if route_info.get("reason") == "prefix_hit_pair_spill":
            return
        prefix_hash = route_info.get("matched_prefix_hash", route_info.get("prefix_hash"))
        if prefix_hash is None:
            return
        prefix_route_hits[int(prefix_hash)] += 1
        if prefix_route_hits[int(prefix_hash)] < background_copy_hot_threshold:
            return
        _service_background_placement(discover=True, trigger="route")

    def _route_from_placement_lease(msg: dict, prefix_hashes: list[int]) -> tuple[int, dict] | None:
        """Bind copied prefixes to future requests while balancing pair load."""
        matched_hash = next(
            (
                block_hash
                for block_hash in reversed(prefix_hashes)
                if placement_leases.get(int(block_hash), {}).get("remaining", 0) > 0
            ),
            None,
        )
        if matched_hash is None:
            return None
        lease = placement_leases[int(matched_hash)]
        target_gpu = int(lease["target_gpu"])
        hit_summary = scheduler._lookup_contiguous_prefix(
            prefix_hashes,
            msg["requester_rank"],
        )
        target_blocks = hit_summary.get(target_gpu, [])
        if not target_blocks:
            return None
        required_blocks = scheduler._required_new_blocks(
            msg["num_blocks"], len(target_blocks)
        )
        if not gbm.can_allocate_effective(target_gpu, required_blocks, target_blocks):
            return None
        lease["remaining"] -= 1
        if lease["remaining"] <= 0:
            placement_leases.pop(int(matched_hash), None)
        pair = tuple(lease["pair"])
        _record_placement_stat("lease_routes", pair)
        route_info = {
            "requester_rank": msg["requester_rank"],
            "seq_id": msg["seq_id"],
            "num_tokens": msg["num_tokens"],
            "num_blocks": msg["num_blocks"],
            "prefix_hash": matched_hash,
            "prefix_hashes": prefix_hashes,
            "prefix_hit": True,
            "reason": "placement_lease",
            "target_rank": target_gpu,
            "hit_summary": hit_summary,
            "matched_prefix_blocks": len(target_blocks),
            "matched_prefix_hash": prefix_hashes[len(target_blocks) - 1],
            "prefix_owner_rank": target_gpu,
            "required_new_blocks": required_blocks,
            "load_score": scheduler._load_summary(list(range(world_size))),
            "estimated_costs": scheduler._route_cost_summary(
                list(range(world_size)),
                msg["num_tokens"],
                msg["num_blocks"],
                {gpu_id: len(blocks) for gpu_id, blocks in hit_summary.items()},
                hit_summary,
            ),
            "placement_lease": True,
        }
        scheduler._annotate_target_capacity(
            route_info, target_gpu, required_blocks, target_blocks
        )
        return target_gpu, route_info

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
            _service_background_placement()
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
                msg.get("block_parent_hashes"),
                msg.get("block_access_stats"),
            )
            # Worker snapshots update authoritative state, but candidate
            # discovery is ingress/route driven. Scanning every decode-step
            # snapshot previously produced tens of thousands of no-op scans.
            _service_background_placement(discover=False, trigger="block_state")
            service_heartbeats()
            continue
        if msg_type == "background_copy_flush":
            background_future_demands.clear()
            background_future_demands.update({
                int(block_hash): max(0, int(count))
                for block_hash, count in msg.get("prefix_demands", {}).items()
            })
            background_flush_waiters.append({
                "request_id": msg["request_id"],
                "reply_rank": msg["reply_rank"],
            })
            _service_background_placement(discover=True, trigger="ingress_forecast")
            service_heartbeats()
            continue
        if msg_type == "route_admitted":
            gbm.acknowledge_route_load(
                msg["rank"],
                msg.get("num_tokens", 0),
                seq_id=msg.get("seq_id"),
            )
            service_heartbeats()
            continue
        if msg_type == "route_blocks_committed":
            for seq_id in msg.get("seq_ids", []):
                gbm.acknowledge_route_blocks(msg["rank"], seq_id)
            service_heartbeats()
            continue
        if msg_type == "transfer_observation":
            scheduler.observe_transfer(
                int(msg.get("transfer_bytes", 0)),
                float(msg.get("transfer_time_s", 0.0)),
                msg.get("src_gpu"),
                msg.get("dst_gpu"),
            )
            service_heartbeats()
            continue
        if msg_type == "prefill_observation":
            scheduler.observe_prefill(
                int(msg["rank"]),
                int(msg.get("uncached_tokens", 0)),
                float(msg.get("elapsed_s", 0.0)),
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
                if msg.get("role") == "source" and msg.get("phase") == "execute":
                    observations = result.get("transfer_observations") or [{
                        "transfer_bytes": result.get("transfer_bytes", 0),
                        "transfer_time_s": result.get("transfer_time_s", 0.0),
                    }]
                    for observation in observations:
                        scheduler.observe_transfer(
                            int(observation.get("transfer_bytes", 0)),
                            float(observation.get("transfer_time_s", 0.0)),
                            observation.get("src_gpu"),
                            observation.get("dst_gpu"),
                        )
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
                _release_rebalance_inflight(plan["plan"])
                del pending_rebalances[plan_id]
                _service_background_placement()
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
                    _release_rebalance_inflight(plan["plan"], succeeded=True)
                    del pending_rebalances[plan_id]
                    _service_background_placement()
            service_heartbeats()
            continue
        if msg_type == "route_request":
            try:
                prefix_hash = msg["prefix_hash"]
                prefix_hashes = msg.get("prefix_hashes") or ([prefix_hash] if prefix_hash is not None else [])
                cached_prefix_hash = next(
                    (block_hash for block_hash in reversed(prefix_hashes) if block_hash in route_cache),
                    None,
                )
                cached_target = route_cache.get(cached_prefix_hash) if cached_prefix_hash is not None else None
                target_rank = None
                route_info = {}
                leased_route = _route_from_placement_lease(msg, prefix_hashes)
                if leased_route is not None:
                    target_rank, route_info = leased_route
                if (
                    target_rank is None
                    and
                    enable_route_cache
                    and cached_target is not None
                ):
                    contiguous_hits = scheduler._lookup_contiguous_prefix(
                        prefix_hashes,
                        msg["requester_rank"],
                    )
                    candidates = scheduler._candidate_gpus(msg["requester_rank"])
                    owner_candidates = sorted({
                        gpu_id
                        for gpu_id, block_ids in contiguous_hits.items()
                        if (
                            gpu_id in candidates
                            and gbm.can_allocate_effective(
                                gpu_id,
                                scheduler._required_new_blocks(
                                    msg["num_blocks"],
                                    len(block_ids),
                                ),
                                block_ids,
                            )
                        )
                    })
                    cache_owner = None
                    if owner_candidates:
                        cache_owner = min(
                            owner_candidates,
                            key=lambda gpu_id: (
                                scheduler._route_cost(
                                    gpu_id,
                                    msg["num_tokens"],
                                    msg["num_blocks"],
                                    len(contiguous_hits[gpu_id]),
                                    contiguous_hits[gpu_id],
                                ),
                                -gbm.get_free_blocks_count(gpu_id),
                                gpu_id,
                            ),
                        )
                    candidate_costs = {
                        gpu_id: scheduler._route_cost(
                            gpu_id,
                            msg["num_tokens"],
                            msg["num_blocks"],
                            len(contiguous_hits.get(gpu_id, [])),
                            contiguous_hits.get(gpu_id, []),
                        )
                        for gpu_id in candidates
                        if gbm.can_allocate_effective(
                            gpu_id,
                            scheduler._required_new_blocks(
                                msg["num_blocks"],
                                len(contiguous_hits.get(gpu_id, [])),
                            ),
                            contiguous_hits.get(gpu_id, []),
                        )
                    }
                    cached_cost = (
                        candidate_costs.get(cache_owner, float("inf"))
                        if cache_owner is not None
                        else float("inf")
                    )
                    cache_owner_is_not_congested = (
                        cache_owner is not None
                        and candidate_costs
                        and cached_cost
                        <= min(candidate_costs.values()) + route_cache_queue_slack
                    )
                    if cache_owner_is_not_congested:
                        target_rank = cache_owner
                        route_info = {
                            "requester_rank": msg["requester_rank"],
                            "seq_id": msg["seq_id"],
                            "num_tokens": msg["num_tokens"],
                            "num_blocks": msg["num_blocks"],
                            "prefix_hash": cached_prefix_hash,
                            "prefix_hashes": prefix_hashes,
                            "prefix_hit": True,
                            "reason": "route_cache",
                            "target_rank": target_rank,
                            "hit_summary": contiguous_hits,
                            "matched_prefix_blocks": len(contiguous_hits[target_rank]),
                            "matched_prefix_hash": prefix_hashes[len(contiguous_hits[target_rank]) - 1],
                            "prefix_owner_rank": target_rank,
                            "required_new_blocks": scheduler._required_new_blocks(
                                msg["num_blocks"],
                                len(contiguous_hits[target_rank]),
                            ),
                            "load_score": scheduler._load_summary(candidates),
                            "estimated_costs": scheduler._route_cost_summary(
                                candidates,
                                msg["num_tokens"],
                                msg["num_blocks"],
                                {
                                    gpu_id: len(block_ids)
                                    for gpu_id, block_ids in contiguous_hits.items()
                                },
                                contiguous_hits,
                            ),
                        }
                        scheduler._annotate_target_capacity(
                            route_info,
                            target_rank,
                            route_info["required_new_blocks"],
                            contiguous_hits[target_rank],
                        )
                if target_rank is None:
                    target_rank = scheduler.route_sequence_meta(
                        requester_rank=msg["requester_rank"],
                        seq_id=msg["seq_id"],
                        num_tokens=msg["num_tokens"],
                        num_blocks=msg["num_blocks"],
                        prefix_hash=prefix_hash,
                        return_info=True,
                        prefix_hashes=prefix_hashes,
                    )
                    if isinstance(target_rank, tuple):
                        target_rank, route_info = target_rank
                    else:
                        route_info = {}
                    if route_info:
                        route_info["prefix_hashes"] = prefix_hashes
                    matched_prefix_blocks = int(route_info.get("matched_prefix_blocks", 0))
                    if matched_prefix_blocks > 0 and target_rank in route_info.get("hit_summary", {}):
                        route_cache[prefix_hashes[matched_prefix_blocks - 1]] = target_rank
                gbm.reserve_blocks(
                    target_rank,
                    int(route_info.get("required_new_blocks", msg["num_blocks"])),
                    seq_id=msg["seq_id"],
                    protected_block_ids=list(
                        route_info.get("hit_summary", {}).get(target_rank, [])
                    ),
                )
                gbm.reserve_route_load(
                    target_rank,
                    int(msg["num_tokens"] + route_decode_token_weight * msg.get("max_tokens", 0)),
                    seq_id=msg["seq_id"],
                )
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
                    excluded_source_blocks={
                        block_id
                        for source_gpu, block_id in rebalance_source_blocks_inflight
                        if source_gpu == msg["gpu_id"]
                    },
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
                    reason = plan.get("enqueue_fail_reason", "no_plan")
                    response_queues[msg["reply_rank"]].put({
                        "type": "rebalance_response",
                        "request_id": msg["request_id"],
                        "success": False,
                        "error": f"rebalance plan rejected: {reason}",
                        "reason": reason,
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
