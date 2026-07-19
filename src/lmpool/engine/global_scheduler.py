"""
全局调度器 (Global Scheduler)

负责跨 GPU 的两类决策：
1. 请求路由：新序列应该在哪个 GPU 上执行（前缀复用 + 负载均衡）
2. 显存重平衡：本地空闲块不足时，编排跨 GPU 的 transfer 操作

设计要点：
1. 依赖 GlobalBlockManager 获取全局页表和空闲块分布
2. 依赖本地 BlockManager 做前缀 hash 计算
3. transfer 编排需要和目标 GPU 上的 GlobalBlockManager 协同
"""

import logging
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional
from lmpool.engine.sequence import Sequence

logger = logging.getLogger(__name__)


class GlobalScheduler:
    """
    全局调度器

    职责：
    - route_sequence:   决定新序列的归属 GPU
    - rebalance:        编排 transfer，为本 GPU 腾出空闲块
    - preempt_for_transfer: 当 rebalance 失败时，选择序列回退
    """

    def __init__(self, gbm, block_manager, model_runner=None):
        """
        参数:
            gbm: GlobalBlockManager 实例（维护全局页表）
            block_manager: 本地 BlockManager 实例（提供 compute_hash 等接口）
            model_runner: ModelRunner 实例（提供 kv_cache 张量引用）
        """
        self.gbm = gbm
        self.block_manager = block_manager
        self.model_runner = model_runner
        self.last_rebalance_fail_reason = ""
        self.prefix_hit_weight = 8.0
        self.queue_pressure_weight = 1.0
        self.free_block_weight = 0.05
        self.load_weight = 0.01
        self.waiting_token_weight = 1.0
        self.running_token_weight = 0.25
        self.running_sequence_weight = 32.0
        self.load_bypass_threshold = 512.0
        self.owner_spill_sequence_skew = 2.0
        self.owner_spill_max_extra_cost = 2048.0
        self.enable_routing_guided_copy = False
        self.routing_guided_copy_expected_reuses = 4.0
        # Route costs use token-equivalent units so queue work, repeated
        # prefill, and cache reclamation can be compared directly.
        self.block_size = 256
        self.prefill_cost_weight = 1.0
        self.reclaim_cost_weight = 0.5
        self.transfer_cost_weight = 1.0
        self.foreground_transfer_min_benefit_ratio = 1.5
        # Foreground transfer admission is evaluated in wall-clock time. These
        # defaults are deliberately conservative and can be calibrated from
        # benchmark_kv_transfer.py on the target machine.
        self.num_layers = 28
        self.num_kv_heads = 8
        self.head_dim = 128
        self.kv_dtype_bytes = 2
        self.transfer_bandwidth_gib_s = 3.5
        self.transfer_fixed_latency_ms = 2.0
        self.transfer_interference_multiplier = 1.5
        self.prefill_token_time_ms = 0.02
        self.prefill_observation_discount = 0.5
        self.prefill_observation_ewma_alpha = 0.2
        self.observed_prefill_token_time_ms_by_gpu: dict[int, float] = {}
        self.future_reuse_discount = 0.5
        self.transfer_cost_ewma_alpha = 0.25
        self.observed_transfer_extra_ms: float | None = None
        self.observed_transfer_extra_ms_by_pair: dict[tuple[int, int], float] = {}
        self.observed_placement_extra_ms_by_pair: dict[tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    # 请求路由
    # ------------------------------------------------------------------

    def route_sequence(self, seq: Sequence) -> int:
        """
        决定 seq 应该在哪个 GPU 上执行

        策略（按优先级）:
        1. 计算 seq 的前缀 hash
        2. 查询 gbm.lookup_prefix 获取前缀命中的 GPU 列表
        3. 选择前缀命中数 × 拓扑权重最高的 GPU
        4. 若无命中，选择当前空闲块最多的 GPU
        5. 如果命中 GPU 空闲块不足，选择最适合做 transfer 的目标 GPU

        返回:
            target_gpu_id: 推荐的执行 GPU rank
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        prefix_hashes = self._compute_prefix_hashes(seq)
        prefix_hash = prefix_hashes[-1] if prefix_hashes else None
        return self.route_sequence_meta(
            requester_rank=rank,
            seq_id=seq.seq_id,
            num_tokens=seq.num_tokens,
            num_blocks=seq.num_blocks,
            prefix_hash=prefix_hash,
            prefix_hashes=prefix_hashes,
        )

    def route_sequence_meta(
        self,
        requester_rank: int,
        seq_id: int,
        num_tokens: int,
        num_blocks: int,
        prefix_hash: Optional[int],
        return_info: bool = False,
        prefix_hashes: Optional[List[int]] = None,
    ) -> int | tuple[int, dict]:
        """
        Route using metadata only.

        This is the control-plane API used by GlobalControlProcess. Prefix hash
        is computed by the requester because it depends on the local
        BlockManager hash implementation, but all global state is read from GBM.
        """
        rank = requester_rank
        prefix_hashes = list(prefix_hashes or ([prefix_hash] if prefix_hash is not None else []))
        candidates = self._candidate_gpus(rank)
        free_snapshot = {gpu_id: self.gbm.get_free_blocks_count(gpu_id) for gpu_id in candidates}
        reclaimable_snapshot = {
            gpu_id: self.gbm.get_reclaimable_blocks_count(gpu_id)
            for gpu_id in candidates
        }
        effective_capacity_snapshot = {
            gpu_id: self.gbm.get_effective_capacity(gpu_id)
            for gpu_id in candidates
        }
        route_info = {
            "requester_rank": rank,
            "seq_id": seq_id,
            "num_tokens": num_tokens,
            "num_blocks": num_blocks,
            "prefix_hash": prefix_hash,
            "prefix_hashes": prefix_hashes,
            "free_snapshot": free_snapshot,
            "reclaimable_snapshot": reclaimable_snapshot,
            "effective_capacity_snapshot": effective_capacity_snapshot,
            "prefix_hit": False,
            "reason": None,
        }

        if not prefix_hashes:
            # 没有完整的块前缀，只在本地 / NVLink 伙伴之间选空闲更多的 GPU
            target = self._select_best_candidate(
                rank,
                candidates,
                required_blocks=num_blocks,
                num_tokens=num_tokens,
                num_blocks=num_blocks,
            )
            route_info["reason"] = "most_free_no_full_blocks"
            route_info["target_rank"] = target
            self._annotate_target_capacity(route_info, target, num_blocks)
            logger.info(
                "route seq %s: tokens=%s blocks=%s prefix=none free=%s -> GPU %s "
                "(reason=most_free_no_full_blocks)",
                seq_id, num_tokens, num_blocks, free_snapshot, target,
            )
            return (target, route_info) if return_info else target

        # Only a contiguous chain from block zero is reusable. A deeper hash
        # without all of its predecessors cannot seed paged-attention state.
        hit_summary = self._lookup_contiguous_prefix(prefix_hashes, rank)
        gpu_hit_count = {
            gpu_id: len(block_ids)
            for gpu_id, block_ids in hit_summary.items()
            if block_ids
        }

        if not gpu_hit_count:
            # 没有命中任何 GPU，只在本地 / NVLink 伙伴之间选空闲更多的 GPU
            target = self._select_best_candidate(
                rank,
                candidates,
                required_blocks=num_blocks,
                num_tokens=num_tokens,
                num_blocks=num_blocks,
            )
            route_info["reason"] = "most_free_no_prefix_hit"
            route_info["target_rank"] = target
            self._annotate_target_capacity(route_info, target, num_blocks)
            logger.info(
                "route seq %s: tokens=%s blocks=%s prefix=%s hits={} free=%s -> GPU %s "
                "(reason=most_free_no_prefix_hit)",
                seq_id, num_tokens, num_blocks, prefix_hash, free_snapshot, target,
            )
            return (target, route_info) if return_info else target

        # 4. 加权打分
        # score = 命中块数 × 拓扑权重 × prefix_hit_weight
        #         - token-aware_load × load_weight
        #         + free_blocks × free_block_weight
        # 拓扑权重：同 GPU=2.0, NVLink 伙伴=1.0, 其他 GPU=0.0
        # 也就是说：prefix-hit 只在“本地 / NVLink 直连伙伴”之间竞争，
        # 但不会无视 worker queue pressure。
        best_gpu = rank  # 默认本地
        best_score = float("-inf")
        best_cost = float("inf")
        failed_gpus = []  # 记录空闲不足的命中 GPU

        for gpu_id, hit_count in gpu_hit_count.items():
            topo_weight = self._get_topo_weight(rank, gpu_id)
            if topo_weight <= 0:
                continue
            score = self._route_score(rank, gpu_id, hit_count)

            needed = self._required_new_blocks(num_blocks, hit_count)
            if self.gbm.can_allocate_effective(
                gpu_id,
                needed,
                hit_summary.get(gpu_id, []),
            ):
                cost = self._route_cost(
                    gpu_id,
                    num_tokens,
                    num_blocks,
                    hit_count,
                    hit_summary.get(gpu_id, []),
                )
                if (cost, -score, gpu_id) < (best_cost, -best_score, best_gpu):
                    best_score = score
                    best_cost = cost
                    best_gpu = gpu_id
            else:
                # 空闲不足，暂存作为备选
                failed_gpus.append((gpu_id, score, hit_count))

        if best_score > float("-inf"):
            load_summary = self._load_summary(candidates)
            candidate_costs = {
                gpu_id: self._route_cost(
                    gpu_id,
                    num_tokens,
                    num_blocks,
                    gpu_hit_count.get(gpu_id, 0),
                    hit_summary.get(gpu_id, []),
                )
                for gpu_id in candidates
                if self.gbm.can_allocate_effective(
                    gpu_id,
                    self._required_new_blocks(
                        num_blocks,
                        gpu_hit_count.get(gpu_id, 0),
                    ),
                    hit_summary.get(gpu_id, []),
                )
            }
            least_loaded_gpu = min(
                candidate_costs,
                key=lambda gpu_id: (candidate_costs[gpu_id], gpu_id),
            )
            partner_gpu = self.gbm._get_nvlink_partner(best_gpu)
            spill_gpu = (
                partner_gpu
                if partner_gpu in candidate_costs
                else least_loaded_gpu
            )
            owner_pressure = self.gbm.get_queue_pressure(best_gpu)
            spill_pressure = self.gbm.get_queue_pressure(spill_gpu)
            replica_copy_cost_ms = self._estimate_transfer_cost_ms(
                self._estimate_transfer_bytes(gpu_hit_count.get(best_gpu, 0)),
                best_gpu,
                spill_gpu,
            )
            replica_copy_saved_ms = (
                self.routing_guided_copy_expected_reuses
                * gpu_hit_count.get(best_gpu, 0)
                * self.block_size
                * self.prefill_token_time_ms
            )
            replica_copy_candidate = (
                self.enable_routing_guided_copy
                and best_gpu != spill_gpu
                and owner_pressure - spill_pressure >= self.owner_spill_sequence_skew
                and replica_copy_saved_ms
                >= replica_copy_cost_ms * self.foreground_transfer_min_benefit_ratio
            )
            pair_spill = (
                not replica_copy_candidate
                and best_gpu != spill_gpu
                and owner_pressure - spill_pressure >= self.owner_spill_sequence_skew
                and candidate_costs[spill_gpu]
                <= best_cost + self.owner_spill_max_extra_cost
            )
            if (
                pair_spill
                or (
                    best_gpu != least_loaded_gpu
                    and candidate_costs[least_loaded_gpu]
                    + self.load_bypass_threshold < best_cost
                )
            ):
                bypass_gpu = spill_gpu if pair_spill else least_loaded_gpu
                route_info["prefix_hit"] = True
                route_info["reason"] = (
                    "prefix_hit_pair_spill" if pair_spill else "prefix_hit_load_bypass"
                )
                route_info["target_rank"] = bypass_gpu
                route_info["hit_summary"] = hit_summary
                route_info["matched_prefix_blocks"] = gpu_hit_count.get(bypass_gpu, 0)
                route_info["prefix_owner_rank"] = best_gpu
                route_info["matched_prefix_hash"] = prefix_hashes[gpu_hit_count[best_gpu] - 1]
                route_info["required_new_blocks"] = self._required_new_blocks(
                    num_blocks,
                    gpu_hit_count.get(bypass_gpu, 0),
                )
                route_info["owner_pressure"] = owner_pressure
                route_info["spill_pressure"] = spill_pressure
                route_info["scores"] = self._score_summary(rank, gpu_hit_count)
                route_info["load_score"] = load_summary
                route_info["queue_pressure"] = self._queue_pressure_summary(candidates)
                route_info["estimated_costs"] = self._route_cost_summary(
                    candidates,
                    num_tokens,
                    num_blocks,
                    gpu_hit_count,
                    hit_summary,
                )
                self._annotate_target_capacity(
                    route_info,
                    bypass_gpu,
                    route_info["required_new_blocks"],
                    hit_summary.get(bypass_gpu, []),
                )
                logger.info(
                    "route seq %s: prefix=%s hits=%s load=%s -> GPU %s "
                    "(reason=prefix_hit_load_bypass, owner=%s)",
                    seq_id,
                    prefix_hash,
                    hit_summary,
                    load_summary,
                    bypass_gpu,
                    best_gpu,
                )
                return (bypass_gpu, route_info) if return_info else bypass_gpu

            route_info["prefix_hit"] = True
            route_info["reason"] = "prefix_hit"
            route_info["target_rank"] = best_gpu
            route_info["hit_summary"] = hit_summary
            route_info["matched_prefix_blocks"] = gpu_hit_count[best_gpu]
            route_info["prefix_owner_rank"] = best_gpu
            route_info["matched_prefix_hash"] = prefix_hashes[gpu_hit_count[best_gpu] - 1]
            route_info["required_new_blocks"] = self._required_new_blocks(
                num_blocks,
                gpu_hit_count[best_gpu],
            )
            if replica_copy_candidate:
                route_info["reason"] = "prefix_hit_replica_copy"
                route_info["replica_copy_candidate"] = True
                route_info["owner_pressure"] = owner_pressure
                route_info["spill_pressure"] = spill_pressure
                route_info["replica_copy_cost_ms"] = replica_copy_cost_ms
                route_info["replica_copy_saved_ms"] = replica_copy_saved_ms
            route_info["scores"] = self._score_summary(rank, gpu_hit_count)
            route_info["load_score"] = load_summary
            route_info["queue_pressure"] = self._queue_pressure_summary(candidates)
            route_info["estimated_costs"] = self._route_cost_summary(
                candidates,
                num_tokens,
                num_blocks,
                gpu_hit_count,
                hit_summary,
            )
            self._annotate_target_capacity(
                route_info,
                best_gpu,
                route_info["required_new_blocks"],
                hit_summary.get(best_gpu, []),
            )
            logger.info(
                "route seq %s: tokens=%s blocks=%s prefix=%s hits=%s free=%s scores=%s "
                "-> GPU %s (reason=prefix_hit, score=%.1f, target_free=%s/%s)",
                seq_id,
                num_tokens,
                num_blocks,
                prefix_hash,
                hit_summary,
                free_snapshot,
                self._score_summary(rank, gpu_hit_count),
                best_gpu,
                best_score,
                self.gbm.get_free_blocks_count(best_gpu),
                route_info["required_new_blocks"],
            )
            return (best_gpu, route_info) if return_info else best_gpu

        if failed_gpus:
            allocatable_candidates = [
                gpu_id
                for gpu_id in candidates
                if self.gbm.can_allocate_effective(
                    gpu_id,
                    self._required_new_blocks(
                        num_blocks,
                        gpu_hit_count.get(gpu_id, 0),
                    ),
                    hit_summary.get(gpu_id, []),
                )
            ]
            if allocatable_candidates:
                target = min(
                    allocatable_candidates,
                    key=lambda gpu_id: (
                        self._route_cost(
                            gpu_id,
                            num_tokens,
                            num_blocks,
                            gpu_hit_count.get(gpu_id, 0),
                            hit_summary.get(gpu_id, []),
                        ),
                        -self.gbm.get_effective_capacity(
                            gpu_id,
                            hit_summary.get(gpu_id, []),
                        ),
                        gpu_id,
                    ),
                )
                route_info["prefix_hit"] = True
                route_info["reason"] = "prefix_owner_full_fallback"
                route_info["target_rank"] = target
                route_info["hit_summary"] = hit_summary
                route_info["matched_prefix_blocks"] = gpu_hit_count.get(target, 0)
                route_info["prefix_owner_rank"] = max(
                    gpu_hit_count,
                    key=lambda gpu_id: gpu_hit_count[gpu_id],
                )
                owner_hit_count = gpu_hit_count[route_info["prefix_owner_rank"]]
                route_info["matched_prefix_hash"] = prefix_hashes[owner_hit_count - 1]
                route_info["required_new_blocks"] = self._required_new_blocks(
                    num_blocks,
                    gpu_hit_count.get(target, 0),
                )
                route_info["failed_gpus"] = [(g, s) for g, s, _ in failed_gpus]
                route_info["load_score"] = self._load_summary(candidates)
                route_info["queue_pressure"] = self._queue_pressure_summary(candidates)
                self._annotate_target_capacity(
                    route_info,
                    target,
                    route_info["required_new_blocks"],
                    hit_summary.get(target, []),
                )
                logger.info(
                    "route seq %s: prefix owners lack space; free=%s -> GPU %s "
                    "(reason=prefix_owner_full_fallback)",
                    seq_id,
                    free_snapshot,
                    target,
                )
                return (target, route_info) if return_info else target

            failed_gpus.sort(
                key=lambda item: (
                    self._route_cost(
                        item[0],
                        num_tokens,
                        num_blocks,
                        item[2],
                        hit_summary.get(item[0], []),
                    ),
                    -item[1],
                    item[0],
                )
            )
            target = failed_gpus[0][0]
            route_info["prefix_hit"] = True
            route_info["reason"] = "prefix_hit_needs_rebalance"
            route_info["target_rank"] = target
            route_info["hit_summary"] = hit_summary
            route_info["matched_prefix_blocks"] = gpu_hit_count[target]
            route_info["prefix_owner_rank"] = target
            route_info["matched_prefix_hash"] = prefix_hashes[gpu_hit_count[target] - 1]
            route_info["required_new_blocks"] = self._required_new_blocks(
                num_blocks,
                gpu_hit_count[target],
            )
            route_info["failed_gpus"] = [(g, s) for g, s, _ in failed_gpus]
            route_info["load_score"] = self._load_summary(candidates)
            route_info["queue_pressure"] = self._queue_pressure_summary(candidates)
            self._annotate_target_capacity(
                route_info,
                target,
                route_info["required_new_blocks"],
                hit_summary.get(target, []),
            )
            logger.info(
                "route seq %s: tokens=%s blocks=%s prefix=%s hits=%s free=%s failed=%s "
                "-> GPU %s (reason=prefix_hit_needs_rebalance)",
                seq_id,
                num_tokens,
                num_blocks,
                prefix_hash,
                hit_summary,
                free_snapshot,
                [(g, s) for g, s, _ in failed_gpus],
                target,
            )
            return (target, route_info) if return_info else target

        # 兜底：只在本地 / NVLink 伙伴之间选一个空闲更多的 GPU
        target = self._select_best_candidate(
            rank,
            candidates,
            required_blocks=num_blocks,
            num_tokens=num_tokens,
            num_blocks=num_blocks,
        )
        route_info["reason"] = "fallback_most_free"
        route_info["target_rank"] = target
        route_info["required_new_blocks"] = num_blocks
        self._annotate_target_capacity(route_info, target, num_blocks)
        logger.info(
            "route seq %s: tokens=%s blocks=%s prefix=%s hits=%s free=%s -> GPU %s "
            "(reason=fallback_most_free)",
            seq_id, num_tokens, num_blocks, prefix_hash, hit_summary, free_snapshot, target,
        )
        return (target, route_info) if return_info else target

    def _free_snapshot(self, world_size: int) -> dict[int, int]:
        return {
            gpu_id: self.gbm.get_free_blocks_count(gpu_id)
            for gpu_id in range(world_size)
        }

    def _hit_summary(self, hits) -> dict[int, list[int]]:
        summary: dict[int, list[int]] = {}
        for loc in hits:
            summary.setdefault(loc.gpu_id, []).append(loc.block_id)
        return summary

    def _lookup_contiguous_prefix(
        self,
        prefix_hashes: List[int],
        requester_rank: int,
    ) -> dict[int, list[int]]:
        """Return per-GPU physical blocks for the longest contiguous hash chain."""
        contiguous: dict[int, list[int]] = {}
        active_gpus: set[int] | None = None
        for block_hash in prefix_hashes:
            locations = self.gbm.lookup_prefix(block_hash, requester_rank=requester_rank)
            by_gpu = {loc.gpu_id: loc.block_id for loc in locations}
            if active_gpus is None:
                active_gpus = set(by_gpu)
                for gpu_id in active_gpus:
                    contiguous[gpu_id] = [by_gpu[gpu_id]]
                if not active_gpus:
                    break
                continue

            active_gpus &= set(by_gpu)
            if not active_gpus:
                break
            for gpu_id in active_gpus:
                contiguous[gpu_id].append(by_gpu[gpu_id])
        return contiguous

    @staticmethod
    def _required_new_blocks(num_blocks: int, matched_prefix_blocks: int) -> int:
        return max(0, int(num_blocks) - int(matched_prefix_blocks))

    def _score_summary(self, rank: int, gpu_hit_count: dict[int, int]) -> dict[int, float]:
        return {
            gpu_id: self._route_score(rank, gpu_id, hit_count)
            for gpu_id, hit_count in gpu_hit_count.items()
        }

    def _queue_pressure_summary(self, candidates: List[int]) -> dict[int, float]:
        return {
            gpu_id: self.gbm.get_queue_pressure(gpu_id)
            for gpu_id in candidates
        }

    def _load_summary(self, candidates: List[int]) -> dict[int, float]:
        return {
            gpu_id: self._load_score(gpu_id)
            for gpu_id in candidates
        }

    def _load_score(self, gpu_id: int) -> float:
        return self.gbm.get_load_score(
            gpu_id,
            self.waiting_token_weight,
            self.running_token_weight,
            self.running_sequence_weight,
        )

    def _route_score(self, my_rank: int, gpu_id: int, hit_count: int) -> float:
        prefix_score = hit_count * self._get_topo_weight(my_rank, gpu_id) * self.prefix_hit_weight
        queue_penalty = self._load_score(gpu_id) * self.load_weight
        capacity_bonus = self.gbm.get_effective_capacity(gpu_id) * self.free_block_weight
        return prefix_score - queue_penalty + capacity_bonus

    def _route_cost_components(
        self,
        gpu_id: int,
        num_tokens: int,
        num_blocks: int,
        hit_count: int,
        protected_block_ids: Optional[List[int]] = None,
    ) -> dict[str, float]:
        """Estimate completion work in token-equivalent units."""
        required_blocks = self._required_new_blocks(num_blocks, hit_count)
        missing_tokens = min(
            max(0, int(num_tokens)),
            required_blocks * max(1, int(self.block_size)),
        )
        free_blocks = self.gbm.get_free_blocks_count(gpu_id)
        reclaim_blocks = max(0, required_blocks - free_blocks)
        components = {
            "queue": self._load_score(gpu_id),
            "prefill": float(missing_tokens) * self.prefill_cost_weight,
            "reclaim": (
                float(reclaim_blocks)
                * float(max(1, int(self.block_size)))
                * self.reclaim_cost_weight
            ),
        }
        components["total"] = sum(components.values())
        return components

    def _route_cost(
        self,
        gpu_id: int,
        num_tokens: int,
        num_blocks: int,
        hit_count: int,
        protected_block_ids: Optional[List[int]] = None,
    ) -> float:
        return self._route_cost_components(
            gpu_id,
            num_tokens,
            num_blocks,
            hit_count,
            protected_block_ids,
        )["total"]

    def _route_cost_summary(
        self,
        candidates: List[int],
        num_tokens: int,
        num_blocks: int,
        gpu_hit_count: dict[int, int],
        hit_summary: dict[int, list[int]],
    ) -> dict[int, dict[str, float]]:
        return {
            gpu_id: self._route_cost_components(
                gpu_id,
                num_tokens,
                num_blocks,
                gpu_hit_count.get(gpu_id, 0),
                hit_summary.get(gpu_id, []),
            )
            for gpu_id in candidates
        }

    def _annotate_target_capacity(
        self,
        route_info: dict,
        target_gpu: int,
        required_blocks: int,
        protected_block_ids: Optional[List[int]] = None,
    ) -> None:
        free_blocks = self.gbm.get_free_blocks_count(target_gpu)
        reclaimable_blocks = self.gbm.get_reclaimable_blocks_count(
            target_gpu,
            protected_block_ids,
        )
        route_info["target_free_blocks"] = free_blocks
        route_info["target_reclaimable_blocks"] = reclaimable_blocks
        route_info["target_effective_capacity"] = self.gbm.get_effective_capacity(
            target_gpu,
            protected_block_ids,
        )
        route_info["uses_reclaimable_capacity"] = required_blocks > free_blocks

    def _compute_prefix_hashes(self, seq: Sequence) -> List[int]:
        """
        计算序列的前缀 hash
        使用 BlockManager 的 compute_hash 方法，只 hash 完整的块（不含 partial 尾块）。
        
        返回:
            hash 值，如果序列没有完整块则返回 None
        """
        full_blocks = int(seq.num_tokens // seq.block_size)
        if full_blocks == 0:
            return []
        if self.block_manager is None:
            raise RuntimeError("GlobalScheduler.block_manager is required to compute prefix hashes")

        # 只取完整块的部分做 hash
        hash_val = -1
        hashes = []
        for i in range(full_blocks):
            block_tokens = seq.token_ids[i * seq.block_size : (i + 1) * seq.block_size]
            hash_val = self.block_manager.compute_hash(block_tokens, hash_val)
            hashes.append(hash_val)
        return hashes

    def _compute_prefix_hash(self, seq: Sequence) -> Optional[int]:
        hashes = self._compute_prefix_hashes(seq)
        return hashes[-1] if hashes else None

    def _get_topo_weight(self, my_rank: int, target_gpu: int) -> float:
        """
        计算拓扑权重
        - 同 GPU: 2.0
        - NVLink 直连: 1.0
        - 其他 GPU: 0.0
        """
        if my_rank < 0:
            # launcher / ingress 侧没有“本地 GPU”概念；此时只做全局 prefix / free-space 决策
            return 1.0
        if target_gpu == my_rank:
            return 2.0
        partner = self.gbm._get_nvlink_partner(my_rank)
        if partner is not None and partner == target_gpu:
            return 1.0
        return 0.0

    def _candidate_gpus(self, my_rank: int) -> List[int]:
        if my_rank < 0:
            candidates = [
                gpu_id
                for gpu_id in range(self.gbm.world_size)
                if self.gbm.is_gpu_available(gpu_id)
            ]
            if not candidates:
                raise RuntimeError("no healthy GPU is available for routing")
            return candidates
        candidates = [my_rank] if self.gbm.is_gpu_available(my_rank) else []
        partner = self.gbm._get_nvlink_partner(my_rank)
        if (
            partner is not None
            and partner != my_rank
            and self.gbm.is_gpu_available(partner)
        ):
            candidates.append(partner)
        if not candidates:
            raise RuntimeError("no healthy GPU is available for routing")
        return candidates

    def _select_best_candidate(
        self,
        my_rank: int,
        candidates: List[int],
        required_blocks: int = 0,
        num_tokens: int = 0,
        num_blocks: int = 0,
    ) -> int:
        """Select the minimum-cost candidate with enough effective capacity."""
        allocatable = [
            gpu_id
            for gpu_id in candidates
            if self.gbm.can_allocate_effective(gpu_id, required_blocks)
        ]
        considered = allocatable or candidates
        return min(
            considered,
            key=lambda gpu_id: (
                self._route_cost(gpu_id, num_tokens, num_blocks, 0),
                -self.gbm.get_effective_capacity(gpu_id),
                0 if gpu_id == my_rank and my_rank >= 0 else 1,
                gpu_id,
            ),
        )

    def _candidate_key(self, my_rank: int, gpu_id: int) -> tuple[float, int, int]:
        return (
            -self._load_score(gpu_id),
            self.gbm.get_effective_capacity(gpu_id),
            1 if gpu_id == my_rank and my_rank >= 0 else 0,
        )

    # ------------------------------------------------------------------
    # 显存重平衡
    # ------------------------------------------------------------------

    def plan_rebalance(
        self,
        gpu_id: int,
        needed_blocks: int,
        allow_copy: bool = False,
        excluded_source_blocks: set[int] | None = None,
    ) -> dict | None:
        """
        生成一个可执行的 rebalance 计划，但不直接修改控制面的权威状态。

        返回:
            {
                "gpu_id": gpu_id,
                "needed_blocks": needed_blocks,
                "transfers": [
                    {
                        "src_gpu": gpu_id,
                        "dst_gpu": target_gpu,
                        "src_blocks": [...],
                        "hashes": [...],
                    },
                    ...
                ],
            }
            或在无法满足时返回 None。
        """
        self.last_rebalance_fail_reason = ""
        target_order = self.gbm._get_target_gpu_order(gpu_id)
        if not target_order:
            self.last_rebalance_fail_reason = "no_plan"
            return None
        if not self.gbm.block_hash[gpu_id]:
            self.last_rebalance_fail_reason = "no_plan"
            return None
        if all(self.gbm.get_free_blocks_count(target) <= 0 for target in target_order):
            self.last_rebalance_fail_reason = "no_target_space"
            return None

        excluded = set(excluded_source_blocks or ())
        candidates, release_blocks = self._select_chain_move_candidates(
            gpu_id,
            needed_blocks,
            target_order,
            excluded,
        )
        mode = "move"
        if allow_copy and (not candidates or len(release_blocks) < needed_blocks):
            candidates = self._select_copy_candidates(
                gpu_id,
                needed_blocks,
                target_order,
                excluded_source_blocks=excluded,
            )
            mode = "copy"
            release_blocks = []
        valid_move = mode == "move" and candidates and len(release_blocks) >= needed_blocks
        valid_copy = mode == "copy" and len(candidates) >= needed_blocks
        if not (valid_move or valid_copy):
            self.last_rebalance_fail_reason = "no_plan"
            return None

        actual_candidates = candidates
        grouped: dict[int, list[int]] = {}
        for local_block, target_gpu in actual_candidates:
            grouped.setdefault(target_gpu, []).append(local_block)

        transfers = []
        release_target = actual_candidates[0][1] if actual_candidates else None
        for target_gpu, blocks in grouped.items():
            hashes = []
            parent_hashes = []
            access_counts = []
            generations = []
            for block_id in blocks:
                block_hash = self.gbm.get_block_hash(gpu_id, block_id)
                hashes.append(block_hash if block_hash is not None else -1)
                parent_hashes.append(self.gbm.get_block_parent_hash(gpu_id, block_id))
                access_counts.append(self.gbm.block_access_count[gpu_id].get(block_id, 1))
                generations.append(self.gbm.get_block_generation(gpu_id, block_id))
            transfers.append({
                "src_gpu": gpu_id,
                "dst_gpu": target_gpu,
                "src_blocks": blocks,
                "hashes": hashes,
                "parent_hashes": parent_hashes,
                "access_counts": access_counts,
                "generations": generations,
                "mode": "chain_move" if mode == "move" else mode,
                "release_source_blocks": (
                    list(release_blocks) if target_gpu == release_target else []
                ),
                "release_source_hashes": (
                    [
                        self.gbm.get_block_hash(gpu_id, block_id)
                        for block_id in release_blocks
                    ]
                    if target_gpu == release_target else []
                ),
                "release_source_generations": (
                    [
                        self.gbm.get_block_generation(gpu_id, block_id)
                        for block_id in release_blocks
                    ]
                    if target_gpu == release_target else []
                ),
            })

        plan = {
            "gpu_id": gpu_id,
            "needed_blocks": needed_blocks,
            "mode": "chain_move" if mode == "move" else mode,
            "transfers": transfers,
        }
        transferred_blocks = sum(len(item["src_blocks"]) for item in transfers)
        transfer_bytes = self._estimate_transfer_bytes(transferred_blocks)
        transfer_cost_ms = sum(
            self._estimate_transfer_cost_ms(
                self._estimate_transfer_bytes(len(item["src_blocks"])),
                gpu_id,
                item["dst_gpu"],
            )
            for item in transfers
        )

        # A chain's blocks share the same future request. Summing every block's
        # historical access count overstates demand by roughly chain length.
        # The least-frequent transferred block is the conservative leaf demand;
        # remove the already-observed access and discount history into expected
        # future reuse.
        observed_chain_accesses = min(
            (
                max(1, int(count))
                for item in transfers
                for count in item["access_counts"]
            ),
            default=1,
        )
        predicted_reuses = max(0.0, observed_chain_accesses - 1) * self.future_reuse_discount
        saved_prefill_ms = (
            predicted_reuses
            * transferred_blocks
            * max(1, int(self.block_size))
            * self.prefill_token_time_ms
        )
        plan["estimated_transfer_bytes"] = transfer_bytes
        plan["transfer_pairs"] = sorted({
            tuple(sorted((int(item["src_gpu"]), int(item["dst_gpu"]))))
            for item in transfers
        })
        plan["estimated_future_reuses"] = predicted_reuses
        plan["estimated_transfer_cost_ms"] = transfer_cost_ms
        plan["estimated_saved_prefill_ms"] = saved_prefill_ms
        # Legacy field names remain in the protocol for older result readers,
        # but their values are now milliseconds instead of token equivalents.
        plan["estimated_transfer_cost"] = transfer_cost_ms
        plan["estimated_saved_prefill"] = saved_prefill_ms
        plan["estimated_benefit_ratio"] = saved_prefill_ms / max(transfer_cost_ms, 1e-9)
        if saved_prefill_ms < transfer_cost_ms * self.foreground_transfer_min_benefit_ratio:
            self.last_rebalance_fail_reason = "low_benefit"
            return None
        return plan

    def _estimate_transfer_bytes(self, num_blocks: int) -> int:
        return (
            max(0, int(num_blocks))
            * 2
            * max(1, int(self.num_layers))
            * max(1, int(self.block_size))
            * max(1, int(self.num_kv_heads))
            * max(1, int(self.head_dim))
            * max(1, int(self.kv_dtype_bytes))
        )

    @staticmethod
    def _pair_key(src_gpu: int, dst_gpu: int) -> tuple[int, int]:
        return tuple(sorted((int(src_gpu), int(dst_gpu))))

    def _estimate_transfer_cost_ms(
        self,
        transfer_bytes: int,
        src_gpu: int | None = None,
        dst_gpu: int | None = None,
    ) -> float:
        bandwidth_bytes_s = max(self.transfer_bandwidth_gib_s, 1e-6) * (1024 ** 3)
        wire_ms = max(0, int(transfer_bytes)) / bandwidth_bytes_s * 1000.0
        static_cost_ms = (
            self.transfer_fixed_latency_ms
            + wire_ms * self.transfer_interference_multiplier
        ) * self.transfer_cost_weight
        observed_extra_ms = self.observed_transfer_extra_ms
        if src_gpu is not None and dst_gpu is not None:
            observed_extra_ms = self.observed_transfer_extra_ms_by_pair.get(
                self._pair_key(src_gpu, dst_gpu),
                None,
            )
        observed_cost_ms = (
            (wire_ms + observed_extra_ms) * self.transfer_cost_weight
            if observed_extra_ms is not None
            else 0.0
        )
        placement_extra_ms = None
        if src_gpu is not None and dst_gpu is not None:
            placement_extra_ms = self.observed_placement_extra_ms_by_pair.get(
                self._pair_key(src_gpu, dst_gpu)
            )
        placement_cost_ms = (
            (wire_ms + placement_extra_ms) * self.transfer_cost_weight
            if placement_extra_ms is not None
            else 0.0
        )
        return max(static_cost_ms, observed_cost_ms, placement_cost_ms)

    def observe_transfer(
        self,
        transfer_bytes: int,
        elapsed_s: float,
        src_gpu: int | None = None,
        dst_gpu: int | None = None,
    ) -> None:
        """Update online transfer overhead from a completed source operation."""
        if transfer_bytes <= 0 or elapsed_s <= 0:
            return
        bandwidth_bytes_s = max(self.transfer_bandwidth_gib_s, 1e-6) * (1024 ** 3)
        wire_ms = transfer_bytes / bandwidth_bytes_s * 1000.0
        observed_extra_ms = max(0.0, elapsed_s * 1000.0 - wire_ms)
        alpha = min(1.0, max(0.0, self.transfer_cost_ewma_alpha))
        if self.observed_transfer_extra_ms is None:
            self.observed_transfer_extra_ms = observed_extra_ms
        else:
            self.observed_transfer_extra_ms = (
                alpha * observed_extra_ms
                + (1.0 - alpha) * self.observed_transfer_extra_ms
            )
        if src_gpu is not None and dst_gpu is not None:
            pair = self._pair_key(src_gpu, dst_gpu)
            previous = self.observed_transfer_extra_ms_by_pair.get(pair)
            self.observed_transfer_extra_ms_by_pair[pair] = (
                observed_extra_ms
                if previous is None
                else alpha * observed_extra_ms + (1.0 - alpha) * previous
            )

    def observe_placement(
        self,
        transfer_bytes: int,
        elapsed_s: float,
        src_gpu: int,
        dst_gpu: int,
    ) -> None:
        """Learn plan cost from dispatch through destination commit."""
        if transfer_bytes <= 0 or elapsed_s <= 0:
            return
        bandwidth_bytes_s = max(self.transfer_bandwidth_gib_s, 1e-6) * (1024 ** 3)
        wire_ms = transfer_bytes / bandwidth_bytes_s * 1000.0
        observed_extra_ms = max(0.0, elapsed_s * 1000.0 - wire_ms)
        pair = self._pair_key(src_gpu, dst_gpu)
        alpha = min(1.0, max(0.0, self.transfer_cost_ewma_alpha))
        # The first dispatch-to-commit sample includes process wake-up and
        # allocator cold-start jitter. Blend it with the calibrated static
        # prior instead of allowing one outlier to reject the rest of a
        # forecast batch.
        static_extra_ms = max(
            0.0,
            self.transfer_fixed_latency_ms
            + wire_ms * (self.transfer_interference_multiplier - 1.0),
        )
        previous = self.observed_placement_extra_ms_by_pair.get(
            pair,
            static_extra_ms,
        )
        self.observed_placement_extra_ms_by_pair[pair] = (
            alpha * observed_extra_ms + (1.0 - alpha) * previous
        )

    def observe_prefill(
        self,
        gpu_id: int,
        uncached_tokens: int,
        elapsed_s: float,
    ) -> None:
        """Learn conservative marginal prefill cost from completed batches."""
        if uncached_tokens <= 0 or elapsed_s <= 0:
            return
        sample_ms = (
            elapsed_s * 1000.0 / uncached_tokens * self.prefill_observation_discount
        )
        sample_ms = min(1.0, max(0.001, sample_ms))
        previous = self.observed_prefill_token_time_ms_by_gpu.get(int(gpu_id))
        alpha = min(1.0, max(0.0, self.prefill_observation_ewma_alpha))
        self.observed_prefill_token_time_ms_by_gpu[int(gpu_id)] = (
            sample_ms
            if previous is None
            else alpha * sample_ms + (1.0 - alpha) * previous
        )

    def estimate_prefill_token_time_ms(self, gpu_id: int) -> float:
        return max(
            self.prefill_token_time_ms,
            self.observed_prefill_token_time_ms_by_gpu.get(
                int(gpu_id), self.prefill_token_time_ms
            ),
        )

    def _select_chain_move_candidates(
        self,
        gpu_id: int,
        needed_blocks: int,
        target_order: List[int],
        excluded_source_blocks: set[int],
    ) -> tuple[List[Tuple[int, int]], List[int]]:
        """Plan valuable complete chains and a dependency-safe release suffix."""
        target_free = {target: self.gbm.get_free_blocks_count(target) for target in target_order}
        planned_hashes = {
            target: set(self.gbm.block_hash[target].values()) for target in target_order
        }
        candidates: List[Tuple[int, int]] = []
        selected_blocks: set[int] = set()
        block_depth: dict[int, int] = {}
        target_hashes = planned_hashes[target_order[0]]

        def transfer_utility(block_id: int) -> tuple[float, float, int]:
            chain = self.gbm.get_prefix_chain(gpu_id, block_id)
            if not chain:
                return (0.0, 0.0, block_id)
            frequency = max(
                self.gbm.block_access_count[gpu_id].get(chain_block, 1)
                for chain_block in chain
            )
            missing = sum(
                self.gbm.get_block_hash(gpu_id, chain_block) not in target_hashes
                for chain_block in chain
            )
            utility = frequency * len(chain) / max(missing, 1)
            recency = max(
                self.gbm.block_access_time[gpu_id].get(chain_block, 0.0)
                for chain_block in chain
            )
            return (utility, recency, -block_id)

        leaves = sorted(
            self.gbm.block_access_time[gpu_id],
            key=transfer_utility,
            reverse=True,
        )
        for leaf_block in leaves:
            if leaf_block in excluded_source_blocks:
                continue
            chain = self.gbm.get_prefix_chain(gpu_id, leaf_block)
            # Ready full-prefix ancestors are immutable. They may therefore be
            # copied as dependencies even while referenced locally; the
            # release planner below still forbids freeing pinned blocks. Blocks
            # reserved by another transfer remain excluded from both reading
            # and releasing to avoid overlapping plans.
            if not chain or any(
                block_id in excluded_source_blocks for block_id in chain
            ):
                continue
            for target in target_order:
                missing = [
                    block_id for block_id in chain
                    if self.gbm.get_block_hash(gpu_id, block_id) not in planned_hashes[target]
                ]
                if leaf_block not in missing or target_free[target] < len(missing):
                    continue
                candidates.extend((block_id, target) for block_id in missing)
                selected_blocks.update(chain)
                for depth, block_id in enumerate(chain):
                    block_depth[block_id] = max(block_depth.get(block_id, -1), depth)
                for block_id in missing:
                    planned_hashes[target].add(self.gbm.get_block_hash(gpu_id, block_id))
                target_free[target] -= len(missing)
                break
            release_blocks = self._dependency_safe_release_order(
                gpu_id,
                selected_blocks,
                block_depth,
                excluded_source_blocks,
            )
            # One selected prefix chain is already transferred as one packed
            # batch. Stop once the shortage is covered; extending into another
            # colder chain would spend bandwidth without proven reuse benefit.
            if len(release_blocks) >= needed_blocks:
                break
        release_blocks = self._dependency_safe_release_order(
            gpu_id,
            selected_blocks,
            block_depth,
            excluded_source_blocks,
        )
        if len(release_blocks) < needed_blocks:
            return [], []
        # Release only the requested shortage. The deepest-first prefix is
        # always safe while avoiding unnecessary loss of source locality.
        return candidates, release_blocks[:needed_blocks]

    def _dependency_safe_release_order(
        self,
        gpu_id: int,
        selected_blocks: set[int],
        block_depth: dict[int, int],
        excluded_source_blocks: set[int],
    ) -> List[int]:
        """Return selected blocks releasable deepest-first without orphaning children."""
        hash_to_block = {
            block_hash: block_id
            for block_id, block_hash in self.gbm.block_hash[gpu_id].items()
            if block_hash != -1
        }
        children: dict[int, set[int]] = {
            block_id: set() for block_id in self.gbm.block_hash[gpu_id]
        }
        for child_id, parent_hash in self.gbm.block_parent_hash[gpu_id].items():
            parent_id = hash_to_block.get(parent_hash)
            if parent_id is not None:
                children.setdefault(parent_id, set()).add(child_id)

        memo: dict[int, bool] = {}

        def can_release(block_id: int) -> bool:
            if block_id in memo:
                return memo[block_id]
            if (
                block_id not in selected_blocks
                or block_id in excluded_source_blocks
                or block_id in self.gbm.pinned_block_ids[gpu_id]
            ):
                memo[block_id] = False
                return False
            memo[block_id] = all(can_release(child) for child in children.get(block_id, ()))
            return memo[block_id]

        releasable = [block_id for block_id in selected_blocks if can_release(block_id)]
        return sorted(
            releasable,
            key=lambda block_id: (-block_depth.get(block_id, 0), block_id),
        )

    def _select_copy_candidates(
        self,
        gpu_id: int,
        needed_blocks: int,
        target_order: List[int],
        excluded_source_blocks: set[int] | None = None,
    ) -> List[Tuple[int, int]]:
        """Select pinned blocks for replicated transfer when move eviction is impossible."""
        if not target_order:
            return []
        target_free = {target: self.gbm.get_free_blocks_count(target) for target in target_order}
        pinned_blocks = [
            block_id
            for block_id in self.gbm.block_hash[gpu_id]
            if (
                block_id not in self.gbm.block_access_time[gpu_id]
                and block_id not in set(excluded_source_blocks or ())
            )
        ]
        if not pinned_blocks:
            return []
        candidates: List[Tuple[int, int]] = []
        for block_id in pinned_blocks:
            for target in target_order:
                if target_free[target] <= 0:
                    continue
                block_hash = self.gbm.get_block_hash(gpu_id, block_id)
                if block_hash is not None and any(
                    loc.gpu_id == target for loc in self.gbm.get_block_location(block_hash)
                ):
                    continue
                candidates.append((block_id, target))
                target_free[target] -= 1
                break
            if len(candidates) >= needed_blocks:
                break
        return candidates

    def rebalance(self, gpu_id: int, needed_blocks: int) -> bool:
        """
        当 gpu_id 需要 needed_blocks 个空闲块但本地不足时调用

        流程:
        1. 调用 gbm.select_eviction_candidates 获取换出方案
        2. 逐对执行 transfer out
        3. 更新受影响的序列 block_table
        4. 通知目标 GPU 的 GlobalBlockManager 更新页表

        返回:
            是否成功腾出至少 needed_blocks 个空闲块
        """
        rank = dist.get_rank()

        # 1. 获取驱逐候选
        candidates = self.gbm.select_eviction_candidates(gpu_id, needed_blocks)

        if not candidates:
            return False

        # 2. 检查是否足够
        # 每个 candidate 释放 gpu_id 上的一个块，所以 candidates 长度应 >= needed_blocks
        if len(candidates) < needed_blocks:
            return False

        actual_candidates = candidates[:needed_blocks]

        # 3. 执行 transfer
        # 按 target_gpu 分组，一次 NCCL 操作处理同一目标的批量块
        groups: dict[int, List[int]] = {}
        for local_block, target_gpu in actual_candidates:
            if target_gpu not in groups:
                groups[target_gpu] = []
            groups[target_gpu].append(local_block)

        for target_gpu, blocks in groups.items():
            if rank == gpu_id and target_gpu != rank:
                self._execute_swap_out(blocks, gpu_id, target_gpu)
            elif rank == target_gpu and gpu_id != rank:
                self._execute_swap_in_accept(blocks, gpu_id, target_gpu)

        # 4. 更新本地空闲块计数和页表
        for local_block, target_gpu in actual_candidates:
            # 释放本地块
            self.gbm.free_global(gpu_id, [local_block])
            # 目标 GPU 上减去一个空闲块（由 record_block_transfer 处理）
            # 这里由上层 GlobalBlockManager.record_block_transfer 统一更新

        return True

    # def _execute_swap_out(
    #     self,
    #     blocks: List[int],
    #     local_gpu: int,
    #     target_gpu: int,
    # ):
    #     """
    #     在源 GPU 上执行 transfer out
    #     直接调用 kv_transfer 的 send 逻辑
    #     """
    #     from lmpool.engine.kv_transfer import _send_block_list, _compute_tag
    #     import time

    #     device = f"cuda:{local_gpu}"
    #     # 这里需要 kv_cache 的引用，由外部 ModelRunner 提供
    #     # 暂时留空，由实际调用方注入
    #     raise NotImplementedError(
    #         "transfer out 需要 kv_cache 张量引用，请在 ModelRunner 中调用 "
    #         "kv_transfer 完成实际数据传输"
    #     )

    # def _execute_swap_in_accept(
    #     self,
    #     blocks: List[int],
    #     source_gpu: int,
    #     local_gpu: int,
    # ):
    #     """
    #     在目标 GPU 上接收 transfer 数据
    #     """
    #     from lmpool.engine.kv_transfer import _recv_block_list
    #     raise NotImplementedError(
    #         "transfer in 需要 kv_cache 张量引用，请在 ModelRunner 中调用 "
    #         "kv_transfer 完成实际数据传输"
    #     )

    def _execute_swap_out(self, blocks, local_gpu, target_gpu):
        if self.model_runner is not None:
            self.model_runner.execute_swap_out(blocks, target_gpu)

    def _execute_swap_in_accept(self, blocks, source_gpu, local_gpu):
        if self.model_runner is not None:
            self.model_runner.execute_swap_in(source_gpu, blocks)
    
    # ------------------------------------------------------------------
    # 抢占回退
    # ------------------------------------------------------------------

    def preempt_for_rebalance(
        self,
        running_sequences: list,
        gpu_id: int,
        needed_blocks: int,
    ) -> bool:
        """
        当 transfer 无法满足需求时，选择序列回退到 WAITING 状态
        
        策略：
        选择最短的 running 序列，释放其所有块，直到满足 needed_blocks。
        
        返回:
            是否成功腾出足够空间
        """
        freed = 0
        victims = []

        for seq in running_sequences:
            if freed >= needed_blocks:
                break
            victims.append(seq)
            freed += len(seq.block_table)

        if freed < needed_blocks:
            return False

        for seq in victims:
            self.gbm.free_global(gpu_id, seq.block_table)
            seq.block_table = []
            seq.status = 2  # WAITING
            seq.num_cached_tokens = 0

        return True
