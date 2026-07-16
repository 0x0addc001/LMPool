"""
全局块管理器 (Global Block Manager)

负责跨 GPU 的 KV cache 块状态同步、全局 LRU 冷块选择和 transfer 目标决策
设计要点：
1. 全局页表：hash -> 该 hash 的块分布在哪些 GPU 上（一对多）
2. NVLink 拓扑感知的 transfer 目标选择：只考虑 NVLink 直连伙伴
3. 三级内存池枯竭应对：递归 transfer -> 远端 LRU 覆盖 -> CPU fallback signal
"""

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class BlockLocation:
    """描述一个 KV 块在集群中的物理位置"""
    gpu_id: int              # 块所在的 GPU rank
    block_id: int            # 该 GPU 上的物理块索引
    hash: int                # 块内容的 hash 值（-1 表示 partial block / 未 hash）
    last_access_time: float  # 最近访问时间戳（用于 LRU 驱逐决策）


# ============================================================================
# 全局块管理器
# ============================================================================


def detect_nvlink_pairs_from_nvidia_smi(world_size: int) -> List[Tuple[int, int]]:
    """
    Best-effort NVLink topology detection from `nvidia-smi topo -m`.

    Returns logical GPU index pairs in the current process ordering. If the
    topology cannot be parsed, returns an empty list.
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        return []

    header = None
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.lstrip()
        if "CPU Affinity" in stripped and header is None:
            cols = stripped.split()
            header = cols
        elif stripped.startswith("GPU") and "CPU Affinity" not in stripped:
            cols = stripped.split()
            rows.append(cols)

    if not header or not rows:
        return []

    gpu_cols = [col for col in header if col.startswith("GPU")]
    if len(gpu_cols) < 2:
        return []

    def parse_gpu_index(token: str) -> Optional[int]:
        if not token.startswith("GPU"):
            return None
        try:
            return int(token[3:])
        except ValueError:
            return None

    pairs: set[Tuple[int, int]] = set()
    for row in rows:
        if not row:
            continue
        src = parse_gpu_index(row[0])
        if src is None or src >= world_size:
            continue
        for idx, cell in enumerate(row[1 : 1 + len(gpu_cols)]):
            dst = idx
            if dst >= world_size or src >= dst:
                continue
            cell = cell.upper()
            if cell in {"NVL", "NVLINK", "NV1", "NV2"} or "NV" in cell:
                pairs.add((src, dst))
    return sorted(pairs)


class GlobalBlockManager:
    """
    集群内所有 GPU 的 KV cache 块全局管理器。

    职责：
    - 维护全局页表 (hash -> 所有 BlockLocation)
    - 维护每 GPU 空闲块数量
    - 提供拓扑感知的冷块驱逐选择
    - 记录块迁移后的位置变更
    - 支持全局管理节点可配置
    """

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(
        self,
        rank: int,
        world_size: int,
        num_blocks_per_gpu: int,
        master_rank: int = 0,
        nvlink_pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        参数:
            rank: 当前进程的 GPU rank
            world_size: GPU 总数
            num_blocks_per_gpu: 每 GPU 的物理块总数
            master_rank: 全局管理节点的 rank（默认 0）
            nvlink_pairs: NVLink 直连 GPU 对列表，如 [(0,2), (1,3), (4,5), (6,7)]
        """
        self.rank = rank
        self.world_size = world_size
        self.num_blocks_per_gpu = num_blocks_per_gpu
        self.master_rank = master_rank

        # ---------- 拓扑信息 ----------
        if nvlink_pairs is None:
            nvlink_pairs = detect_nvlink_pairs_from_nvidia_smi(world_size)
        self.nvlink_pairs = set(nvlink_pairs or [])

        # 建立 GPU -> NVLink 伙伴的快速查找
        self.nvlink_partner: Dict[int, int] = {}
        for a, b in self.nvlink_pairs:
            self.nvlink_partner[a] = b
            self.nvlink_partner[b] = a

        # ---------- 全局状态（仅 master_rank 维护权威副本） ----------
        # 全局页表：hash -> 该 hash 所在的所有位置
        self.global_page_table: Dict[int, List[BlockLocation]] = {}
        # 每 GPU 的空闲块计数
        self.free_blocks_per_gpu: List[int] = [num_blocks_per_gpu] * world_size
        # 每 GPU 的已用块访问时间：gpu_id -> {block_id: last_access_time}
        self.block_access_time: List[Dict[int, float]] = [{} for _ in range(world_size)]
        # 每 GPU 的块 hash 记录：gpu_id -> {block_id: hash}
        self.block_hash: List[Dict[int, int]] = [{} for _ in range(world_size)]
        # 每 GPU 的调度队列快照，由 data-plane worker 随 block_state 上报
        self.waiting_sequences_per_gpu: List[int] = [0] * world_size
        self.running_sequences_per_gpu: List[int] = [0] * world_size
        self.waiting_tokens_per_gpu: List[int] = [0] * world_size
        self.running_tokens_per_gpu: List[int] = [0] * world_size
        # Requests routed by the control plane but not yet admitted by a
        # worker. Keep these separate because block_state is an older worker
        # snapshot and must not erase control-plane admission reservations.
        self.pending_sequences_per_gpu: List[int] = [0] * world_size
        self.pending_tokens_per_gpu: List[int] = [0] * world_size
        self.pending_route_tokens: List[Dict[int, int]] = [{} for _ in range(world_size)]

    @property
    def is_master(self) -> bool:
        return self.rank == self.master_rank

    def update_gpu_state(
        self,
        gpu_id: int,
        free_blocks: int,
        block_hashes: Dict[int, int],
        evictable_block_hashes: Optional[Dict[int, int]] = None,
        pinned_block_hashes: Optional[Dict[int, int]] = None,
        waiting_sequences: int = 0,
        running_sequences: int = 0,
        waiting_tokens: int = 0,
        running_tokens: int = 0,
    ):
        """
        Master-only state ingestion boundary.

        Each worker owns the real local BlockManager/KV cache for its GPU and
        reports a compact snapshot through the engine message queue. Rank 0 is
        the authoritative GlobalBlockManager master for routing decisions.
        """
        if not self.is_master:
            return

        now = time.time()
        self.free_blocks_per_gpu[gpu_id] = free_blocks
        self.waiting_sequences_per_gpu[gpu_id] = max(0, int(waiting_sequences))
        self.running_sequences_per_gpu[gpu_id] = max(0, int(running_sequences))
        self.waiting_tokens_per_gpu[gpu_id] = max(0, int(waiting_tokens))
        self.running_tokens_per_gpu[gpu_id] = max(0, int(running_tokens))

        old_hashes = self.block_hash[gpu_id]
        for block_id, old_hash in list(old_hashes.items()):
            if old_hash in self.global_page_table:
                self.global_page_table[old_hash] = [
                    loc for loc in self.global_page_table[old_hash]
                    if not (loc.gpu_id == gpu_id and loc.block_id == block_id)
                ]
                if not self.global_page_table[old_hash]:
                    del self.global_page_table[old_hash]

        self.block_hash[gpu_id] = dict(block_hashes)
        evictable = block_hashes if evictable_block_hashes is None else evictable_block_hashes
        self.block_access_time[gpu_id] = {block_id: now for block_id in evictable}

        for block_id, block_hash in block_hashes.items():
            if block_hash == -1:
                continue
            self.global_page_table.setdefault(block_hash, []).append(
                BlockLocation(gpu_id, block_id, block_hash, now)
            )

    def reserve_blocks(self, gpu_id: int, num_blocks: int):
        """
        Master-only optimistic reservation for requests routed to a worker.

        The worker later reports the exact local BlockManager state through
        update_gpu_state(), which overwrites this estimate.
        """
        if not self.is_master:
            return
        self.free_blocks_per_gpu[gpu_id] = max(
            0,
            self.free_blocks_per_gpu[gpu_id] - num_blocks,
        )

    def reserve_route_load(
        self,
        gpu_id: int,
        num_tokens: int,
        num_sequences: int = 1,
        seq_id: Optional[int] = None,
    ):
        """
        Optimistically account for routed requests before the worker reports
        them in its next block-state snapshot.
        """
        if not self.is_master:
            return
        if seq_id is not None:
            if seq_id in self.pending_route_tokens[gpu_id]:
                return
            self.pending_route_tokens[gpu_id][seq_id] = max(0, int(num_tokens))
        self.pending_sequences_per_gpu[gpu_id] += max(0, int(num_sequences))
        self.pending_tokens_per_gpu[gpu_id] += max(0, int(num_tokens))

    def acknowledge_route_load(
        self,
        gpu_id: int,
        num_tokens: int,
        num_sequences: int = 1,
        seq_id: Optional[int] = None,
    ):
        """Remove an optimistic reservation after the worker admits it."""
        if not self.is_master:
            return
        if seq_id is not None:
            reserved_tokens = self.pending_route_tokens[gpu_id].pop(seq_id, None)
            if reserved_tokens is None:
                return
            num_tokens = reserved_tokens
        self.pending_sequences_per_gpu[gpu_id] = max(
            0,
            self.pending_sequences_per_gpu[gpu_id] - max(0, int(num_sequences)),
        )
        self.pending_tokens_per_gpu[gpu_id] = max(
            0,
            self.pending_tokens_per_gpu[gpu_id] - max(0, int(num_tokens)),
        )

    # ------------------------------------------------------------------
    # 拓扑辅助函数
    # ------------------------------------------------------------------

    def _get_nvlink_partner(self, gpu_id: int) -> Optional[int]:
        """返回 NVLink 直连对端 GPU ID，无则返回 None"""
        return self.nvlink_partner.get(gpu_id)

    def _get_target_gpu_order(self, gpu_id: int) -> List[int]:
        """
        返回 transfer 目标 GPU 的优先级顺序：
        1. NVLink 直连伙伴（如果有的话）
        """
        partner = self._get_nvlink_partner(gpu_id)
        if partner is None:
            return []
        return [partner]

    # ------------------------------------------------------------------
    # 前缀查找
    # ------------------------------------------------------------------

    def lookup_prefix(self, prefix_hash: int, requester_rank: Optional[int] = None) -> List[BlockLocation]:
        """
        查询拥有指定前缀的块分布在哪些 GPU 上。
        返回按「NVLink 亲和性 + 命中数量」降序排列的 BlockLocation 列表。
        
        排序权重：
        - NVLink 伙伴 GPU 上的块优先（权重 × 2）
        - 其他 GPU（权重 × 1.0）
        """
        if prefix_hash not in self.global_page_table:
            return []

        locations = self.global_page_table[prefix_hash]
        rank = self.rank if requester_rank is None else requester_rank

        def sort_key(loc: BlockLocation) -> float:
            score = 1.0
            if self._get_nvlink_partner(loc.gpu_id) == rank:
                score = 2.0
            return -score  # 负号使得高权重排在前面

        return sorted(locations, key=sort_key)

    # ------------------------------------------------------------------
    # 空闲块检查
    # ------------------------------------------------------------------

    def can_allocate_global(self, gpu_id: int, num_blocks: int) -> bool:
        """
        检查指定 GPU 是否有足够空闲块。
        如果不够，返回 False，由上层调用 select_eviction_candidates + transfer。
        """
        return self.free_blocks_per_gpu[gpu_id] >= num_blocks

    def get_free_blocks_count(self, gpu_id: int) -> int:
        """获取指定 GPU 当前空闲块数量"""
        return self.free_blocks_per_gpu[gpu_id]

    def get_queue_pressure(self, gpu_id: int) -> float:
        """
        Return a lightweight queue pressure estimate for routing.

        Running sequences are weighted higher because they already occupy decode
        slots and KV blocks; waiting sequences mostly represent admission delay.
        """
        return (
            float(self.waiting_sequences_per_gpu[gpu_id])
            + float(self.pending_sequences_per_gpu[gpu_id])
            + 2.0 * float(self.running_sequences_per_gpu[gpu_id])
        )

    def get_load_score(
        self,
        gpu_id: int,
        waiting_token_weight: float = 1.0,
        running_token_weight: float = 0.25,
        running_sequence_weight: float = 32.0,
    ) -> float:
        """
        Token-aware load estimate used by global routing.

        Waiting prompt tokens represent queued prefill work. Running tokens are
        discounted because decode advances one token at a time, while running
        sequence count captures scheduler/attention occupancy.
        """
        return (
            waiting_token_weight * float(
                self.waiting_tokens_per_gpu[gpu_id] + self.pending_tokens_per_gpu[gpu_id]
            )
            + running_token_weight * float(self.running_tokens_per_gpu[gpu_id])
            + running_sequence_weight * float(
                self.running_sequences_per_gpu[gpu_id] + self.pending_sequences_per_gpu[gpu_id]
            )
        )

    def get_global_free_blocks_count(self) -> int:
        """获取集群总空闲块数量"""
        return sum(self.free_blocks_per_gpu)

    # ------------------------------------------------------------------
    # 全局分配
    # ------------------------------------------------------------------

    def allocate_global(
        self,
        gpu_id: int,
        num_blocks: int,
        block_hashes: List[int],
    ) -> Optional[List[int]]:
        """
        在指定 GPU 上分配 num_blocks 个块，注册到全局页表。
        
        参数:
            gpu_id: 目标 GPU rank
            num_blocks: 需要分配的块数量
            block_hashes: 每个块的 hash 值（长度 == num_blocks）
        
        返回:
            分配的本地 block_id 列表；如果全局都无法满足则返回 None
        """
        if self.free_blocks_per_gpu[gpu_id] >= num_blocks:
            # 本地空闲充足：直接预分配 block_id
            start_id = self.num_blocks_per_gpu - self.free_blocks_per_gpu[gpu_id]
            new_blocks = list(range(start_id, start_id + num_blocks))
            self._commit_alloc(gpu_id, new_blocks, block_hashes)
            return new_blocks
        else:
            # 本地不足：返回 None，由上层协调 transfer
            return None

    def _commit_alloc(self, gpu_id: int, block_ids: List[int], hashes: List[int]):
        """提交分配：更新空闲计数、访问时间、全局页表"""
        now = time.time()
        for bid, h in zip(block_ids, hashes):
            self.free_blocks_per_gpu[gpu_id] -= 1
            self.block_access_time[gpu_id][bid] = now
            self.block_hash[gpu_id][bid] = h
            self.global_page_table.setdefault(h, []).append(
                BlockLocation(gpu_id, bid, h, now)
            )

    # ------------------------------------------------------------------
    # 释放
    # ------------------------------------------------------------------

    def free_global(self, gpu_id: int, block_ids: List[int]):
        """释放指定 GPU 上的块，从全局页表中移除"""
        for bid in block_ids:
            # 增加空闲计数
            self.free_blocks_per_gpu[gpu_id] += 1
            # 从访问时间中移除
            self.block_access_time[gpu_id].pop(bid, None)
            # 从全局页表中移除
            h = self.block_hash[gpu_id].pop(bid, None)
            if h is not None and h in self.global_page_table:
                self.global_page_table[h] = [
                    loc for loc in self.global_page_table[h]
                    if not (loc.gpu_id == gpu_id and loc.block_id == bid)
                ]
                if not self.global_page_table[h]:
                    del self.global_page_table[h]

    # ------------------------------------------------------------------
    # 冷块驱逐选择
    # ------------------------------------------------------------------

    def select_eviction_candidates(
        self,
        gpu_id: int,
        num_blocks: int,
        allow_recursive: bool = True,
    ) -> List[Tuple[int, int]]:
        """
        从指定 GPU 选择 num_blocks 个 LRU 冷块作为 transfer 候选
        
        transfer 目标选择策略：
        1. 只考虑 NVLink 直连伙伴作为目标 GPU
        2. 若伙伴无空闲块，allow_recursive=True 时尝试在伙伴上递归驱逐冷块
        3. 若伙伴也无块可驱逐，或可执行计划禁用递归驱逐，则返回空列表

        注意：
            当前 data plane 的 rebalance 执行器只支持“源 GPU -> 目标 GPU 空闲块”
            的直接搬运，尚未实现目标侧 victim block 的二阶段递归驱逐。因此控制面
            生成可执行计划时应传 allow_recursive=False，避免计划容量超过目标 rank
            实际可 reserve 的空闲块数。
        
        返回:
            [(local_block_id_to_evict, target_gpu_id), ...]
        """
        # 1. 选出本地最冷的 num_blocks 个块
        if not self.block_access_time[gpu_id]:
            return []

        sorted_blocks = sorted(
            self.block_access_time[gpu_id].items(),
            key=lambda kv: kv[1]  # 按 last_access_time 升序（最老的优先驱逐）
        )
        cold_blocks = [bid for bid, _ in sorted_blocks[:num_blocks]]

        candidates = []
        target_free = {gpu: self.free_blocks_per_gpu[gpu] for gpu in range(self.world_size)}

        # 2. 为目标 GPU 排序（仅 NVLink 直连伙伴）
        target_order = self._get_target_gpu_order(gpu_id)
        if not target_order:
            return []

        # 3. 对每个冷块找目标
        for block_id in cold_blocks:
            placed = False
            for target in target_order:
                if target_free[target] > 0:
                    candidates.append((block_id, target))
                    # 临时扣除目标 GPU 的空闲块（用于后续块的计算）
                    target_free[target] -= 1
                    placed = True
                    break
            if not placed:
                # 伙伴 GPU 都没空闲块：尝试递归驱逐伙伴上的冷块
                if not allow_recursive:
                    return []
                target = target_order[0]  # 拓扑最近的目标
                victim_block = self._select_remote_victim(target)
                if victim_block is None:
                    return []
                self.block_access_time[target].pop(victim_block, None)
                h = self.block_hash[target].pop(victim_block, None)
                if h is not None and h in self.global_page_table:
                    self.global_page_table[h] = [
                        loc for loc in self.global_page_table[h]
                        if loc.block_id != victim_block
                    ]
                target_free[target] += 1
                candidates.append((block_id, target))

        return candidates

    def _select_remote_victim(self, gpu_id: int) -> Optional[int]:
        """在指定 GPU 上选出一个 LRU 最冷的块作为二级驱逐候选"""
        if not self.block_access_time[gpu_id]:
            return None
        return min(self.block_access_time[gpu_id], key=self.block_access_time[gpu_id].get)

    # ------------------------------------------------------------------
    # 块迁移记录
    # ------------------------------------------------------------------

    def record_block_transfer(
        self,
        block_id: int,
        src_gpu: int,
        dst_gpu: int,
        new_block_id: Optional[int] = None,
    ):
        """
        块迁移后更新全局页表。

        参数:
            block_id: 原始 block_id（源 GPU 上）
            src_gpu: 源 GPU rank
            dst_gpu: 目标 GPU rank
            new_block_id: 目标 GPU 上的新 block_id（默认与源相同）
        """
        if new_block_id is None:
            new_block_id = block_id

        now = time.time()

        # 从源 GPU 移除
        self.free_blocks_per_gpu[src_gpu] += 1
        self.block_access_time[src_gpu].pop(block_id, None)
        h = self.block_hash[src_gpu].pop(block_id, None)

        if h is not None and h in self.global_page_table:
            self.global_page_table[h] = [
                loc for loc in self.global_page_table[h]
                if not (loc.gpu_id == src_gpu and loc.block_id == block_id)
            ]
            if not self.global_page_table[h]:
                del self.global_page_table[h]

        # 在目标 GPU 上注册
        self.free_blocks_per_gpu[dst_gpu] -= 1
        self.block_access_time[dst_gpu][new_block_id] = now
        if h is not None:
            self.block_hash[dst_gpu][new_block_id] = h
            self.global_page_table.setdefault(h, []).append(
                BlockLocation(dst_gpu, new_block_id, h, now)
            )

    def record_block_copy(
        self,
        block_id: int,
        src_gpu: int,
        dst_gpu: int,
        new_block_id: int,
    ):
        """
        复制式 transfer 后更新全局页表。

        与 record_block_transfer 不同，copy 不释放源 GPU 的块，只在目标 GPU
        注册一个新的副本位置。
        """
        now = time.time()
        h = self.block_hash[src_gpu].get(block_id)
        self.free_blocks_per_gpu[dst_gpu] -= 1
        self.block_access_time[dst_gpu][new_block_id] = now
        if h is not None:
            self.block_hash[dst_gpu][new_block_id] = h
            self.global_page_table.setdefault(h, []).append(
                BlockLocation(dst_gpu, new_block_id, h, now)
            )

    def get_block_hash(self, gpu_id: int, block_id: int) -> Optional[int]:
        """获取指定块的内容 hash"""
        return self.block_hash[gpu_id].get(block_id)

    def get_block_location(self, hash_val: int) -> List[BlockLocation]:
        """通过 hash 查找块的所有位置"""
        return self.global_page_table.get(hash_val, [])

    def touch_block(self, gpu_id: int, block_id: int):
        """更新块的访问时间（用于 LRU）"""
        self.block_access_time[gpu_id][block_id] = time.time()
