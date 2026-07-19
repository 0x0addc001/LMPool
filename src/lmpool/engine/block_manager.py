import xxhash
import numpy as np
from collections import deque
from typing import Tuple, Optional
import torch
import torch.distributed as dist
import time

from lmpool.engine.sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        # Identifies reuse of the same physical block ID. Transfer plans carry
        # this value to prevent block-ID ABA from silently copying different KV.
        self.generation = 0
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []
        self.kv_ready = False
        self.transfer_locked = False
        self.transfer_pending_publish = False
        self.parent_hash = -1
        self.prefix_depth = -1
        self.last_access_time = time.monotonic()
        self.access_count = 0


    def update(
        self,
        h: int,
        token_ids: list[int],
        parent_hash: int = -1,
        prefix_depth: int = 0,
    ):
        self.hash = h 
        self.token_ids = token_ids
        self.parent_hash = parent_hash
        self.prefix_depth = prefix_depth
        self.last_access_time = time.monotonic()
        self.access_count = max(1, self.access_count)

    def reset(self):
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []
        self.kv_ready = False
        self.transfer_locked = False
        self.transfer_pending_publish = False
        self.parent_hash = -1
        self.prefix_depth = -1
        self.last_access_time = time.monotonic()
        self.access_count = 0

    def touch(self, timestamp: float | None = None, count_access: bool = True):
        self.last_access_time = time.monotonic() if timestamp is None else timestamp
        if count_access:
            self.access_count += 1


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, gbm=None):
        """
        参数:
            num_blocks: 每 GPU 物理块总数
            block_size: 每块容纳的 token 数
            gbm: GlobalBlockManager 实例（启用全局池化时用）
        """
        # block_size: number of tokens per block
        self.block_size: int = block_size
        # list of all blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash to block id: this is for prefix caching (local)
        self.hash_to_block_id: dict[int, int] = {}
        # free block ids
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used block ids
        self.used_block_ids: set[int] = set()

        # 全局块管理器引用（None 则为单卡模式）
        self.gbm = gbm

    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        elif isinstance(token_ids, (list, tuple)):
            token_ids = [t.item() if isinstance(t, torch.Tensor) else t for t in token_ids]
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    # move this block to used list
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        block.generation += 1
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        # 从本地 hash 映射中移除
        if block.hash != -1 and block.hash in self.hash_to_block_id:
            if self.hash_to_block_id[block.hash] == block_id:
                del self.hash_to_block_id[block.hash]
        block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # whether we can allocate a block for this sequence
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= self.num_required_new_blocks(seq)

    def num_required_new_blocks(self, seq: Sequence) -> int:
        """
        Return how many physical blocks must be newly allocated for seq.

        Cached full blocks already present in hash_to_block_id do not consume a
        free block. This matters for shared-prefix routing: requiring
        seq.num_blocks free blocks would reject requests that can mostly reuse
        local KV cache and would unnecessarily trigger foreground rebalance.
        """
        required = 0
        h = -1
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            if len(token_ids) != self.block_size:
                required += 1
                h = -1
                continue

            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].transfer_locked:
                required += 1
                continue

            cached_tokens = self.blocks[block_id].token_ids
            if cached_tokens is not None and cached_tokens != token_ids:
                required += 1
        return required

    def reclaim_cached_blocks(self, num_blocks: int, protected_block_ids: set[int] | None = None) -> int:
        """Release low-frequency prefix leaves without orphaning descendants.

        Recency breaks ties between equally frequent candidates, which keeps
        the policy deterministic while preserving repeatedly reused prefixes.
        """
        protected = protected_block_ids or set()
        remaining = {
            block_id
            for block_id in self.used_block_ids
            if self.blocks[block_id].hash != -1 and self.blocks[block_id].kv_ready
        }
        selected: list[Block] = []
        for _ in range(max(0, int(num_blocks))):
            parent_hashes = {
                self.blocks[block_id].parent_hash
                for block_id in remaining
                if self.blocks[block_id].parent_hash != -1
            }
            leaves = [
                self.blocks[block_id]
                for block_id in remaining
                if (
                    block_id not in protected
                    and self.blocks[block_id].ref_count == 0
                    and not self.blocks[block_id].transfer_locked
                    and self.blocks[block_id].hash not in parent_hashes
                )
            ]
            if not leaves:
                break
            victim = min(
                leaves,
                key=lambda block: (
                    block.access_count,
                    block.last_access_time,
                    -block.prefix_depth,
                    block.block_id,
                ),
            )
            selected.append(victim)
            remaining.remove(victim.block_id)

        for block in selected:
            self._deallocate_block(block.block_id)
        return len(selected)

    def reclaim_for_sequence(
        self,
        seq: Sequence,
        reserve_blocks: int = 0,
        protected_block_ids: set[int] | None = None,
    ) -> int:
        """Reclaim cold cache while preserving ``seq`` and decode headroom."""
        required = self.num_required_new_blocks(seq)
        shortage = max(0, required + max(0, reserve_blocks) - len(self.free_block_ids))
        if shortage == 0:
            return 0

        protected = set(protected_block_ids or ())
        h = -1
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            if len(token_ids) != self.block_size:
                break
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h)
            block_id = self.hash_to_block_id.get(h)
            if block_id is not None and not self.blocks[block_id].transfer_locked:
                protected.add(block_id)
        return self.reclaim_cached_blocks(shortage, protected)

    def resolve_cached_block_ids(self, block_hashes) -> set[int]:
        """Resolve currently resident, ready cache blocks for prefix hashes."""
        resolved = set()
        for block_hash in block_hashes:
            block_id = self.hash_to_block_id.get(int(block_hash))
            if block_id is None:
                continue
            block = self.blocks[block_id]
            if block.kv_ready and block.ref_count == 0 and not block.transfer_locked:
                resolved.add(block_id)
        return resolved


    def allocate(self, seq: Sequence) -> None:
        h = -1
        contiguous_prefix_hit = True
        for i in range(seq.num_blocks):
            no_cache_found = False

            token_ids = seq.block(i)
            # only compute hash for full blocks, always -1 for partial blocks
            parent_hash = h
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id != -1 and self.blocks[block_id].transfer_locked:
                block_id = -1
            
            # if cache miss or hash collision
            if block_id == -1:
                no_cache_found = True
            else:
                cached_tokens = self.blocks[block_id].token_ids
                if cached_tokens is not None and cached_tokens != token_ids:
                    no_cache_found = True

            if not no_cache_found:
                # update sequence information
                if contiguous_prefix_hit:
                    seq.num_cached_tokens += self.block_size  # which == len(token_ids)
                # update block information, considering the edge case that the block is not allocated yet but with hash code
                if block_id not in self.used_block_ids:
                    block = self._allocate_block(block_id)
                else:
                    # update block information
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1
                    block.touch()
            else:
                contiguous_prefix_hit = False
                # cache miss
                block = self._allocate_block(self.free_block_ids[0])
                block.ref_count = 1
                block.update(
                    h=h,
                    token_ids=token_ids,
                    parent_hash=parent_hash,
                    prefix_depth=i,
                )
            seq.block_table.append(block.block_id)

    def deallocate(self, seq: Sequence) -> None:
        # update block information
        release_time = time.monotonic()
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                if block.hash == -1 or not block.kv_ready:
                    self._deallocate_block(block_id)
                else:
                    # A complete KV block remains addressable as an evictable
                    # prefix-cache entry until capacity pressure reclaims it.
                    # One request access is one recency event for the whole
                    # chain. Equal timestamps let leaf/depth ordering preserve
                    # ancestors instead of evicting block zero first.
                    block.touch(release_time, count_access=False)
        # update sequence information
        seq.block_table = []
        seq.num_cached_tokens = 0

    # this is to check whether we can append tokens to this sequence
    # when that token would require allocating a new block.
    def can_append(self, seq: Sequence) -> bool:
        # Scheduler.postprocess() has already appended the sampled token before
        # the next decode schedule. A remainder of one therefore means that
        # this token starts a new logical block which is not in block_table yet.
        if seq.num_tokens % self.block_size == 1:
            return len(self.free_block_ids) > 0
        return True

    # this is the actual work to append tokens to this sequence
    # this is called when the new token has been added to the seq information
    # but no block in gpu has yet allocate for it
    def append(self, seq: Sequence) -> None:
        block_tables = seq.block_table
        last_block_for_seq_id = block_tables[-1]

        # if the last block is now full, compute hash
        if seq.num_tokens % self.block_size == 0:
            token_ids=seq.block(seq.num_blocks - 1)

            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()  # 转成 Python list

            h = self.compute_hash(
                token_ids=token_ids,
                prefix_hash_value=-1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash
            )
            block = self.blocks[last_block_for_seq_id]
            parent_hash = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash
            block.update(
                h=h,
                token_ids=seq.block(seq.num_blocks - 1),
                parent_hash=parent_hash,
                prefix_depth=len(block_tables) - 1,
            )

        # if one new block is needed
        elif seq.num_tokens % self.block_size == 1:
            # Previous block should be finalized
            assert self.blocks[last_block_for_seq_id].hash != -1
            if not self.free_block_ids:
                raise RuntimeError(
                    f"Cannot append seq {seq.seq_id}: a new KV block is required "
                    "but no free block is available"
                )
            block = self._allocate_block(self.free_block_ids[0])
            block.ref_count = 1
            block_tables.append(block.block_id)
        # else, do nothing
        else:
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"

    # ------------------------------------------------------------------
    # 全局 KV cache 池相关方法
    # ------------------------------------------------------------------

    def try_allocate_remote(self, seq: Sequence) -> Tuple[bool, int]:
        """
        检查是否可以通过远程前缀复用来分配块

        流程:
        1. 计算序列前缀 hash
        2. 调用 gbm.lookup_prefix 查找远程命中
        3. 若命中，标记 seq 的远程前缀信息，后续由 ModelRunner 做 transfer in

        返回:
            (是否有远程前缀命中, 命中块所在的 GPU rank)
        """
        if self.gbm is None:
            return False, -1

        # 计算前缀 hash（只取完整块）
        full_blocks = int(seq.num_tokens // self.block_size)
        if full_blocks == 0:
            return False, -1

        prefix_hash = -1
        for i in range(full_blocks):
            token_ids = seq.block(i)
            prefix_hash = self.compute_hash(token_ids, prefix_hash)

        # 查询全局页表
        rank = dist.get_rank() if dist.is_initialized() else 0
        hits = self.gbm.lookup_prefix(prefix_hash, requester_rank=rank)
        if not hits:
            return False, -1

        # 按 GPU 聚合命中块数
        gpu_hit_count: dict[int, list[int]] = {}
        for loc in hits:
            if loc.gpu_id not in gpu_hit_count:
                gpu_hit_count[loc.gpu_id] = []
            gpu_hit_count[loc.gpu_id].append(loc.block_id)

        if not gpu_hit_count:
            return False, -1

        # 选择命中块数最多的 GPU（优先本地的 NVLink 伙伴）
        best_gpu = -1
        best_count = 0

        for gpu_id, blocks in gpu_hit_count.items():
            count = len(blocks)
            # 拓扑加权
            if gpu_id == rank:
                count = count * 3
            elif self.gbm._get_nvlink_partner(rank) == gpu_id:
                count = count * 2

            if count > best_count:
                best_count = count
                best_gpu = gpu_id

        if best_gpu < 0:
            return False, -1

        # 标记序列的远程前缀信息
        seq.is_remote_prefix = True
        seq.remote_gpu_id = best_gpu
        # 记录需要 transfer in 的远端块（按 block table 顺序）
        seq.pending_swap_in = gpu_hit_count[best_gpu]

        return True, best_gpu

    def allocate_with_swap(self, seq: Sequence) -> bool:
        """
        当本地空闲不足时的分配入口

        流程:
        1. 检查本地空闲块是否够
        2. 不够则调用 gbm.select_eviction_candidates 获取换出候选
        3. 标记需要 transfer out 的块（实际传输由 ModelRunner 执行）
        4. 本地分配新块

        返回:
            是否成功分配（本地空闲现在够用）
        """
        if self.can_allocate(seq):
            self.allocate(seq)
            return True

        if self.gbm is None:
            return False

        needed = seq.num_blocks
        available = len(self.free_block_ids)
        shortage = needed - available

        # 获取驱逐候选
        rank = dist.get_rank()
        candidates = self.gbm.select_eviction_candidates(rank, shortage)

        if not candidates or len(candidates) < shortage:
            return False

        # 释放被选中的本地冷块（数据已由 transfer out 搬走）
        for local_block, target_gpu in candidates:
            # 从 used 移到 free（但不立即复用，等分配时再取）
            block = self.blocks[local_block]
            if block.ref_count == 0:
                self._deallocate_block(local_block)
            # 更新全局页表
            self.gbm.record_block_transfer(
                block_id=local_block,
                src_gpu=rank,
                dst_gpu=target_gpu,
            )

        # 现在应该有足够的空闲块了
        if not self.can_allocate(seq):
            return False

        self.allocate(seq)
        return True

    def append_with_swap(self, seq: Sequence) -> bool:
        """
        解码追加时空间不足的 transfer 版本

        流程同 allocate_with_swap（legacy internal name），但只分配 1 个块。

        返回:
            是否成功
        """
        if self.can_append(seq):
            self.append(seq)
            return True

        if self.gbm is None:
            return False

        # 只需要 1 个块
        rank = dist.get_rank()
        candidates = self.gbm.select_eviction_candidates(rank, 1)

        if not candidates:
            return False

        local_block, target_gpu = candidates[0]

        # 释放冷块
        block = self.blocks[local_block]
        if block.ref_count == 0:
            self._deallocate_block(local_block)

        # 更新全局页表
        self.gbm.record_block_transfer(
            block_id=local_block,
            src_gpu=rank,
            dst_gpu=target_gpu,
        )

        # 现在追加
        if not self.can_append(seq):
            return False

        self.append(seq)
        return True

    def reserve_free_blocks(self, num_blocks: int) -> list[int]:
        """
        预留指定数量的空闲块，返回被预留的 block_id 列表。

        这些块会从 free 集合移到 used 集合，但不会立刻绑定到某个
        Sequence。用于控制平面下的 transfer-in 目标块分配。
        """
        if len(self.free_block_ids) < num_blocks:
            raise RuntimeError(
                f"Not enough free blocks to reserve: need {num_blocks}, "
                f"have {len(self.free_block_ids)}"
            )
        reserved = []
        for _ in range(num_blocks):
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            reserved.append(block_id)
        return reserved

    def lock_transfer_blocks(self, block_ids: list[int]) -> None:
        """Prevent local reuse or reclamation while a transfer plan is active."""
        for block_id in block_ids:
            if block_id not in self.used_block_ids:
                raise RuntimeError(f"Cannot lock block {block_id}: block is not allocated")
            self.blocks[block_id].transfer_locked = True

    def unlock_transfer_blocks(self, block_ids: list[int]) -> None:
        """Release source-side transfer locks without changing KV ownership."""
        for block_id in block_ids:
            if block_id in self.used_block_ids:
                self.blocks[block_id].transfer_locked = False

    def validate_transfer_blocks(
        self,
        block_ids: list[int],
        hashes: list[int],
        generations: list[int],
    ) -> tuple[bool, str]:
        """Validate immutable source identities captured by the control plane."""
        if not (len(block_ids) == len(hashes) == len(generations)):
            return False, "transfer metadata length mismatch"
        for block_id, expected_hash, expected_generation in zip(
            block_ids, hashes, generations
        ):
            if block_id not in self.used_block_ids:
                return False, f"block {block_id} is no longer allocated"
            block = self.blocks[block_id]
            if block.generation != int(expected_generation):
                return False, (
                    f"block {block_id} generation changed: "
                    f"expected {expected_generation}, found {block.generation}"
                )
            if not block.kv_ready or block.hash != int(expected_hash):
                return False, (
                    f"block {block_id} identity changed: expected hash "
                    f"{expected_hash}, found {block.hash}, kv_ready={block.kv_ready}"
                )
        return True, ""

    def release_blocks(self, block_ids: list[int]) -> None:
        """
        释放指定块回空闲池。
        仅适用于 ref_count == 0 的块。
        """
        for block_id in block_ids:
            if block_id not in self.used_block_ids:
                raise RuntimeError(f"Cannot release block {block_id}: block is not allocated")
            block = self.blocks[block_id]
            if block.ref_count != 0:
                raise RuntimeError(f"Cannot release block {block_id}: ref_count={block.ref_count}")
            self._deallocate_block(block_id)

    def release_reserved_blocks(self, block_ids: list[int]) -> None:
        """
        释放 prepare 阶段预留但尚未完成 transfer-in 注册的块。
        """
        for block_id in block_ids:
            if block_id not in self.used_block_ids:
                continue
            block = self.blocks[block_id]
            if block.ref_count != 0:
                raise RuntimeError(f"Cannot release reserved block {block_id}: ref_count={block.ref_count}")
            self._deallocate_block(block_id)

    def register_swap_in_blocks(
        self,
        block_ids: list[int],
        hashes: list[int],
        parent_hashes: list[int] | None = None,
        access_counts: list[int] | None = None,
        publish: bool = True,
    ) -> None:
        """
        将已接收的 transfer-in 块登记到本地 hash 表。

        这一步不分配新的物理块，只更新 block.hash / hash_to_block_id，
        便于后续全局页表同步和前缀查找。
        """
        if len(block_ids) != len(hashes):
            raise ValueError("block_ids and hashes must have the same length")
        if parent_hashes is not None and len(parent_hashes) != len(hashes):
            raise ValueError("parent_hashes and hashes must have the same length")
        if access_counts is not None and len(access_counts) != len(hashes):
            raise ValueError("access_counts and hashes must have the same length")
        parents = parent_hashes or [-1] * len(hashes)
        counts = access_counts or [1] * len(hashes)
        for index, (block_id, h, parent_hash, access_count) in enumerate(
            zip(block_ids, hashes, parents, counts)
        ):
            block = self.blocks[block_id]
            block.hash = h
            block.kv_ready = True
            block.parent_hash = parent_hash
            block.prefix_depth = index
            block.access_count = max(1, int(access_count))
            # transfer-in blocks already contain valid KV data but do not carry
            # original token ids. Treat token_ids=None as a trusted hash match.
            block.token_ids = None
            block.transfer_pending_publish = not publish
            block.transfer_locked = not publish
            if publish and h != -1:
                self.hash_to_block_id[h] = block_id

    def publish_transfer_blocks(self, block_ids: list[int]) -> None:
        """Expose received KV blocks only after the control plane commits."""
        for block_id in block_ids:
            if block_id not in self.used_block_ids:
                raise RuntimeError(f"Cannot publish block {block_id}: block is not allocated")
            block = self.blocks[block_id]
            if not block.kv_ready:
                raise RuntimeError(f"Cannot publish block {block_id}: KV is not ready")
            block.transfer_pending_publish = False
            block.transfer_locked = False
            if block.hash != -1:
                self.hash_to_block_id[block.hash] = block_id

    def mark_kv_ready(self, seqs: list[Sequence]) -> None:
        """Publish complete block hashes only after model execution wrote KV."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        for seq in seqs:
            for block_id in seq.block_table:
                block = self.blocks[block_id]
                if block.hash == -1 or block.kv_ready:
                    continue
                block.kv_ready = True
                self.hash_to_block_id[block.hash] = block_id
                if self.gbm is not None:
                    self.gbm._commit_alloc(rank, [block_id], [block.hash])

    def get_local_block_hashes(self) -> dict[int, int]:
        """
        获取本地所有已用块的 (block_id -> hash) 映射
        用于向 GlobalBlockManager 上报本地状态
        """
        return {
            bid: self.blocks[bid].hash
            for bid in self.used_block_ids
            if self.blocks[bid].kv_ready and not self.blocks[bid].transfer_pending_publish
        }

    def get_local_block_generations(self) -> dict[int, int]:
        """Return generations for ready blocks included in the local snapshot."""
        return {
            bid: self.blocks[bid].generation
            for bid in self.used_block_ids
            if self.blocks[bid].kv_ready and not self.blocks[bid].transfer_pending_publish
        }

    def get_local_block_parent_hashes(self) -> dict[int, int]:
        """Return parent hashes for ready blocks in the local prefix DAG."""
        return {
            bid: self.blocks[bid].parent_hash
            for bid in self.used_block_ids
            if self.blocks[bid].kv_ready and not self.blocks[bid].transfer_pending_publish
        }

    def get_local_block_access_stats(self) -> dict[int, dict[str, float | int]]:
        """Return worker-owned recency and frequency metadata for ready blocks."""
        return {
            bid: {
                "last_access_time": self.blocks[bid].last_access_time,
                "access_count": self.blocks[bid].access_count,
            }
            for bid in self.used_block_ids
            if self.blocks[bid].kv_ready and not self.blocks[bid].transfer_pending_publish
        }

    def get_evictable_block_hashes(self) -> dict[int, int]:
        """
        获取本地可 move-evict 的块。

        ref_count > 0 的块仍被序列引用，只能作为复制式 migration 的来源，
        不能作为释放源端空间的 eviction victim。
        """
        ready_parent_hashes = {
            self.blocks[bid].parent_hash
            for bid in self.used_block_ids
            if self.blocks[bid].kv_ready and self.blocks[bid].parent_hash != -1
        }
        return {
            bid: self.blocks[bid].hash
            for bid in self.used_block_ids
            if (
                self.blocks[bid].ref_count == 0
                and self.blocks[bid].kv_ready
                and not self.blocks[bid].transfer_locked
                and not self.blocks[bid].transfer_pending_publish
                and self.blocks[bid].hash not in ready_parent_hashes
            )
        }

    def get_pinned_block_hashes(self) -> dict[int, int]:
        """获取仍被引用、不可释放的块，用于诊断和未来复制式 migration。"""
        return {
            bid: self.blocks[bid].hash
            for bid in self.used_block_ids
            if (
                self.blocks[bid].kv_ready
                and not self.blocks[bid].transfer_pending_publish
                and (self.blocks[bid].ref_count > 0 or self.blocks[bid].transfer_locked)
            )
        }

    def sync_with_global(self):
        """
        将本地块状态推送到 GlobalBlockManager
        在管理节点上调用
        """
        if self.gbm is None or not self.gbm.is_master:
            return

        rank = dist.get_rank()
        for bid in self.used_block_ids:
            block = self.blocks[bid]
            if block.hash != -1:
                self.gbm.block_hash[rank][bid] = block.hash
                self.gbm.block_access_time[rank][bid] = __import__('time').time()
