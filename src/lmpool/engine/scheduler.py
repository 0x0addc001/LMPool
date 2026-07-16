from collections import deque
import time
import torch.distributed as dist
from lmpool.engine.sequence import Sequence, SequenceStatus
from lmpool.engine.block_manager import BlockManager
from lmpool.engine.global_scheduler import GlobalScheduler


class Scheduler:
    """
    本地调度器

    扩展功能（当启用全局池化时）：
    - prefill 阶段：由入口控制面决定序列归属 GPU
    - decode 阶段：本地空闲不足时，通过 GlobalScheduler.rebalance 触发跨 GPU transfer
    - 序列可能被标记为远程前缀（pending_swap_in 非空，legacy internal name），后续由 ModelRunner 拉取
    """

    def __init__(
        self,
        max_num_sequences: int,
        max_num_batched_tokens: int,
        max_cached_blocks: int,
        block_size: int,
        eos: int,
        global_scheduler: GlobalScheduler = None,
    ):
        gbm = global_scheduler.gbm if global_scheduler is not None else None
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size, gbm=gbm)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos
        self._rebalance_cooldown_until: dict[tuple, float] = {}
        self._rebalance_cooldown_s = 0.25
        self.enable_foreground_rebalance = True
        self.foreground_transfer_min_blocks = 1
        self.preemption_count = 0

        # --------------------------------------------- #
        # 全局调度器接口
        # --------------------------------------------- #
        self.global_scheduler = global_scheduler
        if self.global_scheduler is not None:
            self.global_scheduler.block_manager = self.block_manager
        
        # 当前 rank（用于路由判断）
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def _rebalance_on_cooldown(self, key: tuple) -> bool:
        return time.monotonic() < self._rebalance_cooldown_until.get(key, 0.0)

    def _mark_rebalance_failed(self, key: tuple) -> None:
        self._rebalance_cooldown_until[key] = time.monotonic() + self._rebalance_cooldown_s

    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0

    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)

    def _decode_reserve_blocks(self, incoming: Sequence | None = None) -> int:
        # Reserve the next growth block, not the entire completion, so long
        # generations do not serialize admission unnecessarily.
        reserve = sum(min(1, seq.remaining_decode_blocks) for seq in self.running)
        if incoming is not None:
            reserve += min(1, incoming.remaining_decode_blocks)
        return reserve

    def _can_admit_prefill(self, seq: Sequence) -> bool:
        required = self.block_manager.num_required_new_blocks(seq)
        return len(self.block_manager.free_block_ids) >= required + self._decode_reserve_blocks(seq)

    def _sync_local_state_to_global(self):
        if self.global_scheduler is None:
            return
        if hasattr(self.global_scheduler, "report_block_state"):
            self.global_scheduler.report_block_state(
                len(self.block_manager.free_block_ids),
                self.block_manager.get_local_block_hashes(),
                self.block_manager.get_evictable_block_hashes(),
                self.block_manager.get_pinned_block_hashes(),
                len(self.waiting),
                len(self.running),
                sum(len(seq) for seq in self.waiting),
                sum(len(seq) for seq in self.running),
            )
            return
        gbm = self.global_scheduler.gbm
        if gbm is None or not gbm.is_master:
            return
        gbm.update_gpu_state(
            self.rank,
            len(self.block_manager.free_block_ids),
            self.block_manager.get_local_block_hashes(),
        )


    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度

        返回:
            (scheduled_sequences, is_prefill)
            - scheduled_sequences: 本轮需要执行的序列列表
            - is_prefill: True 表示 prefill 阶段，False 表示 decode 阶段
        """
        scheduled_sequences = []
        current_scheduled_tokens = 0

        # ================================================================
        # Prefill 阶段
        # ================================================================
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]

            if len(seq) + current_scheduled_tokens > self.max_num_batched_tokens:
                break

            # -------------------------------------------------------- #
            # 全局路由决策
            # -------------------------------------------------------- #
            if self.global_scheduler is not None:
                if seq.remote_gpu_id == self.rank:
                    target_gpu = self.rank
                elif seq.remote_gpu_id != -1:
                    target_gpu = seq.remote_gpu_id
                else:
                    target_gpu = self.rank
                seq.remote_gpu_id = target_gpu if target_gpu != self.rank else -1

                # 如果序列路由到其他 GPU，从本地 waiting 移除并发送过去
                if target_gpu != self.rank:
                    # self.waiting.popleft()
                    # # 标记为远程前缀（如果命中了远程块）
                    # if seq.remote_gpu_id != -1:
                    #     seq.is_remote_prefix = True
                    # # 发送到目标 GPU（通过外部队列，这里只标记）
                    # # 实际发送逻辑在 LLMEngine 中处理
                    # continue
                    seq = self.waiting.popleft()
                    seq.status = SequenceStatus.RUNNING
                    scheduled_sequences.append(seq)       # 保留在调度列表里
                    current_scheduled_tokens += len(seq)
                    if self.global_scheduler.gbm is not None:
                        self.global_scheduler.gbm.reserve_blocks(target_gpu, seq.num_blocks)
                    # 不在这里分配 block，由目标 rank 自己分配
                    continue
            # -------------------------------------------------------- #

            # Admission includes enough headroom for every active sequence to
            # finish its configured decode without immediately preempting one
            # another at the next block boundary.
            can_alloc = self._can_admit_prefill(seq)

            if not can_alloc:
                required_new_blocks = self.block_manager.num_required_new_blocks(seq)
                reserve_blocks = self._decode_reserve_blocks(seq)
                shortage = max(
                    0,
                    required_new_blocks + reserve_blocks - len(self.block_manager.free_block_ids),
                )
                # Dropping a cold reconstructable cache entry is cheaper than
                # transferring it or preempting live decode work.
                self.block_manager.reclaim_for_sequence(seq, reserve_blocks=reserve_blocks)
                if self._can_admit_prefill(seq):
                    continue

                # The request itself fits, but admitting it would consume
                # decode headroom. Wait for running work instead of moving KV
                # merely to increase concurrency.
                if self.block_manager.can_allocate(seq):
                    break

                # Local cache reclamation was insufficient. The full LMPool
                # path may preserve cache value by moving evictable blocks.
                shortage = max(0, required_new_blocks - len(self.block_manager.free_block_ids))
                cooldown_key = ("capacity", self.rank)
                if (
                    self.global_scheduler is not None
                    and self.enable_foreground_rebalance
                    and shortage >= self.foreground_transfer_min_blocks
                ):
                    rebalance_success = False
                    if shortage > 0 and not self._rebalance_on_cooldown(cooldown_key):
                        rebalance_success = self.global_scheduler.rebalance(self.rank, shortage)
                        if not rebalance_success:
                            self._mark_rebalance_failed(cooldown_key)
                    if rebalance_success and self._can_admit_prefill(seq):
                        continue
                # Do not evict live decode work merely to admit new prefill.
                # Falling through to the decode loop drains current work and
                # naturally creates capacity for the waiting request.
                break

            # 正常分配
            seq = self.waiting.popleft()
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            current_scheduled_tokens += len(seq)
            scheduled_sequences.append(seq)
            self._sync_local_state_to_global()

        if scheduled_sequences:
            return scheduled_sequences, True

        # ================================================================
        # Decode 阶段
        # ================================================================
        while self.running:
            seq = self.running.popleft()

            # 检查是否可以追加一个 token
            if not self.block_manager.can_append(seq):
                if self.block_manager.reclaim_cached_blocks(1) > 0:
                    self.block_manager.append(seq)
                    scheduled_sequences.append(seq)
                    current_scheduled_tokens += 1
                    continue
                # ---------------------------------------------------- #
                # 全局 rebalance：尝试腾出 1 个块
                # ---------------------------------------------------- #
                rebalance_success = False
                if (
                    self.global_scheduler is not None
                    and self.enable_foreground_rebalance
                    and self.foreground_transfer_min_blocks <= 1
                ):
                    cooldown_key = ("capacity", self.rank)
                    if not self._rebalance_on_cooldown(cooldown_key):
                        rebalance_success = self.global_scheduler.rebalance(self.rank, 1)
                        if not rebalance_success:
                            self._mark_rebalance_failed(cooldown_key)

                if rebalance_success:
                    # rebalance 成功，把序列放回队首，下轮重试
                    self.running.appendleft(seq)
                    break

                # ---------------------------------------------------- #
                # rebalance 失败：原有抢占逻辑
                # ---------------------------------------------------- #
                if self.running:
                    self.running.appendleft(seq)
                    victim = self.running.pop()
                    self.preempt(victim, front=False)
                    if self.block_manager.can_append(seq):
                        self.running.popleft()
                        self.block_manager.append(seq)
                        scheduled_sequences.append(seq)
                        current_scheduled_tokens += 1
                else:
                    self.preempt(seq)
                break

            # 检查 token 预算
            if current_scheduled_tokens >= self.max_num_batched_tokens:
                self.running.appendleft(seq)
                break
            if len(scheduled_sequences) >= self.max_num_sequences:
                self.running.appendleft(seq)
                break

            # 追加一个 token
            self.block_manager.append(seq)
            scheduled_sequences.append(seq)
            current_scheduled_tokens += 1
            self._sync_local_state_to_global()

        # 把已调度的序列重新放回 running 队列末尾（保持轮转顺序）
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))

        return scheduled_sequences, False


    def preempt(self, seq: Sequence, front: bool = True) -> None:
        """抢占序列：释放块，放回 waiting 队首"""
        self.block_manager.deallocate(seq)
        self.preemption_count += 1
        seq.preemption_count += 1
        seq.status = SequenceStatus.WAITING
        seq.num_cached_tokens = 0
        seq.block_table = []
        # 清除远程前缀标记（抢占后重新调度可能换 GPU）
        seq.is_remote_prefix = False
        seq.remote_gpu_id = -1
        seq.pending_swap_in = []
        if front:
            self.waiting.appendleft(seq)
        else:
            self.waiting.append(seq)
        self._sync_local_state_to_global()


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        """
        生成后处理：追加 token，检查停止条件，释放已完成序列的块
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)

            # 检查停止条件
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = (
                seq.max_model_length is not None
                and seq.num_tokens >= seq.max_model_length
            )

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                self._sync_local_state_to_global()

            if seq.status == SequenceStatus.WAITING:
                # 进到这里说明该序列顺利完成了推理，且没有触发 FINISHED 结束条件
                # 意味着它刚刚完成了 Prefill 阶段，现在需要强制进入 Decode 状态
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)  # 塞进 running 队列，供下一轮 schedule() 挑出来做 DECODE
