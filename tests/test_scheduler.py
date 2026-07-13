from types import SimpleNamespace

import torch.distributed as dist

from lmpool.engine.block_manager import BlockManager
from lmpool.engine.global_block_manager import GlobalBlockManager
from lmpool.engine.global_scheduler import GlobalScheduler
from lmpool.engine.scheduler import Scheduler
from lmpool.engine.sequence import Sequence, SequenceStatus


class DummyGlobalScheduler:
    def __init__(self, gbm, rebalance_result=True):
        self.gbm = gbm
        self.rebalance_result = rebalance_result
        self.rebalance_calls = []

    def rebalance(self, gpu_id, needed_blocks):
        self.rebalance_calls.append((gpu_id, needed_blocks))
        return self.rebalance_result


def test_prefill_routes_remote_sequences_without_local_allocation():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=4, nvlink_pairs=[(0, 1)])
    dummy = DummyGlobalScheduler(gbm)
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=16,
        max_cached_blocks=4,
        block_size=2,
        eos=999,
        global_scheduler=dummy,
    )

    seq = Sequence([1, 2, 3, 4], block_size=2)
    seq.remote_gpu_id = 1
    scheduler.add_sequence(seq)

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill is True
    assert scheduled == [seq]
    assert len(scheduler.waiting) == 0
    assert dummy.gbm.get_free_blocks_count(1) == 2


def test_prefill_allocates_locally_and_updates_running():
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=16,
        max_cached_blocks=4,
        block_size=2,
        eos=999,
        global_scheduler=None,
    )
    seq = Sequence([1, 2, 3, 4], block_size=2)
    scheduler.add_sequence(seq)

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill is True
    assert scheduled == [seq]
    assert seq.status == SequenceStatus.RUNNING
    assert list(scheduler.running) == [seq]


def test_decode_rebalance_prevents_preemption(monkeypatch):
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=2, nvlink_pairs=[(0, 1)])
    dummy = DummyGlobalScheduler(gbm)
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=16,
        max_cached_blocks=2,
        block_size=2,
        eos=999,
        global_scheduler=dummy,
    )
    seq = Sequence([1, 2], block_size=2)
    scheduler.block_manager.allocate(seq)
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)
    scheduler.block_manager.free_block_ids.clear()

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill is False
    assert scheduled == []
    assert dummy.rebalance_calls == [(0, 1)]
    assert list(scheduler.running) == [seq]


def test_prefill_rebalance_failure_preempts_running_sequence():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=1, nvlink_pairs=[(0, 1)])
    dummy = DummyGlobalScheduler(gbm, rebalance_result=False)
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=16,
        max_cached_blocks=1,
        block_size=2,
        eos=999,
        global_scheduler=dummy,
    )
    running_seq = Sequence([1, 2], block_size=2)
    scheduler.block_manager.allocate(running_seq)
    running_seq.status = SequenceStatus.RUNNING
    scheduler.running.append(running_seq)

    waiting_seq = Sequence([3, 4], block_size=2)
    scheduler.add_sequence(waiting_seq)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is False
    assert scheduled == []
    assert dummy.rebalance_calls == [(0, 1)]
    assert running_seq in scheduler.waiting
    assert waiting_seq in scheduler.waiting


def test_prefill_rebalance_failure_does_not_preempt_current_batch():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=2, nvlink_pairs=[(0, 1)])
    dummy = DummyGlobalScheduler(gbm, rebalance_result=False)
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=16,
        max_cached_blocks=2,
        block_size=2,
        eos=999,
        global_scheduler=dummy,
    )
    first = Sequence([1, 2], block_size=2)
    second = Sequence([3, 4, 5], block_size=2)
    scheduler.add_sequence(first)
    scheduler.add_sequence(second)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert scheduled == [first]
    assert len(first.block_table) == first.num_blocks
    assert first in scheduler.running
    assert second in scheduler.waiting
    assert dummy.rebalance_calls == [(0, 1)]


def test_prefill_capacity_uses_cached_prefix_blocks():
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=16,
        max_cached_blocks=2,
        block_size=2,
        eos=999,
        global_scheduler=None,
    )
    cached = Sequence([1, 2], block_size=2)
    scheduler.block_manager.allocate(cached)
    cached.status = SequenceStatus.RUNNING
    scheduler.running.append(cached)

    seq = Sequence([1, 2, 3, 4], block_size=2)
    scheduler.add_sequence(seq)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert scheduled == [seq]
    assert seq.num_cached_tokens == 2


def test_prefill_can_disable_foreground_rebalance():
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=1, nvlink_pairs=[(0, 1)])
    dummy = DummyGlobalScheduler(gbm, rebalance_result=True)
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=16,
        max_cached_blocks=1,
        block_size=2,
        eos=999,
        global_scheduler=dummy,
    )
    scheduler.enable_foreground_rebalance = False

    running_seq = Sequence([1, 2], block_size=2)
    scheduler.block_manager.allocate(running_seq)
    running_seq.status = SequenceStatus.RUNNING
    scheduler.running.append(running_seq)

    waiting_seq = Sequence([3, 4], block_size=2)
    scheduler.add_sequence(waiting_seq)

    scheduled, is_prefill = scheduler.schedule()

    assert scheduled == []
    assert is_prefill is False
    assert dummy.rebalance_calls == []


def test_postprocess_finishes_and_requeues_non_finished():
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=16,
        max_cached_blocks=4,
        block_size=2,
        eos=999,
        global_scheduler=None,
    )
    seq_done = Sequence([1, 2], block_size=2, sampling_params=SimpleNamespace(
        temperature=1.0, max_tokens=1, ignore_eos=False, max_model_length=None
    ))
    seq_waiting = Sequence([3, 4], block_size=2, sampling_params=SimpleNamespace(
        temperature=1.0, max_tokens=4, ignore_eos=True, max_model_length=None
    ))
    scheduler.block_manager.allocate(seq_done)
    scheduler.block_manager.allocate(seq_waiting)
    scheduler.running.extend([seq_done, seq_waiting])
    seq_waiting.status = SequenceStatus.WAITING

    scheduler.postprocess([seq_done, seq_waiting], [999, 111])

    assert seq_done.is_finished is True
    assert seq_waiting.status == SequenceStatus.RUNNING
    assert seq_waiting in scheduler.running
