from types import SimpleNamespace

import torch.distributed as dist

from lmpool.engine.block_manager import BlockManager
from lmpool.engine.global_block_manager import GlobalBlockManager
from lmpool.engine.global_scheduler import GlobalScheduler
from lmpool.engine.scheduler import Scheduler
from lmpool.engine.sequence import Sequence, SequenceStatus


class DummyGlobalScheduler:
    def __init__(self, gbm, rebalance_result=True, fail_reason="no_plan"):
        self.gbm = gbm
        self.rebalance_result = rebalance_result
        self.last_rebalance_fail_reason = fail_reason
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
    seq.append_token(3)
    seq.status = SequenceStatus.RUNNING
    scheduler.running.append(seq)
    scheduler.block_manager.free_block_ids.clear()

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill is False
    assert scheduled == []
    assert dummy.rebalance_calls == [(0, 1)]
    assert list(scheduler.running) == [seq]


def test_decode_batch_handles_two_sequences_crossing_boundary_with_one_free_block():
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=16,
        max_cached_blocks=3,
        block_size=2,
        eos=999,
        global_scheduler=None,
    )
    first = Sequence([1, 2], block_size=2)
    second = Sequence([3, 4], block_size=2)
    scheduler.block_manager.allocate(first)
    scheduler.block_manager.allocate(second)
    first.append_token(5)
    second.append_token(6)
    first.status = SequenceStatus.RUNNING
    second.status = SequenceStatus.RUNNING
    scheduler.running.extend([first, second])

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is False
    assert scheduled == [first]
    assert len(first.block_table) == 2
    assert second in scheduler.waiting
    assert scheduler.preemption_count == 1


def test_prefill_rebalance_failure_preserves_running_decode():
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
    running_seq = Sequence([1, 2], block_size=2)
    scheduler.block_manager.allocate(running_seq)
    running_seq.status = SequenceStatus.RUNNING
    scheduler.running.append(running_seq)

    waiting_seq = Sequence([3, 4, 5, 6], block_size=2)
    scheduler.add_sequence(waiting_seq)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is False
    assert scheduled == [running_seq]
    assert dummy.rebalance_calls == [(0, 1)]
    assert running_seq in scheduler.running
    assert waiting_seq in scheduler.waiting
    assert scheduler.preemption_count == 0


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
    no_decode = SimpleNamespace(
        temperature=1.0, max_tokens=0, ignore_eos=True, max_model_length=None
    )
    cached = Sequence([1, 2], block_size=2, sampling_params=no_decode)
    scheduler.block_manager.allocate(cached)
    scheduler.block_manager.mark_kv_ready([cached])
    cached.status = SequenceStatus.RUNNING
    scheduler.running.append(cached)

    seq = Sequence([1, 2, 3, 4], block_size=2, sampling_params=no_decode)
    scheduler.add_sequence(seq)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert scheduled == [seq]
    assert seq.num_cached_tokens == 2


def test_prefill_admission_reserves_next_decode_block():
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=16,
        max_cached_blocks=3,
        block_size=2,
        eos=999,
        global_scheduler=None,
    )
    params = SimpleNamespace(
        temperature=1.0, max_tokens=1, ignore_eos=True, max_model_length=None
    )
    running = Sequence([1, 2], block_size=2, sampling_params=params)
    scheduler.block_manager.allocate(running)
    running.status = SequenceStatus.RUNNING
    scheduler.running.append(running)
    waiting = Sequence([3, 4], block_size=2, sampling_params=params)
    scheduler.add_sequence(waiting)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is False
    assert scheduled == [running]
    assert waiting in scheduler.waiting
    assert scheduler.preemption_count == 0


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

    assert scheduled == [running_seq]
    assert is_prefill is False
    assert dummy.rebalance_calls == []


def test_prefill_can_transfer_cache_before_local_reclaim(monkeypatch):
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
    scheduler.preserve_cache_via_transfer = True
    no_decode = SimpleNamespace(
        temperature=1.0, max_tokens=0, ignore_eos=True, max_model_length=None
    )
    cached = Sequence([1, 2], block_size=2, sampling_params=no_decode)
    scheduler.block_manager.allocate(cached)
    scheduler.block_manager.mark_kv_ready([cached])
    scheduler.block_manager.deallocate(cached)
    waiting = Sequence([3, 4, 5, 6], block_size=2, sampling_params=no_decode)
    scheduler.add_sequence(waiting)

    def transfer_one_block(gpu_id, needed_blocks):
        dummy.rebalance_calls.append((gpu_id, needed_blocks))
        return scheduler.block_manager.reclaim_cached_blocks(needed_blocks) == needed_blocks

    dummy.rebalance = transfer_one_block
    monkeypatch.setattr(
        scheduler.block_manager,
        "reclaim_for_sequence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local reclaim ran before successful transfer")
        ),
    )

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert scheduled == [waiting]
    assert dummy.rebalance_calls == [(0, 1)]


def test_prefill_transfer_shortage_includes_decode_headroom(monkeypatch):
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
    scheduler.preserve_cache_via_transfer = True
    no_decode = SimpleNamespace(
        temperature=1.0, max_tokens=0, ignore_eos=True, max_model_length=None
    )
    one_decode = SimpleNamespace(
        temperature=1.0, max_tokens=1, ignore_eos=True, max_model_length=None
    )
    cached = Sequence([1, 2], block_size=2, sampling_params=no_decode)
    scheduler.block_manager.allocate(cached)
    scheduler.block_manager.mark_kv_ready([cached])
    scheduler.block_manager.deallocate(cached)
    waiting = Sequence([3, 4], block_size=2, sampling_params=one_decode)
    scheduler.add_sequence(waiting)

    def transfer_one_block(gpu_id, needed_blocks):
        dummy.rebalance_calls.append((gpu_id, needed_blocks))
        return scheduler.block_manager.reclaim_cached_blocks(needed_blocks) == needed_blocks

    dummy.rebalance = transfer_one_block
    monkeypatch.setattr(
        scheduler.block_manager,
        "reclaim_for_sequence",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local reclaim ran before successful headroom transfer")
        ),
    )

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert scheduled == [waiting]
    assert dummy.rebalance_calls == [(0, 1)]


def test_prefill_reclaims_locally_only_after_headroom_transfer_fails(monkeypatch):
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
    scheduler.preserve_cache_via_transfer = True
    no_decode = SimpleNamespace(
        temperature=1.0, max_tokens=0, ignore_eos=True, max_model_length=None
    )
    one_decode = SimpleNamespace(
        temperature=1.0, max_tokens=1, ignore_eos=True, max_model_length=None
    )
    cached = Sequence([1, 2], block_size=2, sampling_params=no_decode)
    scheduler.block_manager.allocate(cached)
    scheduler.block_manager.mark_kv_ready([cached])
    scheduler.block_manager.deallocate(cached)
    waiting = Sequence([3, 4], block_size=2, sampling_params=one_decode)
    scheduler.add_sequence(waiting)

    events = []

    def failed_transfer(gpu_id, needed_blocks):
        events.append(("transfer", gpu_id, needed_blocks))
        return False

    original_reclaim = scheduler.block_manager.reclaim_for_sequence

    def reclaim_after_transfer(seq, reserve_blocks=0, protected_block_ids=None):
        events.append(("reclaim", reserve_blocks))
        return original_reclaim(
            seq,
            reserve_blocks=reserve_blocks,
            protected_block_ids=protected_block_ids,
        )

    dummy.rebalance = failed_transfer
    monkeypatch.setattr(scheduler.block_manager, "reclaim_for_sequence", reclaim_after_transfer)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert scheduled == [waiting]
    assert events == [("transfer", 0, 1), ("reclaim", 1)]


def test_prefill_reclaim_preserves_prefix_promised_to_next_waiting_request():
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=16,
        max_cached_blocks=3,
        block_size=2,
        eos=999,
        global_scheduler=None,
    )
    no_decode = SimpleNamespace(
        temperature=1.0, max_tokens=0, ignore_eos=True, max_model_length=None
    )
    cold = Sequence([1, 2], block_size=2, sampling_params=no_decode)
    promised = Sequence([3, 4], block_size=2, sampling_params=no_decode)
    for seq in (cold, promised):
        scheduler.block_manager.allocate(seq)
        scheduler.block_manager.mark_kv_ready([seq])
        scheduler.block_manager.deallocate(seq)

    promised_hash = scheduler.block_manager.compute_hash([3, 4], -1)
    promised.routed_prefix_hashes = [promised_hash]
    incoming = Sequence([5, 6, 7, 8], block_size=2, sampling_params=no_decode)
    scheduler.add_sequence(incoming)
    scheduler.add_sequence(promised)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert scheduled == [incoming, promised]
    assert promised.num_cached_tokens == 2


def test_prefill_reclaim_drops_tail_route_promise_when_all_cache_is_protected():
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=16,
        max_cached_blocks=3,
        block_size=2,
        eos=999,
        global_scheduler=None,
    )
    no_decode = SimpleNamespace(
        temperature=1.0, max_tokens=0, ignore_eos=True, max_model_length=None
    )
    promised = []
    for tokens in ([1, 2], [3, 4]):
        seq = Sequence(tokens, block_size=2, sampling_params=no_decode)
        scheduler.block_manager.allocate(seq)
        scheduler.block_manager.mark_kv_ready([seq])
        scheduler.block_manager.deallocate(seq)
        seq.routed_prefix_hashes = [scheduler.block_manager.compute_hash(tokens, -1)]
        promised.append(seq)

    incoming = Sequence([5, 6, 7, 8], block_size=2, sampling_params=no_decode)
    scheduler.add_sequence(incoming)
    scheduler.add_sequence(promised[0])
    scheduler.add_sequence(promised[1])

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is True
    assert scheduled[0] is incoming
    assert incoming.status == SequenceStatus.RUNNING


def test_decode_growth_preserves_waiting_routed_prefix_when_another_victim_exists():
    scheduler = Scheduler(
        max_num_sequences=4,
        max_num_batched_tokens=1,
        max_cached_blocks=3,
        block_size=2,
        eos=999,
        global_scheduler=None,
    )
    no_decode = SimpleNamespace(
        temperature=1.0, max_tokens=0, ignore_eos=True, max_model_length=None
    )
    cold = Sequence([1, 2], block_size=2, sampling_params=no_decode)
    promised = Sequence([3, 4], block_size=2, sampling_params=no_decode)
    for cached in (cold, promised):
        scheduler.block_manager.allocate(cached)
        scheduler.block_manager.mark_kv_ready([cached])
        scheduler.block_manager.deallocate(cached)

    promised_hash = scheduler.block_manager.compute_hash([3, 4], -1)
    promised.routed_prefix_hashes = [promised_hash]
    scheduler.add_sequence(promised)

    running = Sequence([5, 6], block_size=2, sampling_params=no_decode)
    scheduler.block_manager.allocate(running)
    running.append_token(7)
    running.status = SequenceStatus.RUNNING
    scheduler.running.append(running)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill is False
    assert scheduled == [running]
    assert promised_hash in scheduler.block_manager.hash_to_block_id


def test_structural_rebalance_failures_use_exponential_cooldown(monkeypatch):
    gbm = GlobalBlockManager(rank=0, world_size=2, num_blocks_per_gpu=1, nvlink_pairs=[(0, 1)])
    dummy = DummyGlobalScheduler(gbm, rebalance_result=False, fail_reason="no_target_space")
    scheduler = Scheduler(
        max_num_sequences=1,
        max_num_batched_tokens=4,
        max_cached_blocks=1,
        block_size=2,
        eos=999,
        global_scheduler=dummy,
    )
    scheduler._rebalance_cooldown_s = 2.0
    scheduler._rebalance_cooldown_max_s = 30.0
    now = [100.0]
    monkeypatch.setattr("lmpool.engine.scheduler.time.monotonic", lambda: now[0])

    key = ("capacity", 0)
    scheduler._mark_rebalance_failed(key)
    assert scheduler._rebalance_cooldown_until[key] == 102.0
    now[0] = 102.0
    scheduler._mark_rebalance_failed(key)
    assert scheduler._rebalance_cooldown_until[key] == 106.0
    now[0] = 106.0
    scheduler._mark_rebalance_failed(key)
    assert scheduler._rebalance_cooldown_until[key] == 114.0

    scheduler._mark_rebalance_succeeded(key)
    assert key not in scheduler._rebalance_failure_streak
    assert key not in scheduler._rebalance_cooldown_until


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
