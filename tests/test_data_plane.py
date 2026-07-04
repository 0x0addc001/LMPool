import queue
import threading
from types import SimpleNamespace
import sys

from lmpool.engine import data_plane as data_plane_module
from lmpool.engine.block_manager import BlockManager
from lmpool.engine.sequence import Sequence, SequenceStatus


class FakeModelRunner:
    def __init__(self, config, rank, gbm=None):
        self.config = config
        self.rank = rank
        self.gbm = gbm
        self.calls = []

    def run(self, seqs, is_prefill):
        self.calls.append((tuple(s.seq_id for s in seqs), is_prefill))
        return [7 for _ in seqs]

    def exit(self):
        self.calls.append(("exit",))

    def execute_swap_out(self, blocks, target_gpu):
        self.calls.append(("swap_out", tuple(blocks), target_gpu))

    def execute_swap_in(self, src_gpu, blocks, local_target_blocks=None):
        self.calls.append(("swap_in", src_gpu, tuple(blocks), tuple(local_target_blocks or [])))


class FakeScheduler:
    def __init__(self, **kwargs):
        self.block_manager = BlockManager(num_blocks=4, block_size=2)
        self._queue = []

    def add_sequence(self, seq):
        self._queue.append(seq)

    def is_finished(self):
        return not self._queue

    def schedule(self):
        if not self._queue:
            return [], False
        seq = self._queue.pop(0)
        seq.status = SequenceStatus.RUNNING
        self.block_manager.allocate(seq)
        return [seq], True

    def postprocess(self, seqs, token_ids):
        for seq, token in zip(seqs, token_ids):
            seq.append_token(token)
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)


def _run_data_plane(result_queue, config, rank, recv_queue, send_queue):
    data_plane_module.ModelRunner = FakeModelRunner
    data_plane_module.Scheduler = FakeScheduler
    data_plane_module.os.fdopen = lambda *args, **kwargs: data_plane_module.sys.stdout
    data_plane_module.data_plane_process(
        config,
        rank,
        recv_queue,
        send_queue,
        None,
        None,
    )
    result_queue.put("done")


def test_data_plane_process_handles_sequence_and_exit():
    recv_queue = queue.Queue()
    send_queue = queue.Queue()
    result_queue = queue.Queue()
    config = {
        "block_size": 2,
        "world_size": 1,
        "max_num_sequences": 4,
        "max_num_batched_tokens": 16,
        "max_cached_blocks": 4,
        "eos": 999,
        "enable_global_pool": False,
        "use_control_plane_process": False,
        "worker_queue_poll_timeout": 0.01,
        "log_level": "ERROR",
    }
    thread = threading.Thread(
        target=_run_data_plane,
        args=(result_queue, config, 0, recv_queue, send_queue),
        daemon=True,
    )
    thread.start()
    try:
        seq = Sequence([1, 2], block_size=2)
        seq.remote_gpu_id = 0
        recv_queue.put({"type": "sequence", "seq": seq})
        assert send_queue.get(timeout=10)["type"] == "finished"
        recv_queue.put({"type": "exit"})
        thread.join(timeout=10)
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            recv_queue.put({"type": "exit"})
            thread.join(timeout=10)
