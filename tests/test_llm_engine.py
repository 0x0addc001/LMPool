import queue
from types import SimpleNamespace

import pytest

from lmpool.engine import llm_engine as llm_engine_module
from lmpool.engine.sequence import Sequence


class DummyTokenizer:
    def encode(self, prompt):
        return [ord(ch) for ch in prompt]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class DummyQueue(queue.Queue):
    pass


class DummyProcess:
    def __init__(self, target, args):
        self.target = target
        self.args = args
        self.started = False
        self.exitcode = 0

    def start(self):
        self.started = True

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False


class DummyContext:
    def Queue(self):
        return DummyQueue()

    def Process(self, target, args):
        return DummyProcess(target, args)


def test_llm_engine_add_prompt_and_drain_messages(monkeypatch):
    monkeypatch.setattr(llm_engine_module.mp, "get_context", lambda _: DummyContext())
    monkeypatch.setattr(llm_engine_module.AutoTokenizer, "from_pretrained", lambda _: DummyTokenizer())
    monkeypatch.setattr(llm_engine_module, "_find_free_port", lambda: 23456)

    engine = llm_engine_module.LLMEngine(
        {
            "world_size": 1,
            "block_size": 2,
            "model_name_or_path": "dummy",
            "enable_global_pool": False,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 16,
            "max_cached_blocks": 4,
            "eos": 999,
            "log_level": "ERROR",
        }
    )
    assert engine.config["distributed_init_method"].startswith("tcp://127.0.0.1:")

    engine.add_prompt("ab", SimpleNamespace(temperature=1.0, max_tokens=4, ignore_eos=True, max_model_length=None))
    queued = engine.send_queues[0].get(timeout=1)
    assert queued["type"] == "sequence"
    assert queued["seq"].token_ids == [97, 98]

    engine.recv_queues[0].put({"type": "sequence", "target": 0, "seq": Sequence([1, 2], block_size=2)})
    engine.recv_queues[0].put({"type": "first_token", "data": [(123, 9)]})
    engine.recv_queues[0].put({"type": "prefill_stats", "data": [{"seq_id": 123, "prefix_hit": True, "num_cached_tokens": 2}]})
    engine.recv_queues[0].put({"type": "runtime_stats", "data": {"swap_count": 2}})
    engine.recv_queues[0].put({"type": "finished", "data": [(123, [9, 8])]})
    finished, first_tokens, prefill_stats, runtime_stats = engine.step()
    assert finished == [(123, [9, 8])]
    assert first_tokens == [(123, 9)]
    assert prefill_stats == [{"seq_id": 123, "prefix_hit": True, "num_cached_tokens": 2}]
    assert runtime_stats == [{"swap_count": 2}]
    forwarded = engine.send_queues[0].get(timeout=1)
    assert forwarded["type"] == "sequence"

    engine.exit()
