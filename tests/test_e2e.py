import queue
from types import SimpleNamespace

from lmpool.engine import llm_engine as llm_engine_module


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
        self.pid = 12345

    def start(self):
        self.started = True

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False

    def terminate(self):
        self.exitcode = -15


class DummyContext:
    def Queue(self):
        return DummyQueue()

    def Process(self, target, args):
        return DummyProcess(target, args)


class FakeControlPlaneClient:
    def reset_control_epoch(self, control_epoch):
        self.control_epoch = control_epoch

    def route_sequence(self, seq, return_meta=False):
        if return_meta:
            return {
                "target_rank": 1,
                "route_info": {
                    "prefix_hit": True,
                    "hit_summary": {1: [0]},
                    "reason": "test",
                },
            }
        return 1


def test_e2e_ingress_route_worker_result_flow(monkeypatch):
    monkeypatch.setattr(llm_engine_module.mp, "get_context", lambda _: DummyContext())
    monkeypatch.setattr(llm_engine_module.AutoTokenizer, "from_pretrained", lambda _: DummyTokenizer())
    monkeypatch.setattr(llm_engine_module, "_find_free_port", lambda: 23456)

    engine = llm_engine_module.LLMEngine(
        {
            "world_size": 2,
            "block_size": 2,
            "model_name_or_path": "dummy",
            "enable_global_pool": True,
            "use_control_plane_process": True,
            "max_num_sequences": 4,
            "max_num_batched_tokens": 16,
            "max_cached_blocks": 4,
            "eos": 999,
            "log_level": "ERROR",
        }
    )
    engine.control_plane_client = FakeControlPlaneClient()

    sampling = SimpleNamespace(
        temperature=1.0,
        max_tokens=4,
        ignore_eos=True,
        max_model_length=None,
    )
    engine.add_prompt("ab", sampling)
    queued = engine.send_queues[1].get(timeout=1)
    seq = queued["seq"]
    assert queued["type"] == "sequence"
    assert seq.token_ids == [97, 98]

    engine.recv_queues[1].put({"type": "first_token", "data": [(seq.seq_id, 7)]})
    engine.recv_queues[1].put({
        "type": "prefill_stats",
        "data": [{"seq_id": seq.seq_id, "prefix_hit": True, "num_cached_tokens": 2}],
    })
    engine.recv_queues[1].put({
        "type": "runtime_stats",
        "data": {"transfer_count": 1, "transfer_copy_count": 1},
    })
    engine.recv_queues[1].put({"type": "finished", "data": [(seq.seq_id, [7, 8])]})

    finished, first_tokens, prefill_stats, runtime_stats = engine.step()

    assert finished == [(seq.seq_id, [7, 8])]
    assert first_tokens == [(seq.seq_id, 7)]
    assert prefill_stats == [{"seq_id": seq.seq_id, "prefix_hit": True, "num_cached_tokens": 2}]
    assert runtime_stats == [{"transfer_count": 1, "transfer_copy_count": 1}]
    engine.exit()
