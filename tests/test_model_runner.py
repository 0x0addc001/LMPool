from types import SimpleNamespace

import torch

from lmpool.engine import model_runner as model_runner_module
from lmpool.engine.sequence import Sequence


class FakeModule:
    def __init__(self):
        self.k_cache = None
        self.v_cache = None


class FakeModel:
    def __init__(self):
        self._modules = [FakeModule(), FakeModule()]
        self.forward_calls = []
        self.logit_calls = []

    def modules(self):
        return iter(self._modules)

    def __call__(self, input_ids):
        self.forward_calls.append(tuple(input_ids.shape))
        return torch.ones(input_ids.shape[0], 4)

    def compute_logits(self, hidden_states):
        self.logit_calls.append(tuple(hidden_states.shape))
        return torch.arange(hidden_states.shape[0] * 5, dtype=torch.float32).view(hidden_states.shape[0], 5)


def _make_runner():
    runner = model_runner_module.ModelRunner.__new__(model_runner_module.ModelRunner)
    runner.config = {
        "block_size": 2,
        "world_size": 2,
        "enforce_eager": True,
        "enable_global_pool": True,
        "log_timing": False,
        "log_decode_every_n": 2,
        "max_num_batch_tokens": 4,
        "max_model_length": 2,
        "num_layers": 2,
        "num_kv_heads": 1,
        "head_dim": 1,
        "hidden_size": 2,
        "num_heads": 2,
        "scale": 1,
        "rms_norm_epsilon": 1e-6,
        "qkv_bias": False,
        "base": 1000000,
        "max_position": 16,
        "intermediate_size": 8,
        "ffn_bias": False,
        "tie_word_embeddings": True,
        "vocab_size": 16,
        "model_name_or_path": "Qwen/Qwen3-0.6B",
        "gpu_memory_utilization": 0.5,
    }
    runner.block_size = 2
    runner.world_size = 2
    runner.enforce_eager = True
    runner.enable_global_pool = True
    runner.log_timing = False
    runner.log_decode_every_n = 2
    runner._decode_log_counter = 0
    runner._decode_log_tokens = 0
    runner._decode_log_seconds = 0.0
    runner.rank = 0
    runner.default_dtype = torch.float32
    runner.gbm = None
    runner.model = FakeModel()
    runner.sampler = lambda logits, temp: [int(logits.shape[0])] * logits.shape[0]
    return runner


def test_allocate_kv_cache_binds_all_layers(monkeypatch):
    runner = _make_runner()
    runner.config.update({"max_cached_blocks": 4})

    real_tensor = torch.tensor
    real_zeros = torch.zeros

    monkeypatch.setattr(model_runner_module.torch.cuda, "mem_get_info", lambda: (8 * 1024 * 1024, 16 * 1024 * 1024))
    monkeypatch.setattr(model_runner_module.torch.cuda, "memory_stats", lambda: {"allocated_bytes.all.peak": 0, "allocated_bytes.all.current": 0})
    monkeypatch.setattr(model_runner_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(model_runner_module.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(model_runner_module.torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_runner_module.torch, "tensor", lambda data, *args, **kwargs: real_tensor(data, *args, **{k: v for k, v in kwargs.items() if k != "device"}))
    monkeypatch.setattr(model_runner_module.torch, "zeros", lambda *args, **kwargs: real_zeros(*args))
    monkeypatch.setattr(model_runner_module.dist, "all_reduce", lambda *args, **kwargs: None)

    runner.allocate_kv_cache()

    caches = [(m.k_cache, m.v_cache) for m in runner.model._modules]
    assert all(k is not None and v is not None for k, v in caches)
    assert runner.config["max_cached_blocks"] >= 1


def test_prepare_prefill_and_decode_set_context(monkeypatch):
    runner = _make_runner()
    seq = Sequence([1, 2, 3], block_size=2)
    seq.block_table = [10, 11]
    seq.num_cached_tokens = 0

    contexts = {}

    def fake_set_context(**kwargs):
        contexts.update(kwargs)

    monkeypatch.setattr(model_runner_module, "set_context", fake_set_context)
    real_tensor = torch.tensor
    monkeypatch.setattr(model_runner_module.torch, "tensor", lambda data, *args, **kwargs: real_tensor(data, *args, **{k: v for k, v in kwargs.items() if k not in {"pin_memory", "device"}}))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self, raising=False)

    prefill_ids = runner.prepare_prefill([seq])
    assert prefill_ids.tolist() == [1, 2, 3]
    assert contexts["is_prefill"] is True
    assert contexts["slot_mapping"].tolist() == [20, 21, 22]
    assert contexts["block_tables"] is None

    seq.last_token = 4
    seq.block_table = [10, 11]
    seq.num_tokens = 4
    decode_ids = runner.prepare_decode([seq])
    assert decode_ids.tolist() == [4]
    assert contexts["is_prefill"] is False
    assert contexts["slot_mapping"].tolist() == [23]


def test_run_model_and_kv_helpers(monkeypatch):
    runner = _make_runner()
    runner.model = FakeModel()
    runner.graphs = {1: SimpleNamespace(replay=lambda: None)}
    runner.graph_vars = {
        "input_ids": torch.zeros(1, dtype=torch.long),
        "slot_mapping": torch.zeros(1, dtype=torch.long),
        "context_lens": torch.zeros(1, dtype=torch.long),
        "block_tables": torch.zeros(1, 1, dtype=torch.int32),
        "outputs": torch.zeros(1, 5),
    }

    monkeypatch.setattr(model_runner_module, "get_context", lambda: SimpleNamespace(slot_mapping=torch.tensor([0]), context_lens=torch.tensor([1]), block_tables=torch.tensor([[1]], dtype=torch.int32)))

    logits = runner.run_model(torch.tensor([1]), is_prefill=True)
    assert logits.shape == (1, 5)

    runner.model._modules[0].k_cache = "k0"
    runner.model._modules[0].v_cache = "v0"
    assert runner._get_kv_cache() == "k0"


def test_execute_swap_methods_forward_to_kv_transfer(monkeypatch):
    runner = _make_runner()
    runner.model = FakeModel()
    runner.model._modules[0].k_cache = torch.zeros(2, 2)
    runner.model._modules[0].v_cache = torch.zeros(2, 2)
    runner.config["num_layers"] = 2

    calls = {}

    def fake_swap_out(**kwargs):
        calls["out"] = kwargs
        return [7, 8]

    def fake_swap_in(**kwargs):
        calls["in"] = kwargs
        return [9, 10]

    monkeypatch.setattr(model_runner_module, "swap_out", fake_swap_out, raising=False)
    monkeypatch.setattr(model_runner_module, "swap_in", fake_swap_in, raising=False)
    monkeypatch.setattr("lmpool.engine.kv_transfer.swap_out", fake_swap_out)
    monkeypatch.setattr("lmpool.engine.kv_transfer.swap_in", fake_swap_in)

    assert runner.execute_swap_out([1, 2], target_gpu=1) == [7, 8]
    assert runner.execute_swap_in(remote_gpu=1, remote_blocks=[3, 4], local_target_blocks=[5, 6]) == [9, 10]
    assert calls["out"]["blocks_to_evict"] == [1, 2]
    assert calls["in"]["remote_blocks"] == [3, 4]
