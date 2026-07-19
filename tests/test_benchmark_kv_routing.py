import sys
from types import SimpleNamespace

from benchmarks.benchmark_kv_routing import (
    _configure,
    _configure_routing,
    parse_args,
)


def _args(**overrides):
    values = {
        "model_name_or_path": "/tmp/model",
        "kv_block_budget": 64,
        "gpu_memory_utilization": 0.5,
        "seed": 7,
        "route_load_weight": 0.03,
        "route_decode_token_weight": 8.0,
        "route_owner_spill_sequence_skew": 2.0,
        "route_owner_spill_max_extra_cost": 2048.0,
        "route_load_bypass_threshold": 256.0,
        "route_prefill_cost_weight": 1.0,
        "route_reclaim_cost_weight": 0.5,
        "route_cache_queue_slack": 256.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_common_routing_benchmark_config_uses_exact_shared_budget():
    config = _configure({}, _args())

    assert config["model_name_or_path"] == "/tmp/model"
    assert config["max_cached_blocks"] == 64
    assert config["gpu_memory_utilization"] == 0.5
    assert config["require_exact_kv_block_budget"] is True
    assert config["random_seed"] == 7


def test_routing_only_config_disables_all_transfer_paths():
    config = _configure_routing({}, _args())

    assert config["enable_foreground_rebalance"] is False
    assert config["enable_background_copy"] is False
    assert config["preserve_cache_via_transfer"] is False
    assert config["route_decode_token_weight"] == 8.0
    assert config["route_prefill_cost_weight"] == 1.0


def test_routing_entry_has_independent_argument_parser(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_kv_routing.py",
            "--world-size",
            "6",
            "--kv-block-budget",
            "128",
            "--repetitions",
            "5",
        ],
    )

    args = parse_args()

    assert args.world_size == 6
    assert args.kv_block_budget == 128
    assert args.repetitions == 5
    assert not hasattr(args, "workload")
    assert not hasattr(args, "disable_background_copy")
