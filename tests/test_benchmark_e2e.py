import re
from types import SimpleNamespace

import pytest

from benchmarks.shared_prefix_benchmark import (
    build_prompts,
    measure_single_gpu_prefix_hit_rate,
    resolve_memory_skew_source_ranks,
    resolve_kv_block_budget,
)


class IdentityChatTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"]

    def encode(self, prompt):
        return [ord(char) for char in prompt]


def _locality_groups(prompts):
    return [
        re.search(r"prefix group (locality-\d{4})", prompt).group(1)
        for prompt in prompts
    ]


def _prefix_group(prompt):
    return re.search(r"prefix group ([^.]*)", prompt).group(1)


def test_locality_workload_builds_balanced_distinct_prefix_groups():
    prompts = build_prompts(
        IdentityChatTokenizer(),
        num_prompts=32,
        prompt_repeat=2,
        workload="locality",
        locality_prefix_groups=8,
        seed=7,
    )

    groups = _locality_groups(prompts)
    assert set(groups) == {f"locality-{group:04d}" for group in range(8)}
    assert all(groups.count(group) == 4 for group in set(groups))
    assert groups != [f"locality-{index % 8:04d}" for index in range(32)]


def test_locality_workload_order_is_seeded():
    kwargs = {
        "num_prompts": 32,
        "prompt_repeat": 1,
        "workload": "locality",
        "locality_prefix_groups": 8,
    }

    first = _locality_groups(build_prompts(IdentityChatTokenizer(), seed=3, **kwargs))
    repeated = _locality_groups(build_prompts(IdentityChatTokenizer(), seed=3, **kwargs))
    different = _locality_groups(build_prompts(IdentityChatTokenizer(), seed=4, **kwargs))

    assert first == repeated
    assert first != different


def test_memory_skew_workload_has_warmup_pressure_and_reuse_phases():
    prompts = build_prompts(
        IdentityChatTokenizer(),
        num_prompts=16,
        prompt_repeat=4,
        workload="memory-skew",
        seed=0,
    )

    groups = [_prefix_group(prompt) for prompt in prompts]
    assert groups[:4] == ["transfer-hot"] * 4
    assert groups[4:8] == [f"pressure-{index:04d}" for index in range(4)]
    assert groups[8:] == ["transfer-hot"] * 8


def test_memory_skew_placement_is_explicit_for_topology_blind_baseline():
    config = {
        "world_size": 6,
        "benchmark_memory_skew_source_ranks": [0, 2, 4],
    }

    assert resolve_memory_skew_source_ranks(config) == [0, 2, 4]


def test_single_gpu_prefix_measurement_publishes_ready_kv_blocks():
    tokenizer = IdentityChatTokenizer()

    hit_rate = measure_single_gpu_prefix_hit_rate(
        tokenizer,
        prompts=["abcd", "wxyz", "abcd", "wxyz"],
        block_size=2,
        max_cached_blocks=16,
    )

    assert hit_rate == 0.5


def test_theoretical_prefix_measurement_is_not_limited_by_runtime_budget():
    tokenizer = IdentityChatTokenizer()

    hit_rate = measure_single_gpu_prefix_hit_rate(
        tokenizer,
        prompts=["abcd", "wxyz", "abcd", "wxyz"],
        block_size=2,
        max_cached_blocks=1,
    )

    assert hit_rate == 0.5


def test_kv_block_budget_accepts_equal_legacy_values():
    args = SimpleNamespace(
        kv_block_budget=64,
        routing_max_cached_blocks=64,
        eviction_max_cached_blocks=64,
    )

    assert resolve_kv_block_budget(args) == 64


def test_kv_block_budget_rejects_unfair_scenario_limits():
    args = SimpleNamespace(
        kv_block_budget=None,
        routing_max_cached_blocks=1024,
        eviction_max_cached_blocks=64,
    )

    with pytest.raises(ValueError, match="must be equal"):
        resolve_kv_block_budget(args)
