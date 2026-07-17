import re
from types import SimpleNamespace

import pytest

from benchmarks.shared_prefix_benchmark import (
    build_prompts,
    compute_sequence_prefix_hashes,
    measure_single_gpu_prefix_hit_rate,
    prepare_benchmark_rendezvous,
    resolve_memory_skew_prefix_groups,
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


def test_benchmark_trials_use_unique_file_rendezvous():
    first_config, first_path = prepare_benchmark_rendezvous({"world_size": 6})
    second_config, second_path = prepare_benchmark_rendezvous({"world_size": 6})

    assert first_path is not None
    assert second_path is not None
    assert first_path != second_path
    assert first_config["distributed_init_method"] == first_path.resolve().as_uri()
    assert second_config["distributed_init_method"] == second_path.resolve().as_uri()


def test_benchmark_preserves_explicit_rendezvous_method():
    config, path = prepare_benchmark_rendezvous(
        {"distributed_init_method": "tcp://127.0.0.1:23456"}
    )

    assert config["distributed_init_method"] == "tcp://127.0.0.1:23456"
    assert path is None


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
        memory_skew_prefix_groups=3,
        seed=0,
    )

    groups = [_prefix_group(prompt) for prompt in prompts]
    assert groups[:4] == [
        "transfer-hot-0000",
        "transfer-hot-0001",
        "transfer-hot-0002",
        "transfer-hot-0000",
    ]
    assert groups[4:8] == [f"pressure-{index:04d}" for index in range(4)]
    assert groups[8:] == [
        "transfer-hot-0000",
        "transfer-hot-0001",
        "transfer-hot-0002",
        "transfer-hot-0000",
        "transfer-hot-0001",
        "transfer-hot-0002",
        "transfer-hot-0000",
        "transfer-hot-0001",
    ]


def test_memory_skew_placement_is_explicit_for_topology_blind_baseline():
    config = {
        "world_size": 6,
        "benchmark_memory_skew_source_ranks": [0, 2, 4],
    }

    assert resolve_memory_skew_source_ranks(config) == [0, 2, 4]


def test_memory_skew_prefix_groups_auto_fit_phase_and_avoid_even_period():
    assert resolve_memory_skew_prefix_groups(128, 0) == 15
    assert resolve_memory_skew_prefix_groups(32, 0) == 7
    with pytest.raises(ValueError):
        resolve_memory_skew_prefix_groups(16, 5)


def test_sequence_prefix_hashes_are_cumulative_and_ignore_partial_block():
    seq = SimpleNamespace(
        token_ids=[1, 2, 3, 4, 5],
        block_size=2,
        num_tokens=5,
        block=lambda index: [[1, 2], [3, 4], [5]][index],
    )

    hashes = compute_sequence_prefix_hashes(seq)

    assert len(hashes) == 2
    assert hashes[0] != hashes[1]


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
