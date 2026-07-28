import json
import re
from types import SimpleNamespace

import pytest

from benchmarks.benchmark_e2e import (
    MODEL_CONFIG,
    ScenarioResult,
    aggregate_scenario_trials,
    build_prompts,
    confidence_interval_95,
    compute_goodput_sla_sweep,
    compute_sequence_prefix_hashes,
    decode_tpot_s,
    measure_single_gpu_prefix_hit_rate,
    profile_trace_prefix_sharing,
    parse_goodput_sla_sweep_ms,
    prepare_benchmark_rendezvous,
    resolve_load_skew_phases,
    resolve_memory_skew_phases,
    resolve_memory_skew_prefix_groups,
    resolve_transfer_calibration_prefix_groups,
    resolve_transfer_calibration_warmup_prompts,
    resolve_kv_block_budget,
    save_reuse_phase_figure,
    save_summary_figure,
    save_summary_json,
    workload_summary_title,
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


def test_load_skew_workload_has_source_warmup_and_hot_reuse_phases():
    prompts = build_prompts(
        IdentityChatTokenizer(),
        num_prompts=18,
        prompt_repeat=4,
        workload="load-skew",
        load_skew_prefix_groups=3,
        load_skew_warmup_prompts=6,
        load_skew_hot_groups=3,
        load_skew_hot_share=0.8,
        seed=0,
    )

    groups = [_prefix_group(prompt) for prompt in prompts]
    expected = [f"load-hot-{group:04d}" for group in range(3)]
    assert groups[:6] == expected * 2
    reuse_groups = groups[6:]
    hot_reuse = [group for group in reuse_groups if group.startswith("load-hot-")]
    cold_reuse = [group for group in reuse_groups if group.startswith("load-cold-")]
    assert len(hot_reuse) == 10
    assert set(hot_reuse) == set(expected)
    assert len(cold_reuse) == len(set(cold_reuse)) == 2
    assert set(cold_reuse) == {"load-cold-0000", "load-cold-0001"}


def test_load_skew_phases_require_hot_group_coverage():
    assert resolve_load_skew_phases(192, 6, 48) == (48, 144)
    assert resolve_load_skew_phases(32, 4, 0) == (8, 24)

    with pytest.raises(ValueError, match="warm-up phase"):
        resolve_load_skew_phases(16, 5, 4)
    with pytest.raises(ValueError, match="reuse phase"):
        resolve_load_skew_phases(16, 9, 9)


def test_load_skew_reuse_order_breaks_round_robin_prefix_alignment():
    prompts = build_prompts(
        IdentityChatTokenizer(),
        num_prompts=384,
        prompt_repeat=2,
        workload="load-skew",
        load_skew_prefix_groups=24,
        load_skew_warmup_prompts=48,
        load_skew_hot_groups=24,
        load_skew_hot_share=0.8,
        seed=0,
    )

    targets_by_group = {group: set() for group in range(24)}
    for reuse_index, prompt in enumerate(prompts[48:]):
        group_name = _prefix_group(prompt)
        if not group_name.startswith("load-hot-"):
            continue
        group = int(group_name.rsplit("-", 1)[1])
        targets_by_group[group].add(reuse_index % 6)

    assert all(len(targets) >= 3 for targets in targets_by_group.values())
    assert any(len(targets) >= 4 for targets in targets_by_group.values())


def test_memory_skew_workload_separates_session_anchor_pressure_trigger_and_reuse():
    prompts = build_prompts(
        IdentityChatTokenizer(),
        num_prompts=16,
        prompt_repeat=4,
        workload="memory-skew",
        memory_skew_prefix_groups=3,
        seed=0,
    )

    groups = [_prefix_group(prompt) for prompt in prompts]
    assert all("transfer-session-" in prompt for prompt in prompts[:4])
    assert set(prompts[11:]).issubset(set(prompts[:4]))
    assert all(
        f"memory-pressure-tail-{index:04d}" in prompts[4 + index]
        for index in range(4)
    )
    assert all("transfer-anchor-" in prompts[8 + index] for index in range(3))
    assert all(
        f"memory-trigger-tail-{index:04d}" in prompts[8 + index]
        for index in range(3)
    )
    assert all(
        "memory-pressure-tail" not in prompt and "memory-trigger-tail" not in prompt
        for prompt in prompts[11:]
    )


def test_memory_skew_workload_accepts_explicit_phase_sizes():
    prompts = build_prompts(
        IdentityChatTokenizer(),
        num_prompts=14,
        prompt_repeat=4,
        workload="memory-skew",
        memory_skew_prefix_groups=3,
        memory_skew_warmup_prompts=3,
        memory_skew_pressure_prompts=2,
        memory_skew_trigger_prompts=3,
        seed=0,
    )

    groups = [_prefix_group(prompt) for prompt in prompts]
    assert all("transfer-session-" in prompt for prompt in prompts[:3])
    assert all(group.startswith("transfer-anchor-") for group in groups[3:5])
    assert "memory-pressure-tail-0000" in prompts[3]
    assert "memory-pressure-tail-0001" in prompts[4]
    assert all("memory-trigger-tail" in prompt for prompt in prompts[5:8])
    assert set(prompts[8:]).issubset(set(prompts[:3]))
    assert all(
        "memory-pressure-tail" not in prompt and "memory-trigger-tail" not in prompt
        for prompt in prompts[8:]
    )


def test_capacity_offload_uses_the_controlled_memory_pressure_trace():
    kwargs = {
        "tokenizer": IdentityChatTokenizer(),
        "num_prompts": 16,
        "prompt_repeat": 4,
        "memory_skew_prefix_groups": 3,
        "seed": 0,
    }
    memory_skew = build_prompts(workload="memory-skew", **kwargs)
    capacity_offload = build_prompts(workload="capacity-offload", **kwargs)

    assert capacity_offload == memory_skew


def test_memory_skew_concentrates_pressure_on_hot_anchors_without_rank_hints():
    prompts = build_prompts(
        IdentityChatTokenizer(),
        num_prompts=40,
        prompt_repeat=4,
        workload="memory-skew",
        memory_skew_prefix_groups=4,
        memory_skew_warmup_prompts=4,
        memory_skew_pressure_prompts=20,
        memory_skew_trigger_prompts=4,
        memory_skew_pressure_hot_groups=2,
        memory_skew_pressure_hot_share=0.8,
        seed=0,
    )

    pressure_groups = [
        int(_prefix_group(prompt).rsplit("-", 1)[1])
        for prompt in prompts[4:24]
    ]
    assert pressure_groups.count(0) == 8
    assert pressure_groups.count(1) == 8
    assert pressure_groups.count(2) == 2
    assert pressure_groups.count(3) == 2


def test_transfer_calibration_repeats_groups_in_two_equal_phases():
    prompts = build_prompts(
        IdentityChatTokenizer(),
        num_prompts=12,
        prompt_repeat=2,
        workload="transfer-calibration",
        calibration_prefix_groups=6,
    )

    groups = [_prefix_group(prompt) for prompt in prompts]
    expected = [f"calibration-{group:04d}" for group in range(6)]
    assert groups[:6] == expected
    assert groups[6:] == expected


def test_transfer_calibration_supports_short_build_and_long_reuse_phase():
    prompts = build_prompts(
        IdentityChatTokenizer(),
        num_prompts=12,
        prompt_repeat=2,
        workload="transfer-calibration",
        calibration_prefix_groups=3,
        calibration_warmup_prompts=3,
    )

    groups = [_prefix_group(prompt) for prompt in prompts]
    expected = [f"calibration-{group:04d}" for group in range(3)]
    assert groups[:3] == expected
    assert groups[3:] == expected * 3


def test_transfer_calibration_prefix_groups_fit_both_phases():
    assert resolve_transfer_calibration_prefix_groups(128, 0) == 32
    assert resolve_transfer_calibration_prefix_groups(128, 64) == 64
    assert resolve_transfer_calibration_prefix_groups(128, 32, 32) == 32
    with pytest.raises(ValueError):
        resolve_transfer_calibration_prefix_groups(128, 65)
    with pytest.raises(ValueError):
        resolve_transfer_calibration_prefix_groups(128, 33, 32)
    with pytest.raises(ValueError):
        resolve_transfer_calibration_prefix_groups(127, 32)


def test_transfer_calibration_warmup_defaults_to_half_or_accepts_explicit():
    assert resolve_transfer_calibration_warmup_prompts(128, 0) == 64
    assert resolve_transfer_calibration_warmup_prompts(128, 32) == 32
    with pytest.raises(ValueError):
        resolve_transfer_calibration_warmup_prompts(128, 128)


def test_memory_skew_prefix_groups_auto_fit_phase_and_avoid_even_period():
    assert resolve_memory_skew_prefix_groups(128, 0) == 15
    assert resolve_memory_skew_prefix_groups(32, 0) == 3
    with pytest.raises(ValueError):
        resolve_memory_skew_prefix_groups(16, 5)


def test_memory_skew_phases_accept_explicit_sizes_and_validate_group_coverage():
    assert resolve_memory_skew_phases(160, 16, 64, 32, 16) == (64, 32, 16, 48)
    assert resolve_memory_skew_phases(16, 3, 0, 0) == (4, 4, 3, 5)

    with pytest.raises(ValueError, match="warm-up phase"):
        resolve_memory_skew_phases(16, 5, 4, 4, 5)
    assert resolve_memory_skew_phases(16, 3, 3, 3, 2) == (3, 3, 2, 8)
    with pytest.raises(ValueError, match="reuse phase"):
        resolve_memory_skew_phases(16, 4, 8, 1, 4)
    with pytest.raises(ValueError, match="non-empty warm-up, pressure, trigger, and reuse"):
        resolve_memory_skew_phases(16, 3, 8, 8, 3)


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


def test_legacy_prefix_measurement_matches_trace_request_share_rate():
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


def test_trace_prefix_profile_reports_request_and_token_sharing():
    profile = profile_trace_prefix_sharing(
        IdentityChatTokenizer(),
        prompts=["abcd", "wxyz", "abcd", "wxyz"],
        block_size=2,
    )

    assert profile["total_requests"] == 4
    assert profile["total_prompt_tokens"] == 16
    assert profile["shareable_requests"] == 2
    assert profile["shareable_prefix_blocks"] == 4
    assert profile["shareable_prefix_tokens"] == 8
    assert profile["request_prefix_share_rate"] == 0.5
    assert profile["token_prefix_share_ratio"] == 0.5


def test_trace_token_sharing_excludes_partial_tail_blocks():
    profile = profile_trace_prefix_sharing(
        IdentityChatTokenizer(),
        prompts=["abcde", "abcde"],
        block_size=2,
    )

    assert profile["request_prefix_share_rate"] == 0.5
    assert profile["shareable_prefix_tokens"] == 4
    assert profile["token_prefix_share_ratio"] == 0.4


def test_kv_block_budget_accepts_explicit_value():
    args = SimpleNamespace(kv_block_budget=64)

    assert resolve_kv_block_budget(args) == 64


def test_kv_block_budget_uses_common_default():
    args = SimpleNamespace(kv_block_budget=None)

    assert resolve_kv_block_budget(args) == MODEL_CONFIG["max_cached_blocks"]


def test_kv_block_budget_rejects_non_positive_value():
    args = SimpleNamespace(kv_block_budget=0)

    with pytest.raises(ValueError, match="must be >= 1"):
        resolve_kv_block_budget(args)


@pytest.mark.parametrize(
    ("workload", "expected"),
    [
        ("locality", "KV Locality End-to-End Benchmark Summary"),
        ("load-skew", "Load-Skew Transfer-Relief Benchmark Summary"),
        ("memory-skew", "Memory-Skew Capacity-Offload Benchmark Summary"),
        ("capacity-offload", "Memory-Skew Capacity-Offload Benchmark Summary"),
        (
            "transfer-calibration",
            "KV Transfer Transaction Calibration Summary",
        ),
    ],
)
def test_workload_summary_title_is_specific(workload, expected):
    assert workload_summary_title(workload) == expected


def test_workload_summary_title_rejects_unknown_workload():
    with pytest.raises(ValueError, match="unknown workload"):
        workload_summary_title("unknown")


def test_confidence_interval_uses_repeated_samples():
    assert confidence_interval_95([1.0]) == 0.0
    assert confidence_interval_95([2.0, 2.0, 2.0]) == 0.0
    assert confidence_interval_95([1.0, 2.0, 4.0]) > 0.0


def test_decode_tpot_excludes_ttft_and_first_output_token():
    assert decode_tpot_s(10.0, 11.5, 4) == pytest.approx(0.5)
    assert decode_tpot_s(10.0, 11.5, 1) is None


def test_confidence_interval_uses_student_t_half_width():
    samples = [10.0, 12.0, 9.0, 11.0, 13.0]
    expected = 2.776 * 1.5811388300841898 / (5 ** 0.5)

    assert confidence_interval_95(samples) == pytest.approx(expected)


def test_goodput_sla_sweep_reuses_one_completion_sample_set():
    sweep = compute_goodput_sla_sweep(
        completion_times={1: 2.0, 2: 4.0},
        submit_times={1: 0.0, 2: 0.0},
        completion_token_counts={1: 10, 2: 20},
        elapsed_s=5.0,
        sla_thresholds_s=[2.0, 3.0, 5.0],
    )

    assert sweep == {"2000": 2.0, "3000": 2.0, "5000": 6.0}


def test_goodput_sla_sweep_parser_adds_primary_and_deduplicates():
    assert parse_goodput_sla_sweep_ms("2000,5000,10000", 3000) == [
        2.0,
        3.0,
        5.0,
        10.0,
    ]
    with pytest.raises(ValueError, match="must be positive"):
        parse_goodput_sla_sweep_ms("0,3000", 3000)


def test_summary_json_keeps_metadata_separate_from_results(tmp_path):
    output = tmp_path / "summary.json"

    save_summary_json(
        {"multi-gpu": {"throughput_tok_s": 12.0}},
        str(output),
        metadata={"schema_version": 2, "model": {"hidden_size": 2048}},
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["model"]["hidden_size"] == 2048
    assert payload["results"]["multi-gpu"]["throughput_tok_s"] == 12.0


def test_repeated_results_preserve_route_decision_counts():
    base = dict(
        name="multi-gpu-lmpool",
        total_requests=2,
        total_tokens=2,
        elapsed_s=1.0,
        throughput_tok_s=2.0,
        goodput_tok_s=2.0,
        mean_ttft_s=1.0,
        p50_ttft_s=1.0,
        p90_ttft_s=1.0,
        p95_ttft_s=1.0,
        mean_tpot_s=1.0,
        p50_tpot_s=1.0,
        p90_tpot_s=1.0,
        p95_tpot_s=1.0,
        mean_e2e_s=1.0,
        p50_e2e_s=1.0,
        p90_e2e_s=1.0,
        p95_e2e_s=1.0,
        route_hit_rate=0.0,
        routed_to_prefix_owner_rate=0.0,
        prefix_hit_rate=0.0,
        initial_cached_token_ratio=0.0,
        prefill_attempts=0,
        preemption_count=0,
        redundant_prefill_tokens=0,
        transfer_count=0,
        transfer_bytes=0,
        transfer_time_s=0.0,
        transfer_source_time_s=0.0,
        transfer_target_time_s=0.0,
        transfer_bandwidth_gib_s=0.0,
        estimated_transfer_cost_ms=0.0,
        estimated_saved_prefill_ms=0.0,
        transfer_copy_count=0,
        transfer_release_count=0,
        chain_transfer_count=0,
        hot_transfer_block_count=0,
        hot_transfer_block_ratio=0.0,
        rebalance_success=0,
        rebalance_fail=0,
        rebalance_fail_reasons={},
        background_copy_success=0,
        background_copy_fail=0,
        background_copy_fail_reasons={},
        gpu_util_mean=0.0,
        gpu_util_p95=0.0,
        gpu_mem_util_mean=0.0,
        gpu_mem_util_p95=0.0,
        route_decision_counts={"prefix_hit": 2},
        rank_stats={0: {"route_decision_counts": {"prefix_hit": 2}}},
    )
    first = ScenarioResult(**base)
    second = ScenarioResult(
        **{**base, "route_decision_counts": {
            "prefix_owner_transfer_admission": 1,
            "prefix_hit": 1,
        }, "rank_stats": {0: {"route_decision_counts": {
            "prefix_owner_transfer_admission": 1,
            "prefix_hit": 1,
        }}}}
    )

    result = aggregate_scenario_trials([first, second])

    assert result.route_decision_counts == {
        "prefix_hit": 2,
        "prefix_owner_transfer_admission": 1,
    }
    assert result.rank_stats[0]["route_decision_counts"] == {
        "prefix_hit": 2,
        "prefix_owner_transfer_admission": 1,
    }


def test_summary_figures_accept_confidence_intervals(tmp_path):
    result = SimpleNamespace(
        name="multi-gpu-lmpool",
        throughput_tok_s=100.0,
        goodput_tok_s=90.0,
        throughput_tok_s_ci95=4.0,
        goodput_tok_s_ci95=3.0,
        mean_ttft_s=0.2,
        mean_tpot_s=0.03,
        mean_e2e_s=1.0,
        p90_e2e_s=1.4,
        mean_ttft_s_ci95=0.01,
        mean_tpot_s_ci95=0.002,
        mean_e2e_s_ci95=0.05,
        p90_e2e_s_ci95=0.08,
        route_hit_rate=0.8,
        routed_to_prefix_owner_rate=0.75,
        prefix_hit_rate=0.7,
        initial_cached_token_ratio=0.65,
        gpu_util_mean=60.0,
        gpu_mem_util_mean=30.0,
        phase_latency_stats={
            "reuse": {
                "throughput_tok_s": 120.0,
                "throughput_tok_s_ci95": 5.0,
                "mean_ttft_s": 0.1,
                "mean_ttft_s_ci95": 0.01,
                "mean_e2e_s": 0.8,
                "mean_e2e_s_ci95": 0.04,
                "p90_e2e_s": 1.1,
                "p90_e2e_s_ci95": 0.07,
            }
        },
    )
    output = tmp_path / "summary.png"

    save_summary_figure([result], str(output), title="Test Model")
    save_reuse_phase_figure([result], str(output), title="Test Model")

    assert output.stat().st_size > 0
    assert (tmp_path / "summary_reuse_phase.png").stat().st_size > 0
