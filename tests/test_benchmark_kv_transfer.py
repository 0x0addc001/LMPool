import json

import pytest

from benchmarks import benchmark_kv_transfer as benchmark


def test_parse_block_counts_supports_a_deduplicated_sweep():
    assert benchmark.parse_block_counts("1,2,4,2,8", 16) == [1, 2, 4, 8]
    assert benchmark.parse_block_counts("", 4) == [4]


def test_parse_block_counts_rejects_non_positive_values():
    with pytest.raises(ValueError, match=">= 1"):
        benchmark.parse_block_counts("1,0,4", 2)


def test_transfer_benchmark_json_contains_config_and_results(tmp_path):
    args = benchmark.parse_args([
        "--block-counts", "1,2",
        "--iterations", "20",
        "--warmup", "5",
    ])
    results = [{
        "num_transfer_blocks": 1,
        "bytes_per_iteration": 1024,
        "gib_per_iteration": 1024 / (1024 ** 3),
        "mean_latency_ms": 1.0,
        "p95_latency_ms": 1.5,
        "effective_bandwidth_gib_s": 2.0,
        "data_validation": "passed",
    }]
    output = tmp_path / "transfer.json"

    benchmark.save_results_json(args, results, str(output))

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"]["block_counts"] == "1,2"
    assert payload["config"]["iterations"] == 20
    assert payload["results"] == results

    figure = tmp_path / "transfer.png"
    benchmark.save_results_figure(results, str(figure))
    assert figure.stat().st_size > 0
