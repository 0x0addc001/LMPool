import json

import pytest

from benchmarks import benchmark_kv_transfer as benchmark


def test_parse_block_counts_supports_a_deduplicated_sweep():
    assert benchmark.parse_block_counts("1,2,4,2,8", 16) == [1, 2, 4, 8]
    assert benchmark.parse_block_counts("", 4) == [4]


def test_parse_block_counts_rejects_non_positive_values():
    with pytest.raises(ValueError, match=">= 1"):
        benchmark.parse_block_counts("1,0,4", 2)


def test_transfer_benchmark_json_contains_metadata_and_results(tmp_path):
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

    metadata = {
        "schema_version": 2,
        "arguments": {"block_counts": "1,2", "iterations": 20},
    }
    benchmark.save_results_json(results, str(output), metadata=metadata)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"] == metadata
    assert payload["results"] == results

    figure = tmp_path / "transfer.png"
    benchmark.save_results_figure(results, str(figure))
    assert figure.stat().st_size > 0


def test_transfer_contract_defaults_to_qwen_geometry_without_model():
    args = benchmark.parse_args([])

    resolved, model_metadata, config = benchmark.resolve_transfer_contract(args)

    assert model_metadata is None
    assert resolved.num_layers == 28
    assert resolved.num_kv_heads == 8
    assert resolved.head_dim == 128
    assert resolved.resolved_dtype == "float16"
    assert config["torch_dtype"] == "float16"
