import json

import pytest

from benchmarks.transfer_profile import (
    add_transaction_residual_profile,
    build_transfer_latency_profile,
    load_transfer_latency_profile,
    parse_pair_list,
)


def write_microbenchmark(path, physical_pair, p95_values):
    bytes_per_block = 1024
    rows = []
    for blocks, p95_ms in zip((1, 2, 4), p95_values):
        rows.append({
            "num_transfer_blocks": blocks,
            "bytes_per_iteration": blocks * bytes_per_block,
            "mean_latency_ms": p95_ms * 0.8,
            "p50_latency_ms": p95_ms * 0.7,
            "p95_latency_ms": p95_ms,
        })
    payload = {
        "metadata": {
            "environment": {
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, physical_pair)),
            },
            "model": {"label": "Qwen/Qwen3-test"},
            "resolved_config": {
                "num_layers": 1,
                "block_size": 16,
                "num_kv_heads": 2,
                "head_dim": 8,
                "torch_dtype": "float16",
                "kv_dtype_bytes": 2,
            },
        },
        "results": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_profile_maps_pairs_and_enforces_monotonic_latency(tmp_path):
    first = tmp_path / "pair_0-1.json"
    second = tmp_path / "pair_3-4.json"
    write_microbenchmark(first, (0, 1), (5.0, 4.0, 9.0))
    write_microbenchmark(second, (3, 4), (6.0, 8.0, 10.0))

    profile = build_transfer_latency_profile(
        [first, second],
        [(0, 1), (2, 3)],
    )

    assert profile["kv_geometry"]["bytes_per_block"] == 1024
    assert profile["pairs"]["2,3"]["physical_pair"] == [3, 4]
    assert [
        point["latency_ms"]
        for point in profile["pairs"]["0,1"]["points"]
    ] == [5.0, 5.0, 9.0]
    assert [
        point["latency_ms"] for point in profile["default_points"]
    ] == [5.5, 6.0, 9.5]


def test_load_profile_rejects_geometry_and_pair_mismatch(tmp_path):
    source = tmp_path / "pair.json"
    output = tmp_path / "profile.json"
    write_microbenchmark(source, (0, 1), (5.0, 7.0, 9.0))
    profile = build_transfer_latency_profile([source], [(0, 1)])
    output.write_text(json.dumps(profile), encoding="utf-8")

    loaded = load_transfer_latency_profile(
        output,
        expected_bytes_per_block=1024,
        expected_pairs=[(0, 1)],
    )
    assert loaded["latency_metric"] == "p95_latency_ms"

    with pytest.raises(ValueError, match="bytes_per_block"):
        load_transfer_latency_profile(output, expected_bytes_per_block=2048)
    with pytest.raises(ValueError, match="missing logical NVLink pairs"):
        load_transfer_latency_profile(output, expected_pairs=[(2, 3)])


def test_parse_pair_list_rejects_duplicate_or_self_pairs():
    assert parse_pair_list("0,1;2,3") == [(0, 1), (2, 3)]
    with pytest.raises(ValueError, match="unique"):
        parse_pair_list("0,1;1,0")
    with pytest.raises(ValueError, match="different"):
        parse_pair_list("2,2")


def test_add_transaction_residual_profile_uses_pair_bucket_p95(tmp_path):
    micro = tmp_path / "pair.json"
    calibration = tmp_path / "calibration.json"
    write_microbenchmark(micro, (0, 1), (5.0, 7.0, 9.0))
    profile = build_transfer_latency_profile([micro], [(0, 1)])
    calibration.write_text(json.dumps({
        "results": {
            "multi-gpu-lmpool": {
                "transfer_placement_observations": [
                    {
                        "pair": "0,1",
                        "bucket_blocks": 2,
                        "bytes": 2048,
                        "residual_ms": 10.0,
                    },
                    {
                        "pair": "0,1",
                        "bucket_blocks": 2,
                        "bytes": 2048,
                        "residual_ms": 30.0,
                    },
                ],
            },
        },
    }), encoding="utf-8")

    calibrated = add_transaction_residual_profile(
        profile,
        [calibration],
        percentile=0.95,
    )

    residual = calibrated["transaction_residual_profile"]
    assert residual["pairs"]["0,1"]["points"][0]["samples"] == 2
    assert residual["pairs"]["0,1"]["points"][0]["residual_ms"] == pytest.approx(29.0)
    assert residual["default_points"][0]["residual_ms"] == pytest.approx(29.0)


def test_add_transaction_residual_profile_rejects_missing_observations(tmp_path):
    micro = tmp_path / "pair.json"
    calibration = tmp_path / "calibration.json"
    write_microbenchmark(micro, (0, 1), (5.0, 7.0, 9.0))
    profile = build_transfer_latency_profile([micro], [(0, 1)])
    calibration.write_text(json.dumps({
        "results": {"multi-gpu-lmpool": {}},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="no transfer placement observations"):
        add_transaction_residual_profile(profile, [calibration])
