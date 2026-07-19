import argparse
import json

import pytest

from benchmarks.benchmark_utils import (
    build_run_metadata,
    dtype_bytes,
    model_display_name,
    normalize_dtype_name,
    resolve_model_runtime_config,
)


def _write_qwen_config(path, *, hidden_size, intermediate_size):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_hidden_layers": 28,
            "max_position_embeddings": 40960,
            "rope_theta": 1000000,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "eos_token_id": 151645,
        }),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size"),
    [(1024, 3072), (2048, 6144)],
)
def test_resolve_qwen_runtime_config_uses_snapshot_metadata(
    tmp_path, hidden_size, intermediate_size
):
    snapshot = tmp_path / f"qwen-{hidden_size}"
    _write_qwen_config(
        snapshot,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    config, metadata = resolve_model_runtime_config(
        str(snapshot),
        {"max_model_length": 2048, "scale": 1},
    )

    assert config["hidden_size"] == hidden_size
    assert config["intermediate_size"] == intermediate_size
    assert config["torch_dtype"] == "bfloat16"
    assert config["kv_dtype_bytes"] == 2
    assert config["num_layers"] == 28
    assert metadata["kv_bytes_per_token"] == 2 * 28 * 8 * 128 * 2


def test_explicit_dtype_override_updates_kv_byte_contract(tmp_path):
    snapshot = tmp_path / "qwen"
    _write_qwen_config(snapshot, hidden_size=1024, intermediate_size=3072)

    config, _ = resolve_model_runtime_config(
        str(snapshot),
        {"max_model_length": 2048},
        dtype_override="float32",
    )

    assert config["torch_dtype"] == "float32"
    assert config["kv_dtype_bytes"] == 4


def test_dtype_names_are_normalized_and_validated():
    assert normalize_dtype_name("torch.bfloat16") == "bfloat16"
    assert normalize_dtype_name("auto", auto_fallback="float16") == "float16"
    assert dtype_bytes("fp32") == 4
    with pytest.raises(ValueError, match="unsupported dtype"):
        normalize_dtype_name("int8")


def test_model_display_name_handles_hugging_face_snapshot_paths():
    path = "/cache/models--Qwen--Qwen3-1.7B/snapshots/abcdef"

    assert model_display_name(path) == "Qwen/Qwen3-1.7B"
    assert model_display_name("Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B"


def test_run_metadata_records_exact_experiment_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(model_name_or_path="/models/qwen", repetitions=5)

    metadata = build_run_metadata(
        "benchmark_e2e",
        args,
        model={"architecture": "Qwen3ForCausalLM"},
        resolved_config={"hidden_size": 2048, "control_epoch": "volatile"},
    )

    assert metadata["schema_version"] == 2
    assert metadata["benchmark"] == "benchmark_e2e"
    assert metadata["arguments"]["repetitions"] == 5
    assert metadata["model"]["architecture"] == "Qwen3ForCausalLM"
    assert metadata["resolved_config"]["hidden_size"] == 2048
    assert "control_epoch" not in metadata["resolved_config"]
