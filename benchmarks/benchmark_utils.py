"""Shared model-resolution and artifact metadata helpers for benchmarks."""

from __future__ import annotations

import datetime as dt
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformers import AutoConfig


_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


def model_display_name(model_name_or_path: str) -> str:
    """Return a readable model label for repository ids and HF snapshots."""
    raw = str(model_name_or_path)
    for part in Path(raw).parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/")
    if "/" in raw and not Path(raw).is_absolute():
        return raw
    return Path(raw).name or raw


def normalize_dtype_name(value: Any, *, auto_fallback: str = "float32") -> str:
    """Return a stable dtype name accepted by the benchmark runtime."""
    if value is None or str(value).lower() == "auto":
        value = auto_fallback
    name = str(value).lower().replace("torch.", "")
    aliases = {
        "fp16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
        "float": "float32",
    }
    name = aliases.get(name, name)
    if name not in _DTYPE_BYTES:
        raise ValueError(
            f"unsupported dtype {value!r}; expected auto, float16, bfloat16, or float32"
        )
    return name


def dtype_bytes(dtype_name: str) -> int:
    return _DTYPE_BYTES[normalize_dtype_name(dtype_name)]


def resolve_model_runtime_config(
    model_name_or_path: str,
    base_config: dict,
    *,
    dtype_override: str = "auto",
) -> tuple[dict, dict]:
    """Resolve custom ModelRunner arguments from a Hugging Face config.

    The custom model classes do not consume ``PretrainedConfig`` directly, so
    every structural field must be copied into the multiprocessing config
    before workers construct the model.
    """
    hf_config = AutoConfig.from_pretrained(model_name_or_path)
    model_type = str(getattr(hf_config, "model_type", "")).lower()
    architectures = list(getattr(hf_config, "architectures", None) or [])
    architecture = architectures[0] if architectures else ""
    identifier = f"{model_type} {architecture}".lower()
    if "qwen3" in identifier:
        family = "qwen3"
    elif "llama" in identifier:
        family = "llama"
    else:
        raise ValueError(
            f"unsupported model architecture for {model_name_or_path}: "
            f"model_type={model_type!r}, architectures={architectures!r}"
        )

    hidden_size = int(hf_config.hidden_size)
    num_heads = int(hf_config.num_attention_heads)
    head_dim = int(getattr(hf_config, "head_dim", hidden_size // num_heads))
    num_kv_heads = int(getattr(hf_config, "num_key_value_heads", num_heads))
    max_position = int(getattr(hf_config, "max_position_embeddings", 16384))
    serialized_config = hf_config.to_dict()
    configured_dtype = normalize_dtype_name(
        serialized_config.get("dtype", serialized_config.get("torch_dtype")),
        auto_fallback="float32",
    )
    dtype_name = (
        configured_dtype
        if str(dtype_override).lower() == "auto"
        else normalize_dtype_name(dtype_override)
    )

    eos_token_id = getattr(hf_config, "eos_token_id", base_config.get("eos", -1))
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0] if eos_token_id else base_config.get("eos", -1)
    if eos_token_id is None:
        eos_token_id = base_config.get("eos", -1)

    config = dict(base_config)
    config.update({
        "model_name_or_path": str(model_name_or_path),
        "model_architecture": architecture,
        "model_type": model_type,
        "vocab_size": int(hf_config.vocab_size),
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "intermediate_size": int(hf_config.intermediate_size),
        "num_layers": int(hf_config.num_hidden_layers),
        "tie_word_embeddings": bool(getattr(hf_config, "tie_word_embeddings", False)),
        "rms_norm_epsilon": float(getattr(hf_config, "rms_norm_eps", 1e-5)),
        "ffn_bias": bool(getattr(hf_config, "mlp_bias", False)),
        "torch_dtype": dtype_name,
        "kv_dtype_bytes": dtype_bytes(dtype_name),
        "eos": int(eos_token_id),
        "max_model_length": min(
            int(config.get("max_model_length", max_position)),
            max_position,
        ),
    })
    if family == "qwen3":
        config.update({
            "num_heads": num_heads,
            "qkv_bias": bool(getattr(hf_config, "attention_bias", False)),
            "base": int(getattr(hf_config, "rope_theta", 10000)),
            "max_position": max_position,
        })
    else:
        config.update({
            "num_qo_heads": num_heads,
            "has_attn_bias": bool(getattr(hf_config, "attention_bias", False)),
            "rope_base": int(getattr(hf_config, "rope_theta", 10000)),
            "max_position_embeddings": max_position,
        })

    model_metadata = {
        "name_or_path": str(model_name_or_path),
        "label": model_display_name(model_name_or_path),
        "model_type": model_type,
        "architecture": architecture,
        "dtype": dtype_name,
        "vocab_size": config["vocab_size"],
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": config["intermediate_size"],
        "num_layers": config["num_layers"],
        "max_position_embeddings": max_position,
        "kv_bytes_per_token": (
            2
            * config["num_layers"]
            * num_kv_heads
            * head_dim
            * config["kv_dtype_bytes"]
        ),
    }
    return config, model_metadata


def build_run_metadata(
    benchmark: str,
    args,
    *,
    model: dict | None = None,
    resolved_config: dict | None = None,
) -> dict:
    """Capture the exact invocation and resolved experiment contract."""

    def git_output(*git_args: str) -> str:
        try:
            return subprocess.run(
                ["git", *git_args],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    metadata = {
        "schema_version": 2,
        "benchmark": benchmark,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
        "git_revision": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        },
        "arguments": dict(vars(args)),
    }
    if model is not None:
        metadata["model"] = dict(model)
    if resolved_config is not None:
        excluded = {
            "distributed_init_method",
            "control_epoch",
        }
        metadata["resolved_config"] = {
            key: value
            for key, value in resolved_config.items()
            if key not in excluded and isinstance(value, (str, int, float, bool, list, dict, type(None)))
        }
    return metadata
