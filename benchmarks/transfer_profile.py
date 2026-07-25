"""Build and validate size-aware KV transfer latency profiles."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


PROFILE_SCHEMA_VERSION = 1
SUPPORTED_LATENCY_METRICS = {
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
}


def parse_pair_list(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"invalid GPU pair {item!r}; expected A,B")
        pair = tuple(sorted((int(parts[0]), int(parts[1]))))
        if pair[0] == pair[1]:
            raise ValueError(f"GPU pair must contain two different ranks: {item!r}")
        pairs.append(pair)
    if len(set(pairs)) != len(pairs):
        raise ValueError("logical GPU pairs must be unique")
    return pairs


def _physical_pair(payload: dict) -> list[int]:
    visible = (
        payload.get("metadata", {})
        .get("environment", {})
        .get("CUDA_VISIBLE_DEVICES", "")
    )
    parts = [part.strip() for part in str(visible).split(",") if part.strip()]
    if len(parts) != 2:
        return []
    try:
        return [int(parts[0]), int(parts[1])]
    except ValueError:
        return []


def _profile_geometry(payload: dict, path: Path) -> dict:
    resolved = payload.get("metadata", {}).get("resolved_config", {})
    required = ("num_layers", "block_size", "num_kv_heads", "head_dim")
    missing = [name for name in required if name not in resolved]
    if missing:
        raise ValueError(f"{path} is missing resolved KV geometry: {missing}")

    results = payload.get("results") or []
    if not results:
        raise ValueError(f"{path} has no transfer results")
    per_block_sizes = {
        int(row["bytes_per_iteration"]) // int(row["num_transfer_blocks"])
        for row in results
    }
    if len(per_block_sizes) != 1:
        raise ValueError(f"{path} has inconsistent bytes per KV block")
    bytes_per_block = per_block_sizes.pop()
    if bytes_per_block <= 0:
        raise ValueError(f"{path} has invalid bytes per KV block")

    return {
        "num_layers": int(resolved["num_layers"]),
        "block_size": int(resolved["block_size"]),
        "num_kv_heads": int(resolved["num_kv_heads"]),
        "head_dim": int(resolved["head_dim"]),
        "dtype": str(resolved.get("torch_dtype", "")),
        "kv_dtype_bytes": int(resolved.get("kv_dtype_bytes", 0)),
        "bytes_per_block": bytes_per_block,
    }


def _monotonic_points(rows: list[dict], latency_metric: str) -> list[dict]:
    points: list[dict] = []
    previous_latency_ms = 0.0
    seen_blocks: set[int] = set()
    for row in sorted(rows, key=lambda item: int(item["num_transfer_blocks"])):
        blocks = int(row["num_transfer_blocks"])
        transfer_bytes = int(row["bytes_per_iteration"])
        raw_latency_ms = float(row[latency_metric])
        if blocks < 1 or transfer_bytes < 1 or raw_latency_ms <= 0:
            raise ValueError("profile points require positive blocks, bytes, and latency")
        if blocks in seen_blocks:
            raise ValueError(f"duplicate transfer result for {blocks} blocks")
        seen_blocks.add(blocks)
        latency_ms = max(previous_latency_ms, raw_latency_ms)
        points.append({
            "blocks": blocks,
            "bytes": transfer_bytes,
            "raw_latency_ms": raw_latency_ms,
            "latency_ms": latency_ms,
            "mean_latency_ms": float(row["mean_latency_ms"]),
            "p50_latency_ms": float(
                row.get("p50_latency_ms", row["mean_latency_ms"])
            ),
            "p95_latency_ms": float(row["p95_latency_ms"]),
        })
        previous_latency_ms = latency_ms
    return points


def build_transfer_latency_profile(
    input_paths: list[str | Path],
    logical_pairs: list[tuple[int, int]],
    *,
    latency_metric: str = "p95_latency_ms",
) -> dict:
    """Map physical-pair microbenchmarks to logical E2E pairs."""
    if latency_metric not in SUPPORTED_LATENCY_METRICS:
        raise ValueError(
            f"unsupported latency metric {latency_metric!r}; expected one of "
            f"{sorted(SUPPORTED_LATENCY_METRICS)}"
        )
    if len(input_paths) != len(logical_pairs):
        raise ValueError(
            f"got {len(input_paths)} input files for {len(logical_pairs)} logical pairs"
        )
    if not input_paths:
        raise ValueError("at least one transfer microbenchmark input is required")

    profile_pairs: dict[str, dict] = {}
    reference_geometry: dict | None = None
    reference_blocks: list[int] | None = None
    points_by_blocks: dict[int, list[dict]] = {}
    model_labels: set[str] = set()

    for raw_path, logical_pair in zip(input_paths, logical_pairs):
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        geometry = _profile_geometry(payload, path)
        if reference_geometry is None:
            reference_geometry = geometry
        elif geometry != reference_geometry:
            raise ValueError(
                f"{path} KV geometry does not match the first profile input"
            )

        points = _monotonic_points(payload["results"], latency_metric)
        block_counts = [point["blocks"] for point in points]
        if reference_blocks is None:
            reference_blocks = block_counts
        elif block_counts != reference_blocks:
            raise ValueError(
                f"{path} block-count sweep {block_counts} does not match "
                f"{reference_blocks}"
            )
        for point in points:
            points_by_blocks.setdefault(point["blocks"], []).append(point)

        model_label = payload.get("metadata", {}).get("model", {}).get("label")
        if model_label:
            model_labels.add(str(model_label))
        pair_key = f"{logical_pair[0]},{logical_pair[1]}"
        profile_pairs[pair_key] = {
            "logical_pair": list(logical_pair),
            "physical_pair": _physical_pair(payload),
            "source": str(path),
            "points": points,
        }

    default_points: list[dict] = []
    previous_latency_ms = 0.0
    for blocks in sorted(points_by_blocks):
        pair_points = points_by_blocks[blocks]
        raw_latency_ms = statistics.median(
            point["raw_latency_ms"] for point in pair_points
        )
        latency_ms = max(previous_latency_ms, raw_latency_ms)
        default_points.append({
            "blocks": blocks,
            "bytes": pair_points[0]["bytes"],
            "raw_latency_ms": raw_latency_ms,
            "latency_ms": latency_ms,
        })
        previous_latency_ms = latency_ms

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_type": "piecewise_linear_kv_transfer_latency",
        "latency_metric": latency_metric,
        "model_labels": sorted(model_labels),
        "kv_geometry": reference_geometry,
        "default_points": default_points,
        "pairs": profile_pairs,
    }


def load_transfer_latency_profile(
    path: str | Path,
    *,
    expected_bytes_per_block: int | None = None,
    expected_pairs: list[tuple[int, int]] | None = None,
) -> dict:
    """Load a profile and reject model/topology mismatches before spawning."""
    profile_path = Path(path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"{profile_path} has unsupported transfer profile schema "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("profile_type") != "piecewise_linear_kv_transfer_latency":
        raise ValueError(f"{profile_path} is not a KV transfer latency profile")
    geometry = payload.get("kv_geometry") or {}
    if expected_bytes_per_block is not None:
        actual = int(geometry.get("bytes_per_block", 0))
        if actual != int(expected_bytes_per_block):
            raise ValueError(
                f"{profile_path} bytes_per_block={actual} does not match "
                f"runtime value {expected_bytes_per_block}"
            )
    if expected_pairs is not None:
        required = {
            f"{min(a, b)},{max(a, b)}"
            for a, b in expected_pairs
        }
        available = set((payload.get("pairs") or {}).keys())
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"{profile_path} is missing logical NVLink pairs: {missing}"
            )
    if not payload.get("default_points"):
        raise ValueError(f"{profile_path} contains no default latency points")
    return payload
