"""Aggregate per-link KV microbenchmarks into one logical-pair E2E profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .transfer_profile import (
        SUPPORTED_LATENCY_METRICS,
        build_transfer_latency_profile,
        parse_pair_list,
    )
except ImportError:
    from transfer_profile import (
        SUPPORTED_LATENCY_METRICS,
        build_transfer_latency_profile,
        parse_pair_list,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a size-aware KV transfer latency profile"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Per-physical-pair benchmark_kv_transfer JSON files, in logical-pair order.",
    )
    parser.add_argument(
        "--logical-pairs",
        required=True,
        help='E2E logical pair mapping, for example "0,1;2,3;4,5".',
    )
    parser.add_argument(
        "--latency-metric",
        choices=sorted(SUPPORTED_LATENCY_METRICS),
        default="p95_latency_ms",
        help="Conservative latency column used by the admission model.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        profile = build_transfer_latency_profile(
            args.inputs,
            parse_pair_list(args.logical_pairs),
            latency_metric=args.latency_metric,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot build transfer profile: {exc}") from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"saved transfer latency profile: {output}")


if __name__ == "__main__":
    main()
