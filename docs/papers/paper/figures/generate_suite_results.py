#!/usr/bin/env python3
"""Render paper, slide, and README figures from one paper-suite directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUITE = ROOT / "benchmarks/results/paper/20260727T231622Z"
PAPER = ROOT / "docs/paper/figures"
SLIDES = ROOT / "docs/slide/figures"
ASSETS = ROOT / "assets"
PREVIEW_DIR: Path | None = None

COLORS = {
    "multi-gpu": "#5F6368",
    "multi-gpu-kv-routing": "#2684FC",
    "multi-gpu-kv-transfer": "#00AC47",
    "multi-gpu-lmpool": "#FBBC04",
}
LABELS = {
    "multi-gpu": "Round robin",
    "multi-gpu-kv-routing": "KV-aware routing",
    "multi-gpu-kv-transfer": "Transfer only",
    "multi-gpu-lmpool": "LMPool",
}
MODELS = (("qwen3-0.6b", "Qwen3-0.6B"), ("qwen3-1.7b", "Qwen3-1.7B"))


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelcolor": "#202124",
        "axes.edgecolor": "#9AA0A6",
        "xtick.color": "#3C4043",
        "ytick.color": "#3C4043",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig: plt.Figure, basename: str) -> None:
    output_dir = PREVIEW_DIR or PAPER
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{basename}.png"
    fig.savefig(png, dpi=260, bbox_inches="tight", facecolor="white")
    if PREVIEW_DIR:
        plt.close(fig)
        return
    for destination in (SLIDES / f"{basename}.png", ASSETS / f"{basename}.png"):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, destination)
    plt.close(fig)


def value(result: dict, name: str, scale: float = 1.0) -> tuple[float, float]:
    return result[name] * scale, result.get(f"{name}_ci95", 0.0) * scale


def label_bars(axis: plt.Axes, bars, errors: list[float], *, decimals: int = 1) -> None:
    """Annotate bar means above their confidence intervals without hiding the CI."""
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        axis.annotate(
            f"{height:.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2, height + error),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#202124",
        )


def plot_routing(suite: Path) -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 5.5), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.80, hspace=0.30, wspace=0.14)
    multipliers = ("1x", "3x", "5x")
    x = np.arange(len(multipliers))
    width = 0.34
    for row, (model_dir, model_label) in enumerate(MODELS):
        payloads = [load(suite / model_dir / "routing" / f"prefix_{m}.json") for m in multipliers]
        for scenario, offset in (("multi-gpu", -width / 2), ("multi-gpu-kv-routing", width / 2)):
            throughputs = [p["results"][scenario]["throughput_tok_s"] for p in payloads]
            throughput_ci = [p["results"][scenario]["throughput_tok_s_ci95"] for p in payloads]
            ttfts = [p["results"][scenario]["mean_ttft_s"] * 1000 for p in payloads]
            ttft_ci = [p["results"][scenario]["mean_ttft_s_ci95"] * 1000 for p in payloads]
            throughput_bars = axes[row, 0].bar(
                x + offset, throughputs, width, yerr=throughput_ci, capsize=2,
                color=COLORS[scenario], label=LABELS[scenario],
            )
            ttft_bars = axes[row, 1].bar(
                x + offset, ttfts, width, yerr=ttft_ci, capsize=2,
                color=COLORS[scenario],
            )
            label_bars(axes[row, 0], throughput_bars, throughput_ci)
            label_bars(axes[row, 1], ttft_bars, ttft_ci, decimals=0)
        axes[row, 0].set_title(f"{model_label}: Throughput")
        axes[row, 1].set_title(f"{model_label}: Mean TTFT")
        axes[row, 0].set_ylabel("Output throughput (tok/s)")
        axes[row, 1].set_ylabel("Mean TTFT (ms)")
        for axis in axes[row]:
            axis.set_xticks(x, [f"{m} prefix" for m in multipliers])
            axis.grid(axis="y", color="#E8EAED", linewidth=0.8)
            axis.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905),
               frameon=False, fontsize=8, ncols=2)
    fig.suptitle("KV-Aware Routing Under Increasing Shared-Prefix Length", fontsize=12,
                 fontweight="bold", y=0.98)
    save(fig, "fig_suite_routing")


def plot_skew(suite: Path) -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 5.6), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.80, hspace=0.30, wspace=0.14)
    scenarios = ("multi-gpu", "multi-gpu-kv-routing", "multi-gpu-kv-transfer", "multi-gpu-lmpool")
    x = np.arange(len(MODELS))
    width = 0.18
    for column, workload in enumerate(("load_skew", "memory_skew")):
        payloads = [load(suite / model_dir / workload / "summary.json") for model_dir, _ in MODELS]
        for index, scenario in enumerate(scenarios):
            offset = (index - 1.5) * width
            throughputs = [p["results"][scenario]["throughput_tok_s"] for p in payloads]
            throughput_ci = [p["results"][scenario]["throughput_tok_s_ci95"] for p in payloads]
            ttfts = [p["results"][scenario]["mean_ttft_s"] * 1000 for p in payloads]
            ttft_ci = [p["results"][scenario]["mean_ttft_s_ci95"] * 1000 for p in payloads]
            throughput_bars = axes[0, column].bar(
                x + offset, throughputs, width, yerr=throughput_ci, capsize=2,
                color=COLORS[scenario], label=LABELS[scenario],
            )
            ttft_bars = axes[1, column].bar(
                x + offset, ttfts, width, yerr=ttft_ci, capsize=2,
                color=COLORS[scenario],
            )
            label_bars(axes[0, column], throughput_bars, throughput_ci)
            label_bars(axes[1, column], ttft_bars, ttft_ci, decimals=0)
        axes[0, column].set_title(workload.replace("_", " ").title())
        axes[0, column].set_ylabel("Output throughput (tok/s)")
        axes[1, column].set_ylabel("Mean TTFT (ms)")
        for axis in axes[:, column]:
            axis.set_xticks(x, [label for _, label in MODELS])
            axis.grid(axis="y", color="#E8EAED", linewidth=0.8)
            axis.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905),
               frameon=False, fontsize=7.6, ncols=4)
    fig.suptitle("Skew Workloads: Throughput and First-Token Latency", fontsize=12,
                 fontweight="bold", y=0.98)
    save(fig, "fig_suite_skew")


def plot_transfer_profile(suite: Path) -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), constrained_layout=True)
    for axis, (model_dir, model_label) in zip(axes, MODELS):
        profile = load(suite / model_dir / "kv_transfer" / "latency_profile.json")
        for pair_index, (pair, data) in enumerate(sorted(profile["pairs"].items())):
            blocks = np.array([point["blocks"] for point in data["points"]])
            payload_gib = np.array([point["bytes"] / 1024 ** 3 for point in data["points"]])
            latency_s = np.array([point["mean_latency_ms"] / 1000 for point in data["points"]])
            bandwidth = payload_gib / latency_s
            line = axis.plot(
                blocks, bandwidth, marker="o", linewidth=1.8, markersize=4, label=f"Pair {pair}"
            )[0]
            for point_index, (block_count, bw) in enumerate(zip(blocks, bandwidth)):
                vertical_offset = 5 + (pair_index - 1) * 7
                axis.annotate(
                    f"{bw:.1f}",
                    (block_count, bw),
                    xytext=(0, vertical_offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=5.8,
                    color=line.get_color(),
                )
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4, 8, 16, 32, 64])
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.set_xlabel("KV blocks per plan")
        axis.set_ylabel("Effective bandwidth (GiB/s)")
        axis.set_title(model_label)
        axis.grid(color="#E8EAED", linewidth=0.8)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Packed NVLink KV Transfer Profile", fontsize=12, fontweight="bold")
    save(fig, "fig_suite_transfer_profile")


def main() -> None:
    global PREVIEW_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument(
        "--preview-dir",
        type=Path,
        help="Write candidate figures only to this directory without updating document assets.",
    )
    args = parser.parse_args()
    suite = args.suite.resolve()
    PREVIEW_DIR = args.preview_dir.resolve() if args.preview_dir else None
    for required in (
        suite / "qwen3-0.6b/routing/prefix_1x.json",
        suite / "qwen3-1.7b/load_skew/summary.json",
        suite / "qwen3-1.7b/memory_skew/summary.json",
    ):
        if not required.is_file():
            raise SystemExit(f"missing suite artifact: {required}")
    plot_routing(suite)
    plot_skew(suite)
    plot_transfer_profile(suite)
    print(f"generated suite figures from {suite}")


if __name__ == "__main__":
    main()
