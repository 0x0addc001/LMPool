"""Generate the compact result figure used by report_20260720.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = ROOT / "benchmarks/results/paper/20260719T072508Z"
OUTPUT = Path(__file__).with_name("report_20260720_summary.png")
ABSOLUTE_OUTPUT = Path(__file__).with_name("report_20260720_absolute_metrics.png")
MODELS = ("qwen3-0.6b", "qwen3-1.7b")
MODEL_LABELS = ("Qwen3-0.6B", "Qwen3-1.7B")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def scenario(model: str, workload: str, name: str) -> dict:
    data = load_json(ARTIFACT_ROOT / model / workload / "summary.json")
    return data["results"][name]


def annotate_bars(axis, bars, suffix: str = "") -> None:
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            f"{value:.1f}{suffix}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Output PNG path (default: report figure next to this script).",
    )
    parser.add_argument(
        "--absolute-output",
        type=Path,
        default=ABSOLUTE_OUTPUT,
        help="Output path for the absolute-metrics PNG.",
    )
    return parser.parse_args()


def finish_axis(axis) -> None:
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_figure(figure, output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    try:
        display_path = output.relative_to(ROOT)
    except ValueError:
        display_path = output
    print(f"saved {display_path}")


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )

    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.5), constrained_layout=True)

    # A. Routing locality: compare the same six-GPU budget with and without KV-aware routing.
    x = np.arange(len(MODELS))
    width = 0.34
    baseline_reuse = [
        100 * scenario(model, "routing", "multi-gpu")["initial_cached_token_ratio"]
        for model in MODELS
    ]
    routing_reuse = [
        100
        * scenario(model, "routing", "multi-gpu-kv-routing")[
            "initial_cached_token_ratio"
        ]
        for model in MODELS
    ]
    bars_a = axes[0].bar(
        x - width / 2,
        baseline_reuse,
        width,
        label="Round-robin",
        color="#4C78A8",
    )
    bars_b = axes[0].bar(
        x + width / 2,
        routing_reuse,
        width,
        label="KV-aware routing",
        color="#F58518",
    )
    annotate_bars(axes[0], bars_a, "%")
    annotate_bars(axes[0], bars_b, "%")
    axes[0].set_title("(a) Routing locality")
    axes[0].set_ylabel("Cached prompt tokens (%)")
    axes[0].set_xticks(x, MODEL_LABELS)
    axes[0].set_ylim(0, 95)
    axes[0].legend(frameon=False, loc="upper left")

    # B. Transfer microbenchmark: mean and observed range over two models and three NVLink pairs.
    block_counts = (4, 8)
    bandwidths: dict[int, list[float]] = {count: [] for count in block_counts}
    for model in MODELS:
        for pair in ("0-1", "3-4", "5-6"):
            data = load_json(ARTIFACT_ROOT / model / "kv_transfer" / f"pair_{pair}.json")
            by_count = {row["num_transfer_blocks"]: row for row in data["results"]}
            for count in block_counts:
                bandwidths[count].append(by_count[count]["effective_bandwidth_gib_s"])

    transfer_means = np.array([np.mean(bandwidths[count]) for count in block_counts])
    lower = transfer_means - np.array([np.min(bandwidths[count]) for count in block_counts])
    upper = np.array([np.max(bandwidths[count]) for count in block_counts]) - transfer_means
    transfer_bars = axes[1].bar(
        np.arange(2),
        transfer_means,
        width=0.55,
        color=("#54A24B", "#E45756"),
        yerr=np.vstack((lower, upper)),
        capsize=5,
    )
    annotate_bars(axes[1], transfer_bars)
    axes[1].set_title("(b) NVLink transfer")
    axes[1].set_ylabel("Effective bandwidth (GiB/s)")
    axes[1].set_xticks(np.arange(2), ("4 blocks\n0.109 GiB", "8 blocks\n0.219 GiB"))
    axes[1].set_ylim(0, 35)

    # C. End-to-end result: positive values consistently mean an improvement over multi-gpu.
    metric_labels = ("Throughput", "Mean TTFT", "Mean E2E", "P90 E2E")
    gains = []
    for model in MODELS:
        baseline = scenario(model, "session_handoff", "multi-gpu")
        lmpool = scenario(model, "session_handoff", "multi-gpu-lmpool")
        gains.append(
            [
                100 * (lmpool["throughput_tok_s"] / baseline["throughput_tok_s"] - 1),
                100 * (1 - lmpool["mean_ttft_s"] / baseline["mean_ttft_s"]),
                100 * (1 - lmpool["mean_e2e_s"] / baseline["mean_e2e_s"]),
                100 * (1 - lmpool["p90_e2e_s"] / baseline["p90_e2e_s"]),
            ]
        )
    metric_x = np.arange(len(metric_labels))
    bars_c = axes[2].bar(
        metric_x - width / 2,
        gains[0],
        width,
        label=MODEL_LABELS[0],
        color="#B279A2",
    )
    bars_d = axes[2].bar(
        metric_x + width / 2,
        gains[1],
        width,
        label=MODEL_LABELS[1],
        color="#72B7B2",
    )
    annotate_bars(axes[2], bars_c, "%")
    annotate_bars(axes[2], bars_d, "%")
    axes[2].set_title("(c) LMPool session handoff")
    axes[2].set_ylabel("Improvement over multi-gpu (%)")
    axes[2].set_xticks(metric_x, metric_labels, rotation=0)
    axes[2].set_ylim(0, 50)
    axes[2].legend(frameon=False, loc="upper left")

    for axis in axes:
        finish_axis(axis)
    save_figure(figure, args.output)

    # Absolute values complement the relative summary above and make the paper
    # claims auditable in their original units.
    absolute_figure, absolute_axes = plt.subplots(
        2, 2, figsize=(10.8, 7.0), constrained_layout=True
    )
    routing_names = ("multi-gpu", "multi-gpu-kv-routing")
    routing_labels = ("Round-robin", "KV-aware routing")
    handoff_names = (
        "multi-gpu",
        "multi-gpu-kv-transfer",
        "multi-gpu-lmpool",
    )
    handoff_labels = ("Round-robin", "Transfer-only", "LMPool")

    def grouped_bars(axis, workload, names, labels, metric, scale, colors, ylabel):
        group_x = np.arange(len(MODELS))
        group_width = 0.78
        bar_width = group_width / len(names)
        maximum = 0.0
        for index, (name, label, color) in enumerate(zip(names, labels, colors)):
            offset = (index - (len(names) - 1) / 2) * bar_width
            values = [
                scale * scenario(model, workload, name)[metric] for model in MODELS
            ]
            maximum = max(maximum, *values)
            bars = axis.bar(
                group_x + offset,
                values,
                width=bar_width * 0.9,
                label=label,
                color=color,
            )
            annotate_bars(axis, bars)
        axis.set_xticks(group_x, MODEL_LABELS)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, maximum * 1.35)
        axis.legend(frameon=False, loc="upper left")
        finish_axis(axis)

    grouped_bars(
        absolute_axes[0, 0],
        "routing",
        routing_names,
        routing_labels,
        "throughput_tok_s",
        1.0,
        ("#4C78A8", "#F58518"),
        "Throughput (tokens/s)",
    )
    absolute_axes[0, 0].set_title("(a) Routing throughput")
    grouped_bars(
        absolute_axes[0, 1],
        "routing",
        routing_names,
        routing_labels,
        "mean_ttft_s",
        1000.0,
        ("#72B7B2", "#E45756"),
        "Mean TTFT (ms)",
    )
    absolute_axes[0, 1].set_title("(b) Routing latency")
    grouped_bars(
        absolute_axes[1, 0],
        "session_handoff",
        handoff_names,
        handoff_labels,
        "throughput_tok_s",
        1.0,
        ("#59A14F", "#EDC948", "#B07AA1"),
        "Throughput (tokens/s)",
    )
    absolute_axes[1, 0].set_title("(c) Session-handoff throughput")
    grouped_bars(
        absolute_axes[1, 1],
        "session_handoff",
        handoff_names,
        handoff_labels,
        "mean_ttft_s",
        1000.0,
        ("#76B7B2", "#FF9DA7", "#9C755F"),
        "Mean TTFT (ms)",
    )
    absolute_axes[1, 1].set_title("(d) Session-handoff latency")
    save_figure(absolute_figure, args.absolute_output)


if __name__ == "__main__":
    main()
