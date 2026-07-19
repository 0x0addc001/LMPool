"""Generate separate routing, prediction, and transfer decision flowcharts."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[3]

LIGHT = {
    "background": "#ffffff",
    "text": "#17212b",
    "muted": "#52606d",
    "routing": "#4477aa",
    "prediction": "#228833",
    "transfer": "#aa3377",
}

DARK = {
    "background": "#0d1117",
    "text": "#e6edf3",
    "muted": "#b1bac4",
    "routing": "#58a6ff",
    "prediction": "#56d364",
    "transfer": "#d2a8ff",
}

FLOW_SPECS = {
    "routing": {
        "paper": Path(__file__).with_name("fig_routing_decision.png"),
        "report": ROOT / "docs/reports/figures/report_20260720_routing_decision.png",
        "readme": ROOT / "assets/fig_routing_decision_dark.png",
    },
    "prediction": {
        "paper": Path(__file__).with_name("fig_hot_prefix_decision.png"),
        "report": ROOT / "docs/reports/figures/report_20260720_hot_prefix_decision.png",
        "readme": ROOT / "assets/fig_hot_prefix_decision_dark.png",
    },
    "transfer": {
        "paper": Path(__file__).with_name("fig_transfer_decision.png"),
        "report": ROOT / "docs/reports/figures/report_20260720_transfer_decision.png",
        "readme": ROOT / "assets/fig_transfer_decision_dark.png",
    },
}


def _mix_with_background(color: str, background: str, amount: float = 0.86) -> str:
    color = color.lstrip("#")
    background = background.lstrip("#")
    rgb = [int(color[index:index + 2], 16) for index in (0, 2, 4)]
    bg = [int(background[index:index + 2], 16) for index in (0, 2, 4)]
    mixed = [round(channel * (1 - amount) + base * amount) for channel, base in zip(rgb, bg)]
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def add_process(ax, center, size, text, palette, accent, *, fontsize=10.2, weight="normal"):
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.7,
        edgecolor=accent,
        facecolor=_mix_with_background(accent, palette["background"]),
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=palette["text"],
        linespacing=1.18,
    )


def add_decision(ax, center, size, text, palette, accent, *, fontsize=9.6):
    x, y = center
    width, height = size
    points = [
        (x, y + height / 2),
        (x + width / 2, y),
        (x, y - height / 2),
        (x - width / 2, y),
    ]
    patch = Polygon(
        points,
        closed=True,
        linewidth=1.7,
        edgecolor=accent,
        facecolor=_mix_with_background(accent, palette["background"]),
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=palette["text"],
        linespacing=1.12,
    )


def add_arrow(
    ax,
    start,
    end,
    accent,
    palette,
    *,
    label=None,
    label_position=None,
    connectionstyle="arc3,rad=0",
):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.55,
            color=accent,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        x, y = label_position or (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
        )
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color=accent,
            bbox={
                "facecolor": palette["background"],
                "edgecolor": "none",
                "pad": 1.3,
            },
        )


def add_title(ax, title, subtitle, palette, accent):
    ax.text(
        0.5,
        0.965,
        title,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=accent,
    )
    ax.text(
        0.5,
        0.925,
        subtitle,
        ha="center",
        va="center",
        fontsize=10.5,
        color=palette["muted"],
    )


def draw_routing(ax, palette):
    accent = palette["routing"]
    add_title(
        ax,
        "KV-aware Routing Decision",
        "The master chooses a worker before the full Sequence enters the data plane.",
        palette,
        accent,
    )
    add_process(ax, (0.5, 0.855), (0.34, 0.075), "Request metadata + complete-block hash chain", palette, accent)
    add_process(
        ax,
        (0.5, 0.745),
        (0.46, 0.085),
        "Read page table, health, pending/running work,\nand effective free/reclaimable capacity",
        palette,
        accent,
        fontsize=9.8,
    )
    add_decision(ax, (0.5, 0.61), (0.37, 0.13), "Any healthy GPU can\nadmit this request?", palette, accent)
    add_process(ax, (0.16, 0.61), (0.22, 0.075), "Backpressure / reject\n(no feasible rank)", palette, accent, fontsize=9.4)
    add_process(
        ax,
        (0.5, 0.47),
        (0.48, 0.09),
        "For each feasible GPU: find longest contiguous prefix;\nestimate queue + missing-prefill + reclaim cost",
        palette,
        accent,
        fontsize=9.7,
    )
    add_decision(ax, (0.5, 0.325), (0.38, 0.13), "Ready prefix owner\navailable?", palette, accent)
    add_process(
        ax,
        (0.19, 0.075),
        (0.28, 0.08),
        "Choose least projected-cost GPU;\nreserve + enqueue",
        palette,
        accent,
        fontsize=9.2,
        weight="bold",
    )
    add_decision(ax, (0.73, 0.185), (0.35, 0.13), "Owner overloaded and bounded\nspill is cheaper?", palette, accent, fontsize=9.2)
    add_process(ax, (0.53, 0.065), (0.23, 0.07), "Keep owner;\nreserve + enqueue", palette, accent, fontsize=9.2, weight="bold")
    add_process(ax, (0.85, 0.065), (0.23, 0.07), "Spill rank;\nreserve + enqueue", palette, accent, fontsize=9.2, weight="bold")

    add_arrow(ax, (0.5, 0.816), (0.5, 0.789), accent, palette)
    add_arrow(ax, (0.5, 0.702), (0.5, 0.675), accent, palette)
    add_arrow(ax, (0.315, 0.61), (0.272, 0.61), accent, palette, label="No", label_position=(0.294, 0.632))
    add_arrow(ax, (0.5, 0.545), (0.5, 0.515), accent, palette, label="Yes", label_position=(0.535, 0.535))
    add_arrow(ax, (0.5, 0.425), (0.5, 0.39), accent, palette)
    add_arrow(
        ax,
        (0.36, 0.285),
        (0.21, 0.116),
        accent,
        palette,
        label="No",
        label_position=(0.285, 0.205),
    )
    add_arrow(
        ax,
        (0.64, 0.285),
        (0.70, 0.247),
        accent,
        palette,
        label="Yes",
        label_position=(0.66, 0.255),
    )
    add_arrow(
        ax,
        (0.625, 0.16),
        (0.56, 0.10),
        accent,
        palette,
        label="No",
        label_position=(0.58, 0.142),
    )
    add_arrow(
        ax,
        (0.835, 0.13),
        (0.85, 0.102),
        accent,
        palette,
        label="Yes",
        label_position=(0.88, 0.125),
    )


def draw_prediction(ax, palette):
    accent = palette["prediction"]
    add_title(
        ax,
        "Hot-prefix Prediction Decision",
        "Prediction creates a candidate; it never bypasses admission or capacity checks.",
        palette,
        accent,
    )
    add_process(
        ax,
        (0.5, 0.845),
        (0.48, 0.09),
        "Observe complete-block access counts, route hits,\nand optional pending-ingress demand",
        palette,
        accent,
        fontsize=9.8,
    )
    add_process(
        ax,
        (0.5, 0.72),
        (0.45, 0.085),
        "Reconstruct maximal complete chains;\nselect deepest hot leaf and ordered ancestors",
        palette,
        accent,
        fontsize=9.8,
    )
    add_decision(ax, (0.5, 0.585), (0.39, 0.13), "Threshold or forecast\ndemand satisfied?", palette, accent)
    add_process(ax, (0.15, 0.585), (0.21, 0.07), "Wait for more evidence", palette, accent, fontsize=9.5)
    add_decision(
        ax,
        (0.5, 0.43),
        (0.43, 0.135),
        "Destination lacks replica and\nload/placement condition holds?",
        palette,
        accent,
        fontsize=9.3,
    )
    add_process(ax, (0.15, 0.43), (0.21, 0.075), "Skip duplicate or\nunneeded placement", palette, accent, fontsize=9.3)
    add_process(
        ax,
        (0.5, 0.285),
        (0.47, 0.08),
        "Estimate expected reuse; deduplicate ancestors;\nbatch candidates by directed NVLink pair",
        palette,
        accent,
        fontsize=9.5,
    )
    add_decision(
        ax,
        (0.5, 0.145),
        (0.43, 0.135),
        "Cooldown clear, pair idle, target space,\nand saved prefill > transfer cost x ratio?",
        palette,
        accent,
        fontsize=8.9,
    )
    add_process(ax, (0.16, 0.06), (0.23, 0.07), "Defer or negative-cache\nthis candidate", palette, accent, fontsize=9.1)
    add_process(ax, (0.81, 0.06), (0.25, 0.07), "Admit background copy plan;\nlease only after commit", palette, accent, fontsize=9.0, weight="bold")

    add_arrow(ax, (0.5, 0.8), (0.5, 0.763), accent, palette)
    add_arrow(ax, (0.5, 0.677), (0.5, 0.65), accent, palette)
    add_arrow(ax, (0.305, 0.585), (0.257, 0.585), accent, palette, label="No", label_position=(0.282, 0.607))
    add_arrow(ax, (0.5, 0.52), (0.5, 0.498), accent, palette, label="Yes", label_position=(0.535, 0.512))
    add_arrow(ax, (0.285, 0.43), (0.257, 0.43), accent, palette, label="No", label_position=(0.272, 0.452))
    add_arrow(ax, (0.5, 0.362), (0.5, 0.325), accent, palette, label="Yes", label_position=(0.535, 0.347))
    add_arrow(ax, (0.5, 0.245), (0.5, 0.212), accent, palette)
    add_arrow(
        ax,
        (0.285, 0.115),
        (0.245, 0.078),
        accent,
        palette,
        label="No",
        label_position=(0.265, 0.108),
    )
    add_arrow(
        ax,
        (0.715, 0.115),
        (0.735, 0.085),
        accent,
        palette,
        label="Yes",
        label_position=(0.745, 0.117),
    )


def draw_transfer(ax, palette):
    accent = palette["transfer"]
    add_title(
        ax,
        "Transfer Admission and Execution",
        "Foreground shortage and admitted background placement share one transactional data path.",
        palette,
        accent,
    )
    add_process(
        ax,
        (0.5, 0.85),
        (0.46, 0.085),
        "Trigger: actual foreground block shortage\nor admitted background copy candidate",
        palette,
        accent,
        fontsize=9.8,
    )
    add_decision(
        ax,
        (0.5, 0.71),
        (0.43, 0.135),
        "Complete, current source chain; direct pair;\ntarget capacity; no in-flight conflict?",
        palette,
        accent,
        fontsize=9.0,
    )
    add_process(ax, (0.15, 0.71), (0.21, 0.075), "Reject with reason;\napply cooldown", palette, accent, fontsize=9.3)
    add_decision(
        ax,
        (0.5, 0.55),
        (0.42, 0.13),
        "Estimated saved work exceeds\nmeasured transfer cost x ratio?",
        palette,
        accent,
        fontsize=9.4,
    )
    add_process(ax, (0.15, 0.55), (0.21, 0.075), "Recompute / evict locally;\ndo not transfer", palette, accent, fontsize=9.2)
    add_process(
        ax,
        (0.5, 0.415),
        (0.45, 0.08),
        "Prepare: lock source generations and\nreserve concrete destination block IDs",
        palette,
        accent,
        fontsize=9.6,
    )
    add_decision(ax, (0.5, 0.285), (0.36, 0.115), "Both workers prepared?", palette, accent)
    add_process(ax, (0.17, 0.285), (0.22, 0.07), "Abort and roll back\nlocks/reservations", palette, accent, fontsize=9.2)
    add_process(
        ax,
        (0.4, 0.16),
        (0.37, 0.085),
        "Pack [layers, K/V, blocks, block size, KV heads, head dim];\none pair-local NCCL send/recv; destination index_copy",
        palette,
        accent,
        fontsize=8.4,
    )
    add_decision(ax, (0.8, 0.16), (0.25, 0.11), "Data path\nsucceeded?", palette, accent, fontsize=9.0)
    add_process(ax, (0.86, 0.04), (0.21, 0.065), "Abort; destination\nremains invisible", palette, accent, fontsize=8.8)
    add_process(ax, (0.42, 0.04), (0.36, 0.065), "Publish + finalize: copy retains source;\nmove releases only safe source suffix", palette, accent, fontsize=8.7, weight="bold")

    add_arrow(ax, (0.5, 0.807), (0.5, 0.777), accent, palette)
    add_arrow(ax, (0.285, 0.71), (0.257, 0.71), accent, palette, label="No", label_position=(0.272, 0.732))
    add_arrow(ax, (0.5, 0.642), (0.5, 0.615), accent, palette, label="Yes", label_position=(0.535, 0.632))
    add_arrow(ax, (0.29, 0.55), (0.257, 0.55), accent, palette, label="No", label_position=(0.274, 0.572))
    add_arrow(ax, (0.5, 0.485), (0.5, 0.455), accent, palette, label="Yes", label_position=(0.535, 0.472))
    add_arrow(ax, (0.5, 0.375), (0.5, 0.342), accent, palette)
    add_arrow(ax, (0.32, 0.285), (0.282, 0.285), accent, palette, label="No", label_position=(0.3, 0.307))
    add_arrow(ax, (0.5, 0.228), (0.42, 0.202), accent, palette, label="Yes", label_position=(0.485, 0.218))
    add_arrow(ax, (0.585, 0.16), (0.675, 0.16), accent, palette)
    add_arrow(
        ax,
        (0.75, 0.125),
        (0.55, 0.072),
        accent,
        palette,
        label="Yes",
        label_position=(0.65, 0.105),
    )
    add_arrow(
        ax,
        (0.84, 0.112),
        (0.86, 0.072),
        accent,
        palette,
        label="No",
        label_position=(0.88, 0.108),
    )


DRAWERS = {
    "routing": draw_routing,
    "prediction": draw_prediction,
    "transfer": draw_transfer,
}


def render(kind: str, output: Path, palette: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 8.8))
    fig.patch.set_facecolor(palette["background"])
    ax.set_facecolor(palette["background"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    DRAWERS[kind](ax, palette)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=palette["background"])
    plt.close(fig)


def main() -> None:
    for kind, outputs in FLOW_SPECS.items():
        render(kind, outputs["paper"], LIGHT)
        render(kind, outputs["report"], LIGHT)
        render(kind, outputs["readme"], DARK)


if __name__ == "__main__":
    main()
