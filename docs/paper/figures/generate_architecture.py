"""Generate synchronized light and dark architecture figures."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
PAPER_OUTPUT = Path(__file__).with_name("fig_architecture.png")
README_LIGHT_OUTPUT = ROOT / "assets" / "fig_architecture.png"
README_DARK_OUTPUT = ROOT / "assets" / "fig_architecture_dark.png"

LIGHT = {
    "background": "#ffffff",
    "text": "#17212b",
    "muted": "#66717a",
    "footer": "#37474f",
    "top_face": "#f2f4f6",
    "top_edge": "#4f5b66",
    "control_face": "#edf3fa",
    "control_title": "#23476b",
    "worker_face": "#e7f4eb",
    "worker_title": "#1c5d37",
    "scheduler_face": "#d7e8f7",
    "block_face": "#e4dcf2",
    "runner_face": "#fff0d4",
    "blue": "#4477aa",
    "green": "#228833",
    "purple": "#aa3377",
    "purple_arrow": "#6b4c8a",
    "orange": "#ee7733",
    "red": "#cc6677",
    "arrow": "#4f5b66",
}

DARK = {
    "background": "#0d1117",
    "text": "#e6edf3",
    "muted": "#9da7b1",
    "footer": "#b1bac4",
    "top_face": "#161b22",
    "top_edge": "#8b949e",
    "control_face": "#101d2b",
    "control_title": "#79c0ff",
    "worker_face": "#10251b",
    "worker_title": "#7ee787",
    "scheduler_face": "#132f4c",
    "block_face": "#32213c",
    "runner_face": "#3a2717",
    "blue": "#58a6ff",
    "green": "#56d364",
    "purple": "#d2a8ff",
    "purple_arrow": "#bc8cff",
    "orange": "#ffa657",
    "red": "#ff7b72",
    "arrow": "#b1bac4",
}


def box(
    ax,
    x,
    y,
    width,
    height,
    text,
    *,
    face,
    edge,
    text_color,
    size=13,
    weight="normal",
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color=text_color,
    )
    return patch


def annotated_box(
    ax,
    x,
    y,
    width,
    height,
    title,
    subtitle,
    *,
    face,
    edge,
    text_color,
    muted_color,
    title_size=13,
    subtitle_size=9.5,
):
    patch = box(
        ax,
        x,
        y,
        width,
        height,
        "",
        face=face,
        edge=edge,
        text_color=text_color,
    )
    center_x = x + width / 2
    center_y = y + height / 2
    ax.text(
        center_x,
        center_y + 0.017,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=text_color,
    )
    ax.text(
        center_x,
        center_y - 0.024,
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        color=muted_color,
    )
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color,
    label_face,
    text=None,
    text_offset=(0, 0),
    style="-|>",
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=1.6,
        color=color,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            (start[0] + end[0]) / 2 + text_offset[0],
            (start[1] + end[1]) / 2 + text_offset[1],
            text,
            ha="center",
            va="center",
            fontsize=9.2,
            color=color,
            bbox={"facecolor": label_face, "edgecolor": "none", "pad": 1.5},
        )
    return patch


def worker(ax, x, rank, palette):
    width = 0.245
    box(
        ax,
        x,
        0.15,
        width,
        0.28,
        "",
        face=palette["worker_face"],
        edge=palette["green"],
        text_color=palette["text"],
    )
    ax.text(
        x + width / 2,
        0.405,
        f"Data Plane Process - Rank {rank}",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=palette["worker_title"],
    )
    box(
        ax,
        x + 0.018,
        0.285,
        0.095,
        0.07,
        "Local\nScheduler",
        face=palette["scheduler_face"],
        edge=palette["blue"],
        text_color=palette["text"],
        size=10.5,
    )
    box(
        ax,
        x + 0.132,
        0.285,
        0.095,
        0.07,
        "Local Block\nManager",
        face=palette["block_face"],
        edge=palette["purple"],
        text_color=palette["text"],
        size=10.5,
    )
    annotated_box(
        ax,
        x + 0.018,
        0.185,
        0.209,
        0.065,
        "Model Runner",
        "Physical KV blocks",
        face=palette["runner_face"],
        edge=palette["orange"],
        text_color=palette["text"],
        muted_color=palette["muted"],
        title_size=10.2,
        subtitle_size=8.2,
    )
    arrow(
        ax,
        (x + 0.113, 0.32),
        (x + 0.132, 0.32),
        color=palette["arrow"],
        label_face=palette["background"],
    )
    arrow(
        ax,
        (x + 0.065, 0.285),
        (x + 0.065, 0.25),
        color=palette["arrow"],
        label_face=palette["background"],
    )
    return x + width / 2


def render(output: Path, palette: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(palette["background"])
    ax.set_facecolor(palette["background"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    annotated_box(
        ax,
        0.04,
        0.89,
        0.92,
        0.075,
        "LLMEngine",
        "API, ingress routing, process launcher/supervisor, and result aggregation",
        face=palette["top_face"],
        edge=palette["top_edge"],
        text_color=palette["text"],
        muted_color=palette["muted"],
        title_size=15,
        subtitle_size=9.8,
    )

    box(
        ax,
        0.07,
        0.59,
        0.86,
        0.23,
        "",
        face=palette["control_face"],
        edge=palette["blue"],
        text_color=palette["text"],
    )
    ax.text(
        0.5,
        0.785,
        "Independent Control Plane Process",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=palette["control_title"],
    )
    annotated_box(
        ax,
        0.13,
        0.64,
        0.27,
        0.105,
        "Global Scheduler",
        "Route and transfer-cost decisions",
        face=palette["scheduler_face"],
        edge=palette["blue"],
        text_color=palette["text"],
        muted_color=palette["muted"],
        title_size=12.5,
        subtitle_size=9.3,
    )
    annotated_box(
        ax,
        0.60,
        0.64,
        0.27,
        0.105,
        "Global Block Manager",
        "Page table and capacity snapshots",
        face=palette["block_face"],
        edge=palette["purple"],
        text_color=palette["text"],
        muted_color=palette["muted"],
        title_size=12.5,
        subtitle_size=9.3,
    )
    arrow(
        ax,
        (0.40, 0.692),
        (0.60, 0.692),
        color=palette["arrow"],
        label_face=palette["control_face"],
        text="authoritative metadata lookup",
        text_offset=(0, 0.026),
    )

    arrow(
        ax,
        (0.28, 0.89),
        (0.28, 0.82),
        color=palette["arrow"],
        label_face=palette["background"],
        text="route metadata",
        text_offset=(-0.07, 0),
    )
    arrow(
        ax,
        (0.72, 0.82),
        (0.72, 0.89),
        color=palette["arrow"],
        label_face=palette["background"],
        text="decision / health",
        text_offset=(0.075, 0),
    )

    ax.text(
        0.07,
        0.545,
        "Distributed Data Plane: one identical process per visible GPU",
        ha="left",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=palette["worker_title"],
    )
    centers = [
        worker(ax, 0.07, "0", palette),
        worker(ax, 0.378, "1", palette),
        worker(ax, 0.686, "N-1", palette),
    ]
    ax.text(
        0.65,
        0.29,
        "...",
        ha="center",
        va="center",
        fontsize=26,
        color=palette["muted"],
    )

    arrow(
        ax,
        (0.23, 0.59),
        (centers[0], 0.43),
        color=palette["arrow"],
        label_face=palette["background"],
        text="target rank +\ntransfer phase",
        text_offset=(-0.055, 0),
    )
    arrow(
        ax,
        (centers[1], 0.43),
        (0.50, 0.59),
        color=palette["arrow"],
        label_face=palette["background"],
        text="versioned block state\n+ heartbeat",
        text_offset=(0.085, 0),
    )
    arrow(
        ax,
        (centers[2], 0.43),
        (0.76, 0.59),
        color=palette["arrow"],
        label_face=palette["background"],
    )

    arrow(
        ax,
        (0.297, 0.2175),
        (0.396, 0.2175),
        color=palette["arrow"],
        label_face=palette["background"],
        style="<->",
    )
    ax.text(
        0.3465,
        0.125,
        "Packed KV transfer\nover one direct NVLink pair",
        ha="center",
        va="center",
        fontsize=8.8,
        color=palette["muted"],
    )
    ax.text(
        0.5,
        0.025,
        "Physical KV ownership stays local; the control plane coordinates metadata, reservations, routing, and transactional copy/move plans.",
        ha="center",
        va="center",
        fontsize=11,
        color=palette["footer"],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=220,
        bbox_inches="tight",
        facecolor=palette["background"],
    )
    plt.close(fig)


def main() -> None:
    render(PAPER_OUTPUT, LIGHT)
    render(README_LIGHT_OUTPUT, LIGHT)
    render(README_DARK_OUTPUT, DARK)


if __name__ == "__main__":
    main()
