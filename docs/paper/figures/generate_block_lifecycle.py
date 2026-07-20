"""Generate synchronized light and dark KV-block lifecycle diagrams."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[3]
PAPER_OUTPUT = Path(__file__).with_name("fig_kv_block_lifecycle.png")
README_OUTPUT = ROOT / "assets/fig_kv_block_lifecycle_dark.png"

LIGHT = {
    "background": "#ffffff",
    "text": "#17212b",
    "muted": "#52606d",
    "arrow": "#4f5b66",
    "free_face": "#f2f4f6",
    "free_edge": "#66717a",
    "active_face": "#dbe9f6",
    "active_edge": "#4477aa",
    "cache_face": "#e2f2e8",
    "cache_edge": "#228833",
    "transfer_face": "#fff0d4",
    "transfer_edge": "#ee7733",
    "pending_face": "#eee3f2",
    "pending_edge": "#aa3377",
}

DARK = {
    "background": "#0d1117",
    "text": "#e6edf3",
    "muted": "#b1bac4",
    "arrow": "#b1bac4",
    "free_face": "#161b22",
    "free_edge": "#8b949e",
    "active_face": "#132f4c",
    "active_edge": "#58a6ff",
    "cache_face": "#102f1e",
    "cache_edge": "#56d364",
    "transfer_face": "#3a2717",
    "transfer_edge": "#ffa657",
    "pending_face": "#32213c",
    "pending_edge": "#d2a8ff",
}


def state_box(
    ax,
    center,
    size,
    title,
    subtitle,
    palette,
    face_key,
    edge_key,
    *,
    title_size=11.2,
    subtitle_size=8.7,
):
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.7,
        edgecolor=palette[edge_key],
        facecolor=palette[face_key],
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y + 0.018,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=palette["text"],
    )
    ax.text(
        x,
        y - 0.027,
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        color=palette["muted"],
        linespacing=1.12,
    )


def decision(ax, center, size, text, palette):
    x, y = center
    width, height = size
    patch = Polygon(
        [
            (x, y + height / 2),
            (x + width / 2, y),
            (x, y - height / 2),
            (x - width / 2, y),
        ],
        closed=True,
        linewidth=1.7,
        edgecolor=palette["pending_edge"],
        facecolor=palette["pending_face"],
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9.3,
        color=palette["text"],
        linespacing=1.1,
    )


def arrow(
    ax,
    start,
    end,
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
            linewidth=1.45,
            color=palette["arrow"],
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
            fontsize=8.2,
            color=palette["muted"],
            bbox={
                "facecolor": palette["background"],
                "edgecolor": "none",
                "pad": 1.2,
            },
        )


def elbow_arrow(ax, points, palette, *, label=None, label_position=None):
    """Draw an orthogonal connector with one arrowhead at the destination."""
    xs = [point[0] for point in points[:-1]]
    ys = [point[1] for point in points[:-1]]
    ax.plot(xs, ys, color=palette["arrow"], linewidth=1.45)
    arrow(ax, points[-2], points[-1], palette, label=label, label_position=label_position)


def render(output: Path, palette: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 9.5))
    fig.patch.set_facecolor(palette["background"])
    ax.set_facecolor(palette["background"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "KV Block Lifecycle",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=palette["text"],
    )
    ax.text(
        0.5,
        0.918,
        "LocalBlockManager owns block state; ModelRunner writes or transfers the physical K/V tensor.",
        ha="center",
        va="center",
        fontsize=10.3,
        color=palette["muted"],
    )
    ax.text(
        0.04,
        0.845,
        "Normal allocation and prefix-cache path",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=palette["active_edge"],
    )

    state_box(
        ax,
        (0.10, 0.72),
        (0.16, 0.105),
        "Free",
        "ID is available\nno visible KV owner",
        palette,
        "free_face",
        "free_edge",
    )
    state_box(
        ax,
        (0.335, 0.72),
        (0.19, 0.105),
        "Allocated / Writing",
        "generation incremented\nKV not reusable yet",
        palette,
        "active_face",
        "active_edge",
    )
    state_box(
        ax,
        (0.58, 0.72),
        (0.18, 0.105),
        "Ready / In Use",
        "kv_ready = true\nref_count > 0",
        palette,
        "active_face",
        "active_edge",
    )
    state_box(
        ax,
        (0.835, 0.72),
        (0.2, 0.105),
        "Cached / Reclaimable",
        "ready prefix; ref_count = 0\nreusable until reclaimed",
        palette,
        "cache_face",
        "cache_edge",
    )

    arrow(ax, (0.18, 0.72), (0.24, 0.72), palette, label="allocate", label_position=(0.21, 0.79))
    arrow(
        ax,
        (0.43, 0.72),
        (0.49, 0.72),
        palette,
        label="KV complete + publish",
        label_position=(0.46, 0.79),
    )
    arrow(
        ax,
        (0.67, 0.72),
        (0.735, 0.72),
        palette,
        label="request release",
        label_position=(0.70, 0.79),
    )
    arrow(
        ax,
        (0.79, 0.785),
        (0.63, 0.785),
        palette,
        label="prefix hit; ref_count++",
        label_position=(0.71, 0.815),
        connectionstyle="arc3,rad=0",
    )
    elbow_arrow(
        ax,
        [(0.835, 0.667), (0.835, 0.625), (0.10, 0.625), (0.10, 0.667)],
        palette,
        label="dependency-safe reclaim / eviction",
        label_position=(0.47, 0.605),
    )
    elbow_arrow(
        ax,
        [(0.30, 0.667), (0.30, 0.65), (0.16, 0.65), (0.16, 0.667)],
        palette,
        label="partial release",
        label_position=(0.23, 0.638),
    )

    ax.plot([0.04, 0.96], [0.565, 0.565], color=palette["free_edge"], linewidth=0.9, alpha=0.65)
    ax.text(
        0.04,
        0.525,
        "Transactional cross-GPU transfer branch",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=palette["transfer_edge"],
    )
    ax.text(
        0.96,
        0.525,
        "entry: Ready/Cached source + Free target",
        ha="right",
        va="center",
        fontsize=9.3,
        color=palette["muted"],
    )

    state_box(
        ax,
        (0.13, 0.39),
        (0.19, 0.105),
        "Prepared",
        "source generation locked\nfree target IDs reserved",
        palette,
        "transfer_face",
        "transfer_edge",
    )
    state_box(
        ax,
        (0.34, 0.39),
        (0.18, 0.105),
        "Transfer Executing",
        "ModelRunner packs K/V\npair-local NCCL send/recv",
        palette,
        "transfer_face",
        "transfer_edge",
    )
    decision(ax, (0.53, 0.39), (0.16, 0.115), "Data path\nsucceeded?", palette)
    state_box(
        ax,
        (0.70, 0.39),
        (0.17, 0.105),
        "Received / Hidden",
        "kv_ready = true\npending_publish = true",
        palette,
        "pending_face",
        "pending_edge",
    )
    state_box(
        ax,
        (0.89, 0.39),
        (0.17, 0.105),
        "Published Replica",
        "hash becomes routable\ntarget enters cached-ready state",
        palette,
        "cache_face",
        "cache_edge",
    )
    decision(ax, (0.74, 0.215), (0.18, 0.105), "Source semantics?", palette)
    state_box(
        ax,
        (0.60, 0.095),
        (0.22, 0.09),
        "Copy Finalize",
        "retain source + unlock\nsource returns to prior ready state",
        palette,
        "cache_face",
        "cache_edge",
        title_size=10.4,
        subtitle_size=8.2,
    )
    state_box(
        ax,
        (0.88, 0.095),
        (0.18, 0.09),
        "Move Finalize",
        "reclaim safe source suffix\nsource enters Free",
        palette,
        "free_face",
        "free_edge",
        title_size=10.4,
        subtitle_size=8.2,
    )
    state_box(
        ax,
        (0.50, 0.215),
        (0.18, 0.09),
        "Abort",
        "target -> Free\nsource -> prior ready state",
        palette,
        "free_face",
        "free_edge",
        title_size=10.4,
        subtitle_size=8.2,
    )

    arrow(ax, (0.225, 0.39), (0.25, 0.39), palette, label="execute", label_position=(0.238, 0.418))
    arrow(ax, (0.43, 0.39), (0.45, 0.39), palette)
    arrow(
        ax,
        (0.53, 0.333),
        (0.50, 0.26),
        palette,
        label="No",
        label_position=(0.49, 0.30),
    )
    arrow(
        ax,
        (0.61, 0.39),
        (0.615, 0.39),
        palette,
        label="Yes",
        label_position=(0.61, 0.42),
    )
    arrow(
        ax,
        (0.785, 0.39),
        (0.805, 0.39),
        palette,
        label="publish",
        label_position=(0.795, 0.42),
    )
    arrow(
        ax,
        (0.89, 0.337),
        (0.79, 0.255),
        palette,
        label="finalize source",
        label_position=(0.86, 0.29),
    )
    arrow(
        ax,
        (0.69, 0.176),
        (0.64, 0.14),
        palette,
        label="Copy",
        label_position=(0.655, 0.175),
    )
    arrow(
        ax,
        (0.79, 0.176),
        (0.84, 0.14),
        palette,
        label="Move",
        label_position=(0.825, 0.175),
    )

    ax.text(
        0.5,
        0.018,
        "Routing sees only published ready blocks. Reserved or received-but-unpublished destination blocks remain invisible; abort restores both local managers.",
        ha="center",
        va="center",
        fontsize=9.2,
        color=palette["muted"],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=palette["background"])
    plt.close(fig)


def main() -> None:
    render(PAPER_OUTPUT, LIGHT)
    render(README_OUTPUT, DARK)


if __name__ == "__main__":
    main()
