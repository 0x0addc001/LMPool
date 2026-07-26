"""Generate synchronized light and dark KV-block lifecycle diagrams."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[3]
PAPER_PNG = Path(__file__).with_name("fig_kv_block_lifecycle.png")
PAPER_PDF = Path(__file__).with_name("fig_kv_block_lifecycle.pdf")
README_LIGHT = ROOT / "assets" / "fig_kv_block_lifecycle.png"
README_DARK = ROOT / "assets" / "fig_kv_block_lifecycle_dark.png"

LIGHT = {
    "background": "#FFFFFF",
    "text": "#202124",
    "muted": "#5F6368",
    "arrow": "#2684FC",
    "free_fill": "#E8F1FF",
    "free_edge": "#2684FC",
    "active_fill": "#E8F1FF",
    "active_edge": "#2684FC",
    "cache_fill": "#E8F1FF",
    "cache_edge": "#2684FC",
    "transfer_fill": "#E8F1FF",
    "transfer_edge": "#2684FC",
    "pending_fill": "#D8E8FF",
    "pending_edge": "#2684FC",
}

DARK = {
    "background": "#111418",
    "text": "#F8F9FA",
    "muted": "#BDC1C6",
    "arrow": "#5EA0FF",
    "free_fill": "#122B4B",
    "free_edge": "#5EA0FF",
    "active_fill": "#122B4B",
    "active_edge": "#5EA0FF",
    "cache_fill": "#122B4B",
    "cache_edge": "#5EA0FF",
    "transfer_fill": "#122B4B",
    "transfer_edge": "#5EA0FF",
    "pending_fill": "#16385F",
    "pending_edge": "#5EA0FF",
}


def state_box(
    ax,
    center,
    size,
    title,
    subtitle,
    palette,
    fill_key,
    edge_key,
    *,
    title_size=9.6,
    subtitle_size=7.6,
):
    x, y = center
    width, height = size
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            linewidth=1.45,
            edgecolor=palette[edge_key],
            facecolor=palette[fill_key],
        )
    )
    ax.text(
        x,
        y + 0.012,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=palette["text"],
    )
    ax.text(
        x,
        y - 0.019,
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        color=palette["muted"],
        linespacing=1.1,
    )


def decision(ax, center, size, text, palette, *, accent_key="pending_edge"):
    x, y = center
    width, height = size
    ax.add_patch(
        Polygon(
            [
                (x, y + height / 2),
                (x + width / 2, y),
                (x, y - height / 2),
                (x - width / 2, y),
            ],
            closed=True,
            linewidth=1.45,
            edgecolor=palette[accent_key],
            facecolor=palette["pending_fill"],
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.3,
        color=palette["text"],
        linespacing=1.08,
    )


def orth_arrow(
    ax,
    points,
    palette,
    *,
    color_key="arrow",
    label=None,
    label_position=None,
):
    path = MplPath(
        points,
        [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1),
    )
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.45,
            color=palette[color_key],
            shrinkA=0,
            shrinkB=0,
        )
    )
    if label:
        ax.text(
            *label_position,
            label,
            ha="center",
            va="center",
            fontsize=7.7,
            fontweight="bold",
            color=palette[color_key],
        )


def heading(ax, title, subtitle, palette):
    ax.text(
        0.5,
        0.958,
        title,
        ha="center",
        va="center",
        fontsize=16.5,
        fontweight="bold",
        color=palette["text"],
    )
    ax.text(
        0.5,
        0.918,
        subtitle,
        ha="center",
        va="center",
        fontsize=9.8,
        color=palette["muted"],
    )


def render(palette, outputs):
    fig, ax = plt.subplots(figsize=(15.2, 8.6))
    fig.patch.set_facecolor(palette["background"])
    ax.set_facecolor(palette["background"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    heading(
        ax,
        "KV Block Lifecycle",
        "Local metadata controls visibility; ModelRunner owns the physical K/V tensor.",
        palette,
    )

    ax.text(
        0.055,
        0.855,
        "Local allocation and prefix reuse",
        fontsize=10.2,
        fontweight="bold",
        color=palette["active_edge"],
        ha="left",
        va="center",
    )

    state_box(
        ax,
        (0.10, 0.735),
        (0.14, 0.10),
        "Free",
        "available ID\nno KV owner",
        palette,
        "free_fill",
        "free_edge",
    )
    state_box(
        ax,
        (0.31, 0.735),
        (0.18, 0.10),
        "Allocated / Writing",
        "generation advanced\nnot reusable",
        palette,
        "active_fill",
        "active_edge",
    )
    state_box(
        ax,
        (0.535, 0.735),
        (0.17, 0.10),
        "Ready / In Use",
        "kv_ready = true\nref_count > 0",
        palette,
        "active_fill",
        "active_edge",
    )
    state_box(
        ax,
        (0.80, 0.735),
        (0.22, 0.10),
        "Cached / Reclaimable",
        "ready prefix; ref_count = 0\nvisible until reclaimed",
        palette,
        "cache_fill",
        "cache_edge",
    )

    orth_arrow(
        ax,
        [(0.17, 0.735), (0.22, 0.735)],
        palette,
        label="allocate",
        label_position=(0.195, 0.775),
    )
    orth_arrow(
        ax,
        [(0.40, 0.735), (0.45, 0.735)],
        palette,
        label="publish",
        label_position=(0.425, 0.775),
    )
    orth_arrow(
        ax,
        [(0.62, 0.735), (0.69, 0.735)],
        palette,
        label="release",
        label_position=(0.655, 0.775),
    )
    orth_arrow(
        ax,
        [(0.80, 0.785), (0.80, 0.825), (0.535, 0.825), (0.535, 0.785)],
        palette,
        label="prefix hit: ref_count++",
        label_position=(0.668, 0.846),
    )
    orth_arrow(
        ax,
        [(0.80, 0.685), (0.80, 0.615), (0.10, 0.615), (0.10, 0.685)],
        palette,
        label="dependency-safe reclaim / eviction",
        label_position=(0.45, 0.596),
    )
    orth_arrow(
        ax,
        [(0.31, 0.685), (0.31, 0.655), (0.17, 0.655), (0.17, 0.685)],
        palette,
        label="partial release",
        label_position=(0.24, 0.641),
    )

    ax.text(
        0.055,
        0.535,
        "Transactional cross-GPU transfer",
        fontsize=10.2,
        fontweight="bold",
        color=palette["transfer_edge"],
        ha="left",
        va="center",
    )
    ax.text(
        0.945,
        0.535,
        "entry: ready source + reserved target",
        fontsize=8.3,
        color=palette["muted"],
        ha="right",
        va="center",
    )

    state_box(
        ax,
        (0.10, 0.405),
        (0.15, 0.10),
        "Prepared",
        "lock generation\nreserve target",
        palette,
        "transfer_fill",
        "transfer_edge",
    )
    state_box(
        ax,
        (0.29, 0.405),
        (0.16, 0.10),
        "Executing",
        "pack / send\nreceive / unpack",
        palette,
        "transfer_fill",
        "transfer_edge",
    )
    decision(
        ax,
        (0.47, 0.405),
        (0.15, 0.11),
        "Data path\nsucceeded?",
        palette,
        accent_key="transfer_edge",
    )
    state_box(
        ax,
        (0.65, 0.405),
        (0.15, 0.10),
        "Received / Hidden",
        "kv_ready = true\npending_publish",
        palette,
        "pending_fill",
        "pending_edge",
    )
    state_box(
        ax,
        (0.84, 0.405),
        (0.17, 0.10),
        "Published",
        "hash becomes routable\ntarget is cache-ready",
        palette,
        "cache_fill",
        "cache_edge",
    )
    state_box(
        ax,
        (0.46, 0.225),
        (0.15, 0.085),
        "Abort",
        "target -> Free\nsource restored",
        palette,
        "free_fill",
        "free_edge",
        title_size=9.2,
        subtitle_size=7.3,
    )
    decision(
        ax,
        (0.80, 0.225),
        (0.18, 0.10),
        "Source\nsemantics?",
        palette,
        accent_key="transfer_edge",
    )
    state_box(
        ax,
        (0.65, 0.075),
        (0.20, 0.08),
        "Copy Finalize",
        "retain source + unlock",
        palette,
        "cache_fill",
        "cache_edge",
        title_size=9.2,
        subtitle_size=7.3,
    )
    state_box(
        ax,
        (0.90, 0.075),
        (0.16, 0.08),
        "Move Finalize",
        "safe source -> Free",
        palette,
        "free_fill",
        "free_edge",
        title_size=9.2,
        subtitle_size=7.3,
    )

    transfer_key = "transfer_edge"
    orth_arrow(
        ax,
        [(0.175, 0.405), (0.21, 0.405)],
        palette,
        color_key=transfer_key,
        label="execute",
        label_position=(0.193, 0.437),
    )
    orth_arrow(
        ax,
        [(0.37, 0.405), (0.395, 0.405)],
        palette,
        color_key=transfer_key,
    )
    orth_arrow(
        ax,
        [(0.545, 0.405), (0.575, 0.405)],
        palette,
        color_key=transfer_key,
        label="Yes",
        label_position=(0.56, 0.438),
    )
    orth_arrow(
        ax,
        [(0.47, 0.35), (0.47, 0.285), (0.46, 0.285), (0.46, 0.2675)],
        palette,
        color_key=transfer_key,
        label="No",
        label_position=(0.50, 0.305),
    )
    orth_arrow(
        ax,
        [(0.725, 0.405), (0.755, 0.405)],
        palette,
        color_key=transfer_key,
        label="publish",
        label_position=(0.74, 0.437),
    )
    orth_arrow(
        ax,
        [(0.84, 0.355), (0.84, 0.295), (0.80, 0.295), (0.80, 0.275)],
        palette,
        color_key=transfer_key,
        label="finalize",
        label_position=(0.87, 0.315),
    )
    orth_arrow(
        ax,
        [(0.71, 0.225), (0.65, 0.225), (0.65, 0.115)],
        palette,
        color_key=transfer_key,
        label="Copy",
        label_position=(0.70, 0.245),
    )
    orth_arrow(
        ax,
        [(0.89, 0.225), (0.90, 0.225), (0.90, 0.115)],
        palette,
        color_key=transfer_key,
        label="Move",
        label_position=(0.88, 0.245),
    )

    ax.text(
        0.055,
        0.015,
        "Only published ready blocks are routable; reserved and received-but-hidden blocks remain invisible.",
        ha="left",
        va="center",
        fontsize=8.2,
        color=palette["muted"],
    )

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output,
            dpi=240,
            bbox_inches="tight",
            facecolor=palette["background"],
        )
    plt.close(fig)


def main():
    render(LIGHT, [PAPER_PNG, PAPER_PDF, README_LIGHT])
    render(DARK, [README_DARK])


if __name__ == "__main__":
    main()
