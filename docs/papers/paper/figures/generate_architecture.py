"""Generate synchronized light and dark LMPool architecture figures."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
PAPER_PNG = Path(__file__).with_name("fig_architecture.png")
README_LIGHT_PNG = ROOT / "assets" / "fig_architecture.png"
README_DARK_PNG = ROOT / "assets" / "fig_architecture_dark.png"

LIGHT = {
    "background": "#FFFFFF",
    "text": "#202124",
    "muted": "#5F6368",
    "line": "#5F6368",
    "section": "#FAFBFC",
    "section_edge": "#AAB0B6",
    "section_text": "#202124",
    "rank": "#FAFBFC",
    "rank_edge": "#8A939B",
    "rank_text": "#202124",
    "rank_glyph": "#E8F1FF",
    "rank_glyph_edge": "#2684FC",
    "engine": "#F2E8FF",
    "engine_edge": "#A142F4",
    "scheduler": "#E8F1FF",
    "scheduler_edge": "#2684FC",
    "manager": "#FFF5CC",
    "manager_edge": "#FBBC04",
    "runner": "#FDEBE8",
    "runner_edge": "#EA4335",
    "cache": "#E6F7ED",
    "cache_edge": "#00AC47",
    "transfer": "#00AC47",
}

DARK = {
    "background": "#111418",
    "text": "#F8F9FA",
    "muted": "#BDC1C6",
    "line": "#BDC1C6",
    "section": "#191D22",
    "section_edge": "#697078",
    "section_text": "#F8F9FA",
    "rank": "#171B20",
    "rank_edge": "#7A838C",
    "rank_text": "#F8F9FA",
    "rank_glyph": "#122B4B",
    "rank_glyph_edge": "#5EA0FF",
    "engine": "#311A48",
    "engine_edge": "#C17DFF",
    "scheduler": "#122B4B",
    "scheduler_edge": "#5EA0FF",
    "manager": "#3B3107",
    "manager_edge": "#FBBC04",
    "runner": "#3B1815",
    "runner_edge": "#FF6B5F",
    "cache": "#103522",
    "cache_edge": "#36D477",
    "transfer": "#36D477",
}


def rounded_box(
    ax,
    x,
    y,
    width,
    height,
    *,
    face,
    edge,
    linewidth=1.35,
    radius=0.06,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def process_band(
    ax,
    x,
    y,
    width,
    height,
    title,
    palette,
    *,
    face_key,
    edge_key,
    title_color_key=None,
):
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        face=palette[face_key],
        edge=palette[edge_key],
        linewidth=1.25,
        radius=0.10,
    )
    ax.text(
        x + width / 2,
        y + height - 0.14,
        title,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=palette[title_color_key or edge_key],
        zorder=8,
    )


def centered_module(
    ax,
    x,
    y,
    width,
    height,
    title,
    subtitle,
    palette,
    *,
    face_key,
    edge_key,
    title_size=8.2,
    subtitle_size=6.5,
):
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        face=palette[face_key],
        edge=palette[edge_key],
        radius=0.045,
    )
    center_x = x + width / 2
    center_y = y + height / 2
    multiline_title = "\n" in title
    ax.text(
        center_x,
        center_y + (0.11 if multiline_title else 0.075),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=palette["text"],
        linespacing=1.0,
    )
    ax.text(
        center_x,
        center_y - (0.18 if multiline_title else 0.105),
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        color=palette["muted"],
    )


def label(
    ax,
    x,
    y,
    text,
    palette,
    *,
    color_key="muted",
    size=7.2,
    horizontal_alignment="center",
    weight="bold",
):
    ax.text(
        x,
        y,
        text,
        ha=horizontal_alignment,
        va="center",
        fontsize=size,
        fontweight=weight,
        color=palette[color_key],
        zorder=8,
    )


def orth_arrow(
    ax,
    points,
    palette,
    *,
    color_key="line",
    linewidth=1.3,
    linestyle="-",
    style="-|>",
    mutation_scale=12,
):
    path = MplPath(
        points,
        [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1),
    )
    patch = FancyArrowPatch(
        path=path,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=palette[color_key],
        shrinkA=0,
        shrinkB=0,
        zorder=6,
    )
    ax.add_patch(patch)
    return patch


def direct_arrow(
    ax,
    start,
    end,
    palette,
    *,
    color_key="line",
    linewidth=1.3,
    linestyle="-",
    style="-|>",
    mutation_scale=12,
):
    return orth_arrow(
        ax,
        [start, end],
        palette,
        color_key=color_key,
        linewidth=linewidth,
        linestyle=linestyle,
        style=style,
        mutation_scale=mutation_scale,
    )


def full_rank(ax, x, rank, palette, *, mirrored=False):
    y = 1.18
    width = 2.85
    height = 2.72
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        face=palette["rank"],
        edge=palette["rank_edge"],
        radius=0.075,
    )
    ax.text(
        x + width / 2,
        y + height - 0.18,
        f"DATA PLANE PROCESS @\nGPU RANK {rank}",
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=palette["rank_text"],
        linespacing=1.0,
    )

    module_width = 1.08
    module_height = 0.65
    left_x = x + 0.20
    right_x = x + width - 0.20 - module_width
    top_y = 2.71
    bottom_y = 1.51

    if mirrored:
        manager_x, scheduler_x = left_x, right_x
        runner_x, cache_x = left_x, right_x
    else:
        scheduler_x, manager_x = left_x, right_x
        cache_x, runner_x = left_x, right_x

    centered_module(
        ax,
        scheduler_x,
        top_y,
        module_width,
        module_height,
        "Local\nScheduler",
        "prefill / decode",
        palette,
        face_key="scheduler",
        edge_key="scheduler_edge",
        title_size=7.9,
        subtitle_size=6.4,
    )
    centered_module(
        ax,
        manager_x,
        top_y,
        module_width,
        module_height,
        "Local Block\nManager",
        "allocate / publish",
        palette,
        face_key="manager",
        edge_key="manager_edge",
        title_size=7.3,
        subtitle_size=6.1,
    )
    centered_module(
        ax,
        runner_x,
        bottom_y,
        module_width,
        module_height,
        "Model Runner",
        "execute / pack",
        palette,
        face_key="runner",
        edge_key="runner_edge",
        title_size=7.9,
        subtitle_size=6.4,
    )
    centered_module(
        ax,
        cache_x,
        bottom_y,
        module_width,
        module_height,
        "Physical KV\nCache",
        "local HBM",
        palette,
        face_key="cache",
        edge_key="cache_edge",
        title_size=7.3,
        subtitle_size=6.1,
    )

    direct_arrow(
        ax,
        (min(scheduler_x, manager_x) + module_width, top_y + module_height / 2),
        (max(scheduler_x, manager_x), top_y + module_height / 2),
        palette,
        style="<->",
    )
    direct_arrow(
        ax,
        (min(runner_x, cache_x) + module_width, bottom_y + module_height / 2),
        (max(runner_x, cache_x), bottom_y + module_height / 2),
        palette,
        style="<->",
    )
    orth_arrow(
        ax,
        [
            (scheduler_x + module_width / 2, top_y),
            (scheduler_x + module_width / 2, 2.48),
            (runner_x + module_width / 2, 2.48),
            (runner_x + module_width / 2, bottom_y + module_height),
        ],
        palette,
    )
    label(
        ax,
        x + width / 2,
        2.55,
        "scheduled batch + block table",
        palette,
        size=7.0,
    )

    return {
        "left": x,
        "right": x + width,
        "center_y": y + height / 2,
        "runner_left": runner_x,
        "runner_right": runner_x + module_width,
        "runner_y": bottom_y + module_height / 2,
        "scheduler_side": (
            scheduler_x + module_width if mirrored else scheduler_x,
            top_y + module_height / 2,
        ),
        "scheduler_branch_x": x + width + 0.05 if mirrored else x - 0.05,
        "manager_side": (
            manager_x if mirrored else manager_x + module_width,
            top_y + module_height / 2,
        ),
        "manager_branch_x": x - 0.05 if mirrored else x + width + 0.05,
    }


def omitted_rank_pairs(ax, center_x, palette):
    node_width = 0.50
    node_height = 0.22
    left_x = center_x - 0.70
    right_x = center_x + 0.20
    for center_y in (3.54, 3.22, 2.88, 2.18, 1.84, 1.52):
        node_y = center_y - node_height / 2
        for node_x in (left_x, right_x):
            rounded_box(
                ax,
                node_x,
                node_y,
                node_width,
                node_height,
                face=palette["rank_glyph"],
                edge=palette["rank_glyph_edge"],
                linewidth=0.9,
                radius=0.025,
            )
        direct_arrow(
            ax,
            (left_x + node_width, center_y),
            (right_x, center_y),
            palette,
            color_key="transfer",
            linewidth=1.4,
            style="<->",
            mutation_scale=7,
        )
    ax.text(
        center_x,
        2.53,
        "...",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=palette["muted"],
    )


def nvlink_pair(ax, left_runner, right_runner, palette):
    start = (left_runner["runner_right"], left_runner["runner_y"])
    end = (right_runner["runner_left"], right_runner["runner_y"])
    direct_arrow(
        ax,
        start,
        end,
        palette,
        color_key="transfer",
        linewidth=2.3,
        style="<->",
    )
    label(
        ax,
        (start[0] + end[0]) / 2,
        start[1] + 0.18,
        "NVLink",
        palette,
        color_key="transfer",
        size=7.8,
    )


def render(output, palette):
    fig, ax = plt.subplots(figsize=(16, 8.6))
    fig.patch.set_facecolor(palette["background"])
    ax.set_facecolor(palette["background"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    ax.text(
        8,
        8.30,
        "LMPool: KV-Aware Multi-GPU Serving",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=palette["text"],
    )
    ax.text(
        8,
        7.96,
        "Route for cache locality; transfer only for profitable cache fluidity",
        ha="center",
        va="center",
        fontsize=11.0,
        color=palette["muted"],
    )

    process_band(
        ax,
        0.50,
        6.90,
        15.00,
        0.85,
        "MAIN PROCESS",
        palette,
        face_key="section",
        edge_key="section_edge",
        title_color_key="section_text",
    )
    centered_module(
        ax,
        5.55,
        7.04,
        4.90,
        0.43,
        "LLMEngine",
        "ingress / launcher / result aggregation",
        palette,
        face_key="engine",
        edge_key="engine_edge",
        title_size=10.3,
        subtitle_size=7.6,
    )

    process_band(
        ax,
        0.50,
        5.15,
        15.00,
        1.15,
        "CONTROL PLANE PROCESS",
        palette,
        face_key="section",
        edge_key="section_edge",
        title_color_key="section_text",
    )
    centered_module(
        ax,
        2.95,
        5.38,
        4.15,
        0.56,
        "Global Scheduler",
        "routing / transfer admission",
        palette,
        face_key="scheduler",
        edge_key="scheduler_edge",
        title_size=9.8,
        subtitle_size=7.2,
    )
    centered_module(
        ax,
        8.90,
        5.38,
        4.15,
        0.56,
        "Global Block Manager",
        "page table / capacity / leases",
        palette,
        face_key="manager",
        edge_key="manager_edge",
        title_size=9.8,
        subtitle_size=7.2,
    )
    direct_arrow(
        ax,
        (7.10, 5.66),
        (8.90, 5.66),
        palette,
        style="<->",
    )
    label(
        ax,
        8.00,
        5.87,
        "prefix / capacity / reservations",
        palette,
        size=8.0,
    )

    process_band(
        ax,
        0.50,
        0.93,
        15.00,
        3.65,
        "DATA PLANE PROCESSES",
        palette,
        face_key="section",
        edge_key="section_edge",
        title_color_key="section_text",
    )

    rank0 = full_rank(ax, 0.80, "0", palette)
    rank1 = full_rank(ax, 4.25, "1", palette, mirrored=True)
    omitted_rank_pairs(ax, 8.00, palette)
    rankn2 = full_rank(ax, 8.90, "N-2", palette)
    rankn1 = full_rank(ax, 12.35, "N-1", palette, mirrored=True)

    nvlink_pair(ax, rank0, rank1, palette)
    nvlink_pair(ax, rankn2, rankn1, palette)

    # LLMEngine consults the control plane before dispatching the full sequence.
    orth_arrow(
        ax,
        [
            (8.00, 7.04),
            (8.00, 6.64),
            (5.025, 6.64),
            (5.025, 5.94),
        ],
        palette,
        style="<->",
    )
    label(
        ax,
        6.48,
        6.78,
        "route query / target GPU rank",
        palette,
        size=8.0,
    )

    # The selected full Sequence bypasses control-plane metadata storage.
    orth_arrow(
        ax,
        [
            (5.55, 7.255),
            (0.24, 7.255),
            (0.24, 3.275),
            (1.00, 3.275),
        ],
        palette,
    )
    label(
        ax,
        2.75,
        7.47,
        "Sequence dispatch to selected GPU rank",
        palette,
        size=8.0,
    )

    # A fan-out bus sends one control stream to every visible Local Scheduler.
    scheduler_bus_y = 4.72
    scheduler_routes = [
        (
            rank_geometry["scheduler_branch_x"],
            rank_geometry["scheduler_side"],
        )
        for rank_geometry in (rank0, rank1, rankn2, rankn1)
    ]
    scheduler_branch_xs = [route[0] for route in scheduler_routes]
    ax.add_line(
        Line2D(
            [5.025, 5.025],
            [5.38, scheduler_bus_y],
            color=palette["scheduler_edge"],
            linewidth=1.3,
            linestyle="--",
            zorder=5,
        )
    )
    ax.add_line(
        Line2D(
            [min(scheduler_branch_xs), max(scheduler_branch_xs)],
            [scheduler_bus_y, scheduler_bus_y],
            color=palette["scheduler_edge"],
            linewidth=1.3,
            linestyle="--",
            zorder=5,
        )
    )
    for branch_x, (scheduler_x, scheduler_y) in scheduler_routes:
        orth_arrow(
            ax,
            [
                (branch_x, scheduler_bus_y),
                (branch_x, scheduler_y),
                (scheduler_x, scheduler_y),
            ],
            palette,
            color_key="scheduler_edge",
            linestyle="--",
        )
    label(
        ax,
        3.10,
        5.07,
        "routing targets / transfer phases",
        palette,
        color_key="scheduler_edge",
        size=8.0,
    )
    # A separate fan-in bus carries every Local Block Manager snapshot upward.
    manager_bus_y = 4.95
    manager_routes = [
        (
            rank_geometry["manager_branch_x"],
            rank_geometry["manager_side"],
        )
        for rank_geometry in (rank0, rank1, rankn2, rankn1)
    ]
    manager_branch_xs = [route[0] for route in manager_routes]
    for branch_x, (manager_x, manager_y) in manager_routes:
        orth_arrow(
            ax,
            [
                (manager_x, manager_y),
                (branch_x, manager_y),
                (branch_x, manager_bus_y),
            ],
            palette,
            color_key="manager_edge",
            linestyle="--",
        )
    ax.add_line(
        Line2D(
            [min(manager_branch_xs), max(manager_branch_xs)],
            [manager_bus_y, manager_bus_y],
            color=palette["manager_edge"],
            linewidth=1.3,
            linestyle="--",
            zorder=5,
        )
    )
    direct_arrow(
        ax,
        (10.975, manager_bus_y),
        (10.975, 5.38),
        palette,
        color_key="manager_edge",
        linestyle="--",
    )
    label(
        ax,
        11.25,
        5.07,
        "versioned block / load snapshots",
        palette,
        color_key="manager_edge",
        size=7.8,
    )

    legend_y = 0.32
    ax.add_line(
        Line2D(
            [4.10, 4.58],
            [legend_y, legend_y],
            color=palette["line"],
            linewidth=1.3,
        )
    )
    ax.text(
        4.70,
        legend_y,
        "request / result",
        ha="left",
        va="center",
        fontsize=7.7,
        color=palette["muted"],
    )
    ax.add_line(
        Line2D(
            [6.80, 7.28],
            [legend_y, legend_y],
            color=palette["line"],
            linewidth=1.25,
            linestyle="--",
        )
    )
    ax.text(
        7.40,
        legend_y,
        "metadata / control",
        ha="left",
        va="center",
        fontsize=7.7,
        color=palette["muted"],
    )
    ax.add_line(
        Line2D(
            [9.80, 10.28],
            [legend_y, legend_y],
            color=palette["transfer"],
            linewidth=2.3,
        )
    )
    ax.text(
        10.40,
        legend_y,
        "KV payload",
        ha="left",
        va="center",
        fontsize=7.7,
        color=palette["muted"],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=260,
        bbox_inches="tight",
        facecolor=palette["background"],
    )
    plt.close(fig)


def main():
    for output in (PAPER_PNG, README_LIGHT_PNG):
        render(output, LIGHT)
    render(README_DARK_PNG, DARK)


if __name__ == "__main__":
    main()
