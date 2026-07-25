"""Generate orthogonal light and dark decision-flow figures."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[3]

LIGHT_BASE = {
    "background": "#FFFFFF",
    "text": "#293744",
    "muted": "#667482",
}
DARK_BASE = {
    "background": "#0E141A",
    "text": "#F1F5F7",
    "muted": "#A8B4BF",
}

PALETTES = {
    "routing": {
        "light": {**LIGHT_BASE, "accent": "#6F91AE", "fill": "#EAF2F8", "outcome": "#DCEAF4"},
        "dark": {**DARK_BASE, "accent": "#87AFCC", "fill": "#172A39", "outcome": "#1B3548"},
    },
    "placement": {
        "light": {**LIGHT_BASE, "accent": "#729987", "fill": "#EAF4EF", "outcome": "#DCEDE5"},
        "dark": {**DARK_BASE, "accent": "#84B49B", "fill": "#172B23", "outcome": "#1B382B"},
    },
    "transfer": {
        "light": {**LIGHT_BASE, "accent": "#8B7DA4", "fill": "#F0ECF5", "outcome": "#E6DFF0"},
        "dark": {**DARK_BASE, "accent": "#B19CCA", "fill": "#281F31", "outcome": "#342840"},
    },
}

OUTPUTS = {
    "routing": {
        "paper": Path(__file__).with_name("fig_routing_decision.png"),
        "paper_pdf": Path(__file__).with_name("fig_routing_decision.pdf"),
        "report": ROOT / "docs/reports/figures/report_20260720_routing_decision.png",
        "readme_light": ROOT / "assets/fig_routing_decision.png",
        "readme_dark": ROOT / "assets/fig_routing_decision_dark.png",
    },
    "placement": {
        "paper": Path(__file__).with_name("fig_hot_prefix_decision.png"),
        "paper_pdf": Path(__file__).with_name("fig_hot_prefix_decision.pdf"),
        "report": ROOT / "docs/reports/figures/report_20260720_hot_prefix_decision.png",
        "readme_light": ROOT / "assets/fig_hot_prefix_decision.png",
        "readme_dark": ROOT / "assets/fig_hot_prefix_decision_dark.png",
    },
    "transfer": {
        "paper": Path(__file__).with_name("fig_transfer_decision.png"),
        "paper_pdf": Path(__file__).with_name("fig_transfer_decision.pdf"),
        "report": ROOT / "docs/reports/figures/report_20260720_transfer_decision.png",
        "readme_light": ROOT / "assets/fig_transfer_decision.png",
        "readme_dark": ROOT / "assets/fig_transfer_decision_dark.png",
    },
}


def process(ax, center, size, title, subtitle, palette, *, outcome=False, fontsize=9.5):
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        linewidth=1.45,
        edgecolor=palette["accent"],
        facecolor=palette["outcome"] if outcome else palette["fill"],
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(
            x,
            y + 0.011,
            title,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=palette["text"],
        )
        ax.text(
            x,
            y - 0.016,
            subtitle,
            ha="center",
            va="center",
            fontsize=max(6.7, fontsize - 2.0),
            color=palette["muted"],
        )
    else:
        ax.text(
            x,
            y,
            title,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=palette["text"],
        )


def decision(ax, center, size, text, palette, *, fontsize=8.8):
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
            edgecolor=palette["accent"],
            facecolor=palette["fill"],
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=palette["text"],
        linespacing=1.08,
    )


def orth_arrow(ax, points, palette, *, label=None, label_position=None):
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    ax.add_patch(
        FancyArrowPatch(
            path=MplPath(points, codes),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.45,
            color=palette["accent"],
            shrinkA=0,
            shrinkB=0,
        )
    )
    if label:
        x, y = label_position
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color=palette["accent"],
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
        fontsize=9.4,
        color=palette["muted"],
    )


def draw_routing(ax, palette):
    heading(
        ax,
        "KV-Aware Routing",
        "Balance contiguous-prefix reuse against queue and capacity pressure",
        palette,
    )
    process(ax, (0.50, 0.840), (0.34, 0.058), "Request Metadata", "prefix hashes / prompt blocks / decode budget", palette)
    process(ax, (0.50, 0.745), (0.38, 0.058), "Read Global State", "page table / queue work / effective capacity", palette)
    decision(ax, (0.50, 0.630), (0.26, 0.095), "Any feasible\nrank?", palette)
    process(ax, (0.15, 0.630), (0.18, 0.052), "Backpressure", "no admissible rank", palette, outcome=True, fontsize=9.0)
    process(ax, (0.50, 0.505), (0.40, 0.064), "Score Feasible Ranks", "Croute = queue + missing prefill + reclaim", palette)
    decision(ax, (0.50, 0.385), (0.26, 0.095), "Ready prefix\nowner?", palette)
    process(ax, (0.15, 0.385), (0.20, 0.052), "Lowest-Cost Rank", "reserve + enqueue", palette, outcome=True, fontsize=8.8)
    decision(
        ax,
        (0.50, 0.235),
        (0.34, 0.105),
        "Owner overloaded and\nbounded spill cheaper?",
        palette,
        fontsize=8.5,
    )
    process(ax, (0.33, 0.075), (0.22, 0.056), "Route to Owner", "reserve + enqueue", palette, outcome=True, fontsize=8.9)
    process(ax, (0.67, 0.075), (0.22, 0.056), "Spill Rank", "reserve + enqueue", palette, outcome=True, fontsize=8.9)

    orth_arrow(ax, [(0.50, 0.811), (0.50, 0.774)], palette)
    orth_arrow(ax, [(0.50, 0.716), (0.50, 0.678)], palette)
    orth_arrow(ax, [(0.37, 0.630), (0.24, 0.630)], palette, label="No", label_position=(0.305, 0.646))
    orth_arrow(ax, [(0.50, 0.582), (0.50, 0.537)], palette, label="Yes", label_position=(0.535, 0.560))
    orth_arrow(ax, [(0.50, 0.473), (0.50, 0.433)], palette)
    orth_arrow(ax, [(0.37, 0.385), (0.25, 0.385)], palette, label="No", label_position=(0.31, 0.401))
    orth_arrow(ax, [(0.50, 0.338), (0.50, 0.288)], palette, label="Yes", label_position=(0.535, 0.313))
    orth_arrow(ax, [(0.33, 0.235), (0.33, 0.103)], palette, label="No", label_position=(0.30, 0.165))
    orth_arrow(ax, [(0.67, 0.235), (0.67, 0.103)], palette, label="Yes", label_position=(0.70, 0.165))


def draw_placement(ax, palette):
    heading(
        ax,
        "Background Placement",
        "Forecast reusable KV, then admit only a profitable replica",
        palette,
    )
    process(ax, (0.50, 0.840), (0.39, 0.058), "Observe Reuse Signals", "block access / route hits / ingress demand", palette)
    process(ax, (0.50, 0.745), (0.38, 0.058), "Build Resident Prefix Chain", "deepest hot leaf + complete ancestors", palette)
    decision(ax, (0.50, 0.630), (0.27, 0.095), "Hot or forecast\nthreshold met?", palette)
    process(ax, (0.15, 0.630), (0.18, 0.052), "Defer", "collect more evidence", palette, outcome=True, fontsize=9.0)
    decision(ax, (0.50, 0.505), (0.28, 0.095), "Replica needed at\ntarget rank?", palette)
    process(ax, (0.15, 0.505), (0.18, 0.052), "Skip", "duplicate placement", palette, outcome=True, fontsize=9.0)
    decision(ax, (0.50, 0.375), (0.34, 0.100), "Pair idle, cooldown clear,\nand target space?", palette, fontsize=8.4)
    process(ax, (0.14, 0.375), (0.22, 0.054), "Negative Cache", "defer repeated failure", palette, outcome=True, fontsize=8.6)
    process(ax, (0.50, 0.255), (0.38, 0.058), "Coalesce Pair Candidates", "deduplicate ancestors / batch by direction", palette)
    decision(ax, (0.50, 0.145), (0.32, 0.090), "Saved prefill >\ntransfer cost x safety?", palette, fontsize=8.4)
    process(ax, (0.31, 0.045), (0.22, 0.050), "Reject Candidate", "", palette, outcome=True, fontsize=8.7)
    process(ax, (0.69, 0.045), (0.22, 0.050), "Admit Copy Plan", "lease after commit", palette, outcome=True, fontsize=8.7)

    orth_arrow(ax, [(0.50, 0.811), (0.50, 0.774)], palette)
    orth_arrow(ax, [(0.50, 0.716), (0.50, 0.678)], palette)
    orth_arrow(ax, [(0.365, 0.630), (0.24, 0.630)], palette, label="No", label_position=(0.303, 0.646))
    orth_arrow(ax, [(0.50, 0.582), (0.50, 0.553)], palette, label="Yes", label_position=(0.535, 0.567))
    orth_arrow(ax, [(0.36, 0.505), (0.24, 0.505)], palette, label="No", label_position=(0.30, 0.521))
    orth_arrow(ax, [(0.50, 0.458), (0.50, 0.425)], palette, label="Yes", label_position=(0.535, 0.442))
    orth_arrow(ax, [(0.33, 0.375), (0.25, 0.375)], palette, label="No", label_position=(0.29, 0.391))
    orth_arrow(ax, [(0.50, 0.325), (0.50, 0.284)], palette, label="Yes", label_position=(0.535, 0.305))
    orth_arrow(ax, [(0.50, 0.226), (0.50, 0.190)], palette)
    orth_arrow(
        ax,
        [(0.34, 0.145), (0.31, 0.145), (0.31, 0.070)],
        palette,
        label="No",
        label_position=(0.325, 0.124),
    )
    orth_arrow(
        ax,
        [(0.66, 0.145), (0.69, 0.145), (0.69, 0.070)],
        palette,
        label="Yes",
        label_position=(0.675, 0.124),
    )


def draw_transfer(ax, palette):
    heading(
        ax,
        "Transactional KV Transfer",
        "Foreground shortage and background placement share one data path",
        palette,
    )
    decision(ax, (0.50, 0.835), (0.25, 0.095), "Actual block\nshortage?", palette)
    process(ax, (0.20, 0.720), (0.20, 0.056), "Foreground Plan", "transfer exact shortage", palette)
    decision(ax, (0.72, 0.720), (0.25, 0.095), "Background plan\nadmitted?", palette)
    process(ax, (0.90, 0.605), (0.16, 0.052), "No Transfer", "continue locally", palette, outcome=True, fontsize=8.8)
    process(ax, (0.65, 0.605), (0.20, 0.056), "Background Plan", "copy reusable chain", palette)
    decision(
        ax,
        (0.42, 0.485),
        (0.30, 0.100),
        "Current source, direct pair,\nand target capacity?",
        palette,
        fontsize=8.3,
    )
    process(ax, (0.12, 0.485), (0.16, 0.052), "Reject", "reason + cooldown", palette, outcome=True, fontsize=8.8)
    decision(ax, (0.42, 0.355), (0.28, 0.095), "Benefit exceeds\nmeasured cost?", palette)
    process(ax, (0.12, 0.355), (0.16, 0.052), "Local Fallback", "recompute / reclaim", palette, outcome=True, fontsize=8.5)
    process(ax, (0.42, 0.240), (0.28, 0.056), "Prepare", "lock source / reserve target", palette)
    decision(ax, (0.32, 0.120), (0.18, 0.080), "Both ranks\nprepared?", palette, fontsize=8.3)
    process(ax, (0.10, 0.120), (0.12, 0.050), "Abort", "rollback", palette, outcome=True, fontsize=8.5)
    process(ax, (0.54, 0.120), (0.14, 0.052), "Execute", "pack / send / unpack", palette, fontsize=8.6)
    decision(ax, (0.75, 0.120), (0.18, 0.080), "Data path\nsucceeded?", palette, fontsize=8.0)
    process(ax, (0.93, 0.120), (0.12, 0.050), "Abort", "target hidden", palette, outcome=True, fontsize=8.5)
    process(ax, (0.75, 0.025), (0.24, 0.046), "Publish + Finalize", "copy keeps source / move reclaims", palette, outcome=True, fontsize=8.4)

    orth_arrow(
        ax,
        [(0.375, 0.835), (0.20, 0.835), (0.20, 0.748)],
        palette,
        label="Yes",
        label_position=(0.285, 0.852),
    )
    orth_arrow(
        ax,
        [(0.625, 0.835), (0.72, 0.835), (0.72, 0.768)],
        palette,
        label="No",
        label_position=(0.67, 0.852),
    )
    orth_arrow(
        ax,
        [(0.845, 0.720), (0.90, 0.720), (0.90, 0.631)],
        palette,
        label="No",
        label_position=(0.875, 0.737),
    )
    orth_arrow(
        ax,
        [(0.72, 0.673), (0.72, 0.650), (0.65, 0.650), (0.65, 0.633)],
        palette,
        label="Yes",
        label_position=(0.685, 0.664),
    )
    orth_arrow(ax, [(0.20, 0.692), (0.20, 0.555), (0.42, 0.555), (0.42, 0.535)], palette)
    orth_arrow(ax, [(0.65, 0.577), (0.65, 0.555), (0.42, 0.555), (0.42, 0.535)], palette)
    orth_arrow(ax, [(0.27, 0.485), (0.20, 0.485)], palette, label="No", label_position=(0.235, 0.501))
    orth_arrow(ax, [(0.42, 0.435), (0.42, 0.403)], palette, label="Yes", label_position=(0.455, 0.419))
    orth_arrow(ax, [(0.28, 0.355), (0.20, 0.355)], palette, label="No", label_position=(0.24, 0.371))
    orth_arrow(ax, [(0.42, 0.308), (0.42, 0.268)], palette, label="Yes", label_position=(0.455, 0.288))
    orth_arrow(ax, [(0.42, 0.212), (0.42, 0.175), (0.32, 0.175), (0.32, 0.160)], palette)
    orth_arrow(ax, [(0.23, 0.120), (0.16, 0.120)], palette, label="No", label_position=(0.195, 0.136))
    orth_arrow(ax, [(0.41, 0.120), (0.47, 0.120)], palette, label="Yes", label_position=(0.44, 0.136))
    orth_arrow(ax, [(0.61, 0.120), (0.66, 0.120)], palette)
    orth_arrow(ax, [(0.84, 0.120), (0.87, 0.120)], palette, label="No", label_position=(0.855, 0.136))
    orth_arrow(ax, [(0.75, 0.080), (0.75, 0.048)], palette, label="Yes", label_position=(0.785, 0.064))


DRAWERS = {
    "routing": draw_routing,
    "placement": draw_placement,
    "transfer": draw_transfer,
}


def render(kind, output, palette):
    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    fig.patch.set_facecolor(palette["background"])
    ax.set_facecolor(palette["background"])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.01, 1)
    ax.axis("off")
    DRAWERS[kind](ax, palette)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight", facecolor=palette["background"])
    plt.close(fig)


def main():
    for kind, outputs in OUTPUTS.items():
        light = PALETTES[kind]["light"]
        dark = PALETTES[kind]["dark"]
        for key in ("paper", "paper_pdf", "report", "readme_light"):
            render(kind, outputs[key], light)
        render(kind, outputs["readme_dark"], dark)


if __name__ == "__main__":
    main()
