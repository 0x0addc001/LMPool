"""Generate synchronized light and dark routing and transfer cost diagrams."""

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
        "light": {
            **LIGHT_BASE,
            "accent": "#6F91AE",
            "fill": "#EAF2F8",
            "outcome": "#DCEAF4",
        },
        "dark": {
            **DARK_BASE,
            "accent": "#87AFCC",
            "fill": "#172A39",
            "outcome": "#1B3548",
        },
    },
    "transfer": {
        "light": {
            **LIGHT_BASE,
            "accent": "#60947B",
            "fill": "#EAF4EF",
            "outcome": "#DCEDE5",
        },
        "dark": {
            **DARK_BASE,
            "accent": "#79B797",
            "fill": "#172B23",
            "outcome": "#1B382B",
        },
    },
}

OUTPUTS = {
    "routing": {
        "paper_png": Path(__file__).with_name("fig_routing_cost_model.png"),
        "paper_pdf": Path(__file__).with_name("fig_routing_cost_model.pdf"),
        "readme_light": ROOT / "assets" / "fig_routing_cost_model.png",
        "readme_dark": ROOT / "assets" / "fig_routing_cost_model_dark.png",
    },
    "transfer": {
        "paper_png": Path(__file__).with_name("fig_transfer_cost_model.png"),
        "paper_pdf": Path(__file__).with_name("fig_transfer_cost_model.pdf"),
        "readme_light": ROOT / "assets" / "fig_transfer_cost_model.png",
        "readme_dark": ROOT / "assets" / "fig_transfer_cost_model_dark.png",
    },
}


def process(
    ax,
    center,
    size,
    title,
    body,
    palette,
    *,
    outcome=False,
    title_size=9.6,
    body_size=7.6,
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
            edgecolor=palette["accent"],
            facecolor=palette["outcome"] if outcome else palette["fill"],
        )
    )
    ax.text(
        x,
        y + height * 0.18,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=palette["text"],
    )
    ax.text(
        x,
        y - height * 0.12,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=palette["muted"],
        linespacing=1.12,
    )


def decision(ax, center, size, text, palette, *, fontsize=8.4):
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


def orth_arrow(
    ax,
    points,
    palette,
    *,
    label=None,
    label_position=None,
    arrowstyle="-|>",
):
    path = MplPath(
        points,
        [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1),
    )
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle=arrowstyle,
            mutation_scale=13,
            linewidth=1.45,
            color=palette["accent"],
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
            fontsize=7.8,
            fontweight="bold",
            color=palette["accent"],
        )


def setup(palette, title, subtitle):
    fig, ax = plt.subplots(figsize=(14.6, 8.4))
    fig.patch.set_facecolor(palette["background"])
    ax.set_facecolor(palette["background"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
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
    return fig, ax


def save(fig, outputs):
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output,
            dpi=240,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


def draw_routing(palette, outputs):
    fig, ax = setup(
        palette,
        "Routing Cost Model",
        "Choose the feasible rank with the least projected token-equivalent work.",
    )

    process(
        ax,
        (0.50, 0.825),
        (0.62, 0.10),
        "Candidate Rank Snapshot",
        "request tokens N, required blocks R, local prefix blocks h_g\n"
        "free capacity, queued/running tokens, active sequences",
        palette,
        title_size=10.0,
        body_size=7.8,
    )
    decision(
        ax,
        (0.50, 0.675),
        (0.28, 0.11),
        "enough effective capacity\nfor R - h_g blocks?",
        palette,
    )
    process(
        ax,
        (0.15, 0.675),
        (0.18, 0.08),
        "Exclude Rank",
        "unsafe admission",
        palette,
        outcome=True,
        title_size=9.0,
        body_size=7.2,
    )
    process(
        ax,
        (0.20, 0.495),
        (0.24, 0.13),
        "Queue Work  Q_g",
        "waiting + pending tokens\n"
        "+ 0.25 x running tokens\n"
        "+ 32 x active sequences",
        palette,
        body_size=7.4,
    )
    process(
        ax,
        (0.50, 0.495),
        (0.24, 0.13),
        "Missing Prefill  M_g",
        "new_g = max(0, R - h_g)\n"
        "M_g = min(N, new_g x block_size)\n"
        "local reuse directly reduces work",
        palette,
        body_size=7.4,
    )
    process(
        ax,
        (0.80, 0.495),
        (0.24, 0.13),
        "Reclaim Pressure  P_g",
        "reclaim_g = max(0, new_g - free_g)\n"
        "P_g = reclaim_g x block_size\n"
        "x reclaim_weight",
        palette,
        body_size=7.4,
    )
    process(
        ax,
        (0.50, 0.315),
        (0.62, 0.095),
        "Projected Route Cost",
        "C_route(g) = Q_g + prefill_weight x M_g + P_g",
        palette,
        title_size=10.0,
        body_size=8.3,
    )
    decision(
        ax,
        (0.50, 0.165),
        (0.38, 0.105),
        "owner pressure exceeds threshold\nand spill cost stays bounded?",
        palette,
        fontsize=8.2,
    )
    process(
        ax,
        (0.25, 0.055),
        (0.30, 0.07),
        "Choose Minimum Cost",
        "locality and topology break ties",
        palette,
        outcome=True,
        title_size=9.0,
        body_size=7.0,
    )
    process(
        ax,
        (0.75, 0.055),
        (0.30, 0.07),
        "Bounded Spill",
        "route to a lower-pressure rank",
        palette,
        outcome=True,
        title_size=9.0,
        body_size=7.0,
    )

    orth_arrow(ax, [(0.50, 0.775), (0.50, 0.73)], palette)
    orth_arrow(
        ax,
        [(0.36, 0.675), (0.24, 0.675)],
        palette,
        label="No",
        label_position=(0.30, 0.695),
    )
    orth_arrow(
        ax,
        [(0.50, 0.62), (0.50, 0.59), (0.20, 0.59), (0.20, 0.56)],
        palette,
        label="Yes",
        label_position=(0.535, 0.602),
    )
    orth_arrow(ax, [(0.50, 0.62), (0.50, 0.56)], palette)
    orth_arrow(ax, [(0.50, 0.62), (0.50, 0.59), (0.80, 0.59), (0.80, 0.56)], palette)
    orth_arrow(ax, [(0.20, 0.43), (0.20, 0.395), (0.35, 0.395), (0.35, 0.3625)], palette)
    orth_arrow(ax, [(0.50, 0.43), (0.50, 0.3625)], palette)
    orth_arrow(ax, [(0.80, 0.43), (0.80, 0.395), (0.65, 0.395), (0.65, 0.3625)], palette)
    orth_arrow(ax, [(0.50, 0.2675), (0.50, 0.2175)], palette)
    orth_arrow(
        ax,
        [(0.31, 0.165), (0.25, 0.165), (0.25, 0.09)],
        palette,
        label="No",
        label_position=(0.275, 0.185),
    )
    orth_arrow(
        ax,
        [(0.69, 0.165), (0.75, 0.165), (0.75, 0.09)],
        palette,
        label="Yes",
        label_position=(0.725, 0.185),
    )

    save(fig, outputs)


def draw_transfer(palette, outputs):
    fig, ax = setup(
        palette,
        "Transfer Cost and Benefit Model",
        "Admit movement only when calibrated transfer cost is lower than saved prefill work.",
    )

    process(
        ax,
        (0.50, 0.825),
        (0.62, 0.10),
        "Candidate Plan",
        "B blocks and tensor shape define payload bytes\n"
        "source generation and destination capacity are validated separately",
        palette,
        title_size=10.0,
        body_size=7.8,
    )
    process(
        ax,
        (0.25, 0.65),
        (0.38, 0.15),
        "Size-Aware Offline Prior",
        "bytes = B x 2 x L x S x H_kv x D x dtype_bytes\n"
        "piecewise-linear P95 latency over 1/2/4/8/16/32/64 blocks\n"
        "T_base = T_fixed + interference x T_data",
        palette,
        title_size=9.5,
        body_size=7.2,
    )
    process(
        ax,
        (0.75, 0.65),
        (0.38, 0.15),
        "Online Pair Observations",
        "bucket by next power-of-two plan size\n"
        "source residual EWMA + placement residual EWMA\n"
        "updates measured pair-local execution cost",
        palette,
        title_size=9.5,
        body_size=7.2,
    )
    process(
        ax,
        (0.50, 0.465),
        (0.62, 0.095),
        "Conservative Transfer Cost",
        "T_xfer = max(T_base, T_data + delta_source, T_base + delta_place) x cost_weight",
        palette,
        title_size=10.0,
        body_size=7.8,
    )
    process(
        ax,
        (0.27, 0.295),
        (0.39, 0.115),
        "Foreground Saved Work",
        "discounted observed reuse estimates reuse_hat\n"
        "T_save_fg = reuse_hat x B x S x prefill_ms/token",
        palette,
        title_size=9.4,
        body_size=7.4,
    )
    process(
        ax,
        (0.73, 0.295),
        (0.39, 0.115),
        "Background Saved Work",
        "forecast qualifies a reusable prefix chain\n"
        "T_save_bg = B x S x destination prefill_ms/token",
        palette,
        title_size=9.4,
        body_size=7.4,
    )
    decision(
        ax,
        (0.50, 0.145),
        (0.38, 0.105),
        "saved work covers transfer cost\nand all validity gates pass?",
        palette,
        fontsize=8.2,
    )
    process(
        ax,
        (0.25, 0.045),
        (0.30, 0.065),
        "Reject / Defer",
        "cache repeated low-value decision",
        palette,
        outcome=True,
        title_size=9.0,
        body_size=6.9,
    )
    process(
        ax,
        (0.75, 0.045),
        (0.30, 0.065),
        "Admit Transaction",
        "prepare -> execute -> publish -> finalize",
        palette,
        outcome=True,
        title_size=9.0,
        body_size=6.9,
    )

    orth_arrow(ax, [(0.50, 0.775), (0.50, 0.75), (0.25, 0.75), (0.25, 0.725)], palette)
    orth_arrow(ax, [(0.50, 0.775), (0.50, 0.75), (0.75, 0.75), (0.75, 0.725)], palette)
    orth_arrow(ax, [(0.25, 0.575), (0.25, 0.545), (0.42, 0.545), (0.42, 0.5125)], palette)
    orth_arrow(ax, [(0.75, 0.575), (0.75, 0.545), (0.58, 0.545), (0.58, 0.5125)], palette)
    orth_arrow(ax, [(0.50, 0.4175), (0.50, 0.385), (0.27, 0.385), (0.27, 0.3525)], palette)
    orth_arrow(ax, [(0.50, 0.4175), (0.50, 0.385), (0.73, 0.385), (0.73, 0.3525)], palette)
    orth_arrow(
        ax,
        [(0.27, 0.2375), (0.27, 0.215), (0.50, 0.215)],
        palette,
        arrowstyle="-",
    )
    orth_arrow(
        ax,
        [(0.73, 0.2375), (0.73, 0.215), (0.50, 0.215)],
        palette,
        arrowstyle="-",
    )
    orth_arrow(ax, [(0.50, 0.215), (0.50, 0.1975)], palette)
    orth_arrow(
        ax,
        [(0.31, 0.145), (0.25, 0.145), (0.25, 0.0775)],
        palette,
        label="No",
        label_position=(0.275, 0.165),
    )
    orth_arrow(
        ax,
        [(0.69, 0.145), (0.75, 0.145), (0.75, 0.0775)],
        palette,
        label="Yes",
        label_position=(0.725, 0.165),
    )

    save(fig, outputs)


def main():
    routing = OUTPUTS["routing"]
    draw_routing(
        PALETTES["routing"]["light"],
        [routing["paper_png"], routing["paper_pdf"], routing["readme_light"]],
    )
    draw_routing(PALETTES["routing"]["dark"], [routing["readme_dark"]])

    transfer = OUTPUTS["transfer"]
    draw_transfer(
        PALETTES["transfer"]["light"],
        [transfer["paper_png"], transfer["paper_pdf"], transfer["readme_light"]],
    )
    draw_transfer(PALETTES["transfer"]["dark"], [transfer["readme_dark"]])


if __name__ == "__main__":
    main()
