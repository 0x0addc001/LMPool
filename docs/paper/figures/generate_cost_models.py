"""Generate separate routing and transfer cost-model figures."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[3]

LIGHT = {
    "background": "#ffffff",
    "text": "#17212b",
    "muted": "#52606d",
    "routing": "#4477aa",
    "transfer": "#aa3377",
}

DARK = {
    "background": "#0d1117",
    "text": "#e6edf3",
    "muted": "#b1bac4",
    "routing": "#58a6ff",
    "transfer": "#d2a8ff",
}

OUTPUTS = {
    "routing": {
        "paper": Path(__file__).with_name("fig_routing_cost_model.png"),
        "readme": ROOT / "assets/fig_routing_cost_model_dark.png",
    },
    "transfer": {
        "paper": Path(__file__).with_name("fig_transfer_cost_model.png"),
        "readme": ROOT / "assets/fig_transfer_cost_model_dark.png",
    },
}


def mix(color: str, background: str, amount: float = 0.87) -> str:
    color = color.lstrip("#")
    background = background.lstrip("#")
    rgb = [int(color[index : index + 2], 16) for index in (0, 2, 4)]
    bg = [int(background[index : index + 2], 16) for index in (0, 2, 4)]
    values = [
        round(channel * (1 - amount) + base * amount)
        for channel, base in zip(rgb, bg)
    ]
    return "#" + "".join(f"{value:02x}" for value in values)


def box(
    ax,
    center,
    size,
    title,
    body,
    palette,
    accent,
    *,
    title_size=11.0,
    body_size=9.0,
    weight="bold",
):
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.7,
        edgecolor=accent,
        facecolor=mix(accent, palette["background"]),
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y + height * 0.20,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight=weight,
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
        linespacing=1.18,
    )


def decision(ax, center, size, text, palette, accent, *, fontsize=9.4):
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
        edgecolor=accent,
        facecolor=mix(accent, palette["background"]),
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


def arrow(
    ax,
    start,
    end,
    palette,
    accent,
    *,
    label=None,
    label_position=None,
):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.45,
            color=accent,
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
            fontsize=8.7,
            fontweight="bold",
            color=accent,
            bbox={
                "facecolor": palette["background"],
                "edgecolor": "none",
                "pad": 1.2,
            },
        )


def title(ax, heading, subtitle, palette, accent):
    ax.text(
        0.5,
        0.96,
        heading,
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=accent,
    )
    ax.text(
        0.5,
        0.915,
        subtitle,
        ha="center",
        va="center",
        fontsize=10.3,
        color=palette["muted"],
    )


def setup(palette):
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    fig.patch.set_facecolor(palette["background"])
    ax.set_facecolor(palette["background"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def draw_routing(palette, output: Path):
    accent = palette["routing"]
    fig, ax = setup(palette)
    title(
        ax,
        "Routing Cost Model",
        "Projected work is measured in token-equivalent units for every feasible rank.",
        palette,
        accent,
    )
    box(
        ax,
        (0.5, 0.82),
        (0.68, 0.105),
        "Inputs for candidate rank g",
        "request: N prompt tokens, R blocks; prefix match: h_g blocks\n"
        "snapshot: free/effective capacity, waiting/running tokens and sequences",
        palette,
        accent,
    )
    decision(
        ax,
        (0.5, 0.68),
        (0.31, 0.12),
        "effective_capacity_g >=\nR - h_g ?",
        palette,
        accent,
    )
    box(
        ax,
        (0.16, 0.68),
        (0.20, 0.09),
        "Exclude rank",
        "cannot safely admit request",
        palette,
        accent,
        title_size=10.2,
        body_size=8.5,
    )
    box(
        ax,
        (0.20, 0.50),
        (0.26, 0.145),
        "Queue work Q_g",
        "1.0 x (waiting + pending tokens)\n"
        "+ 0.25 x running tokens\n"
        "+ 32 x (running + pending sequences)",
        palette,
        accent,
        body_size=8.5,
    )
    box(
        ax,
        (0.50, 0.50),
        (0.26, 0.145),
        "Missing prefill M_g",
        "new_g = max(0, R - h_g)\n"
        "M_g = min(N, new_g x block_size)\n"
        "prefix reuse lowers M_g directly",
        palette,
        accent,
        body_size=8.5,
    )
    box(
        ax,
        (0.80, 0.50),
        (0.26, 0.145),
        "Reclaim pressure P_g",
        "reclaim_g = max(0, new_g - free_g)\n"
        "P_g = reclaim_g x block_size\n"
        "        x reclaim_weight (default 0.5)",
        palette,
        accent,
        body_size=8.5,
    )
    box(
        ax,
        (0.5, 0.315),
        (0.68, 0.10),
        "Projected route cost",
        "C_route(g) = Q_g + prefill_weight x M_g + P_g     (prefill_weight default: 1.0)",
        palette,
        accent,
        body_size=9.2,
    )
    decision(
        ax,
        (0.5, 0.17),
        (0.41, 0.11),
        "owner pressure skew exceeds threshold\nand spill extra cost is bounded?",
        palette,
        accent,
        fontsize=9.1,
    )
    box(
        ax,
        (0.27, 0.055),
        (0.31, 0.07),
        "Choose minimum C_route",
        "prefix/topology score breaks equivalent-cost ties",
        palette,
        accent,
        title_size=9.8,
        body_size=7.9,
    )
    box(
        ax,
        (0.73, 0.055),
        (0.31, 0.07),
        "Bounded spill",
        "route current request to lower-pressure rank",
        palette,
        accent,
        title_size=9.8,
        body_size=7.9,
    )

    arrow(ax, (0.5, 0.767), (0.5, 0.742), palette, accent)
    arrow(ax, (0.345, 0.68), (0.265, 0.68), palette, accent, label="No")
    arrow(ax, (0.5, 0.62), (0.5, 0.59), palette, accent, label="Yes", label_position=(0.535, 0.61))
    arrow(ax, (0.45, 0.62), (0.23, 0.574), palette, accent)
    arrow(ax, (0.55, 0.62), (0.77, 0.574), palette, accent)
    arrow(ax, (0.20, 0.427), (0.41, 0.365), palette, accent)
    arrow(ax, (0.50, 0.427), (0.50, 0.365), palette, accent)
    arrow(ax, (0.80, 0.427), (0.59, 0.365), palette, accent)
    arrow(ax, (0.5, 0.265), (0.5, 0.225), palette, accent)
    arrow(ax, (0.40, 0.135), (0.30, 0.09), palette, accent, label="No", label_position=(0.35, 0.12))
    arrow(ax, (0.60, 0.135), (0.70, 0.09), palette, accent, label="Yes", label_position=(0.65, 0.12))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=palette["background"])
    plt.close(fig)


def draw_transfer(palette, output: Path):
    accent = palette["transfer"]
    fig, ax = setup(palette)
    title(
        ax,
        "Transfer Cost and Benefit Model",
        "Admission compares saved prefill time with calibrated pair-local movement cost.",
        palette,
        accent,
    )
    box(
        ax,
        (0.5, 0.82),
        (0.66, 0.105),
        "Candidate plan",
        "B blocks; L layers; block size S; H_kv heads; head dimension D; dtype bytes d\n"
        "source generations valid, destination capacity checked separately",
        palette,
        accent,
    )
    box(
        ax,
        (0.25, 0.65),
        (0.38, 0.125),
        "Payload and static wire estimate",
        "bytes = B x 2 x L x S x H_kv x D x d\n"
        "wire_ms = bytes / measured_pair_bandwidth\n"
        "T_static = (T_fixed + wire_ms x interference) x cost_weight",
        palette,
        accent,
        body_size=8.7,
    )
    box(
        ax,
        (0.75, 0.65),
        (0.38, 0.125),
        "Online pair observations",
        "source transfer extra-time EWMA\n"
        "dispatch-to-publish placement EWMA\n"
        "microbenchmark bandwidth initializes the prior",
        palette,
        accent,
        body_size=8.7,
    )
    box(
        ax,
        (0.5, 0.475),
        (0.66, 0.095),
        "Conservative transfer cost",
        "T_xfer = max(T_static, T_pair_EWMA, T_placement_EWMA)",
        palette,
        accent,
        body_size=9.4,
    )
    box(
        ax,
        (0.27, 0.305),
        (0.39, 0.12),
        "Foreground saved work",
        "reuse_hat = discounted historical chain reuse\n"
        "T_save_fg = reuse_hat x B x S x configured prefill_ms/token",
        palette,
        accent,
        body_size=8.6,
    )
    box(
        ax,
        (0.73, 0.305),
        (0.39, 0.12),
        "Background saved work",
        "forecast first qualifies a candidate\n"
        "T_save_bg = B x S x destination prefill_ms/token\n"
        "(one avoidable cold prefill; target then self-warms)",
        palette,
        accent,
        body_size=8.4,
    )
    decision(
        ax,
        (0.5, 0.155),
        (0.39, 0.11),
        "T_save >= benefit_ratio x T_xfer\nand all validity/capacity gates pass?",
        palette,
        accent,
        fontsize=9.1,
    )
    box(
        ax,
        (0.27, 0.05),
        (0.31, 0.065),
        "Reject / defer",
        "cache identical low-value rejection",
        palette,
        accent,
        title_size=9.8,
        body_size=7.9,
    )
    box(
        ax,
        (0.73, 0.05),
        (0.31, 0.065),
        "Admit transaction",
        "prepare -> execute -> publish -> finalize",
        palette,
        accent,
        title_size=9.8,
        body_size=7.9,
    )

    arrow(ax, (0.45, 0.767), (0.29, 0.713), palette, accent)
    arrow(ax, (0.55, 0.767), (0.71, 0.713), palette, accent)
    arrow(ax, (0.25, 0.587), (0.42, 0.523), palette, accent)
    arrow(ax, (0.75, 0.587), (0.58, 0.523), palette, accent)
    arrow(ax, (0.45, 0.427), (0.31, 0.365), palette, accent)
    arrow(ax, (0.55, 0.427), (0.69, 0.365), palette, accent)
    arrow(ax, (0.33, 0.245), (0.43, 0.205), palette, accent)
    arrow(ax, (0.67, 0.245), (0.57, 0.205), palette, accent)
    arrow(ax, (0.40, 0.12), (0.30, 0.083), palette, accent, label="No", label_position=(0.35, 0.105))
    arrow(ax, (0.60, 0.12), (0.70, 0.083), palette, accent, label="Yes", label_position=(0.65, 0.105))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=palette["background"])
    plt.close(fig)


def main() -> None:
    draw_routing(LIGHT, OUTPUTS["routing"]["paper"])
    draw_routing(DARK, OUTPUTS["routing"]["readme"])
    draw_transfer(LIGHT, OUTPUTS["transfer"]["paper"])
    draw_transfer(DARK, OUTPUTS["transfer"]["readme"])


if __name__ == "__main__":
    main()
