"""Draw the timetable templates and replication used in Section 5.5."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "results" / "figures"

COLORS = {
    "blue": "#3978B5",
    "red": "#D95F59",
    "green": "#2B8C6B",
    "amber": "#D99A16",
    "neutral": "#333333",
    "light": "#E9ECEF",
}


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 2.0,
    mutation_scale: float = 10,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            color=color,
            linewidth=linewidth,
            mutation_scale=mutation_scale,
            shrinkA=8,
            shrinkB=8,
            capstyle="round",
            joinstyle="round",
        )
    )


def node(
    ax: plt.Axes,
    xy: tuple[float, float],
    label: str,
    *,
    radius: float = 0.095,
) -> None:
    ax.add_patch(
        Circle(
            xy,
            radius=radius,
            facecolor="white",
            edgecolor=COLORS["neutral"],
            linewidth=0.85,
            zorder=3,
        )
    )
    ax.text(*xy, label, ha="center", va="center", fontsize=7.2, zorder=4)


def draw_corridor(ax: plt.Axes) -> None:
    upper = [(0.18, 0.65), (0.50, 0.65), (0.82, 0.65)]
    lower = [(0.18, 0.36), (0.50, 0.36), (0.82, 0.36)]
    for points, color in ((upper, COLORS["blue"]), (lower, COLORS["red"])):
        arrow(ax, points[0], points[1], color=color)
        arrow(ax, points[1], points[2], color=color)
    for index, xy in enumerate(upper, start=1):
        node(ax, xy, f"$s_{index}$")
    for index, xy in enumerate(lower, start=1):
        node(ax, xy, f"$s_{index}$")

    ax.text(0.50, 0.79, "Line A: dep. $0+10k$, run 5", color=COLORS["blue"],
            ha="center", fontsize=7.2)
    ax.text(0.50, 0.21, "Line B: dep. $3+10k$, run 6", color=COLORS["red"],
            ha="center", fontsize=7.2)
    ax.text(0.50, 0.06, "$k=0,\\ldots,3$; $U=6$; demands $(8,8,6)$",
            color=COLORS["neutral"], ha="center", fontsize=7.0)
    ax.set_title("Acyclic corridor template", fontsize=8.5, pad=5)


def draw_diamond(ax: plt.Axes) -> None:
    source = (0.10, 0.50)
    upper = (0.42, 0.70)
    lower = (0.42, 0.30)
    destination = (0.88, 0.50)

    arrow(ax, source, upper, color=COLORS["blue"])
    arrow(ax, source, lower, color=COLORS["red"])
    arrow(ax, upper, destination, color=COLORS["green"])
    arrow(ax, lower, destination, color=COLORS["amber"])
    for xy, label in (
        (source, "$s_A$"),
        (upper, "$h_1$"),
        (lower, "$h_2$"),
        (destination, "$s_D$"),
    ):
        node(ax, xy, label)

    ax.text(0.19, 0.77, "$F_1$: $0+10k$; 5", color=COLORS["blue"],
            ha="center", fontsize=6.8)
    ax.text(0.19, 0.18, "$F_2$: $2+10k$; 6", color=COLORS["red"],
            ha="center", fontsize=6.8)
    ax.text(0.75, 0.77, "$G_1$: $8+10k$; 5", color=COLORS["green"],
            ha="center", fontsize=6.8)
    ax.text(0.75, 0.18, "$G_2$: $9+10k$; 5", color=COLORS["amber"],
            ha="center", fontsize=6.8)
    ax.text(0.50, 0.03, "$k=0,\\ldots,3$; $U=6$; transfer 2; demands $(8,6)$",
            color=COLORS["neutral"], ha="center", fontsize=7.0)
    ax.set_title("Mixed transfer template", fontsize=8.5, pad=5)


def mini_component(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    color: str,
) -> None:
    width, height = 0.18, 0.22
    ax.add_patch(
        Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
        )
    )
    ax.plot(
        [x - 0.055, x, x + 0.055],
        [y - 0.035, y + 0.045, y - 0.015],
        color=color,
        linewidth=1.2,
        marker="o",
        markersize=2.3,
    )
    ax.text(x, y - 0.16, label, ha="center", va="top", fontsize=7.2)


def draw_replication(ax: plt.Axes) -> None:
    mini_component(ax, 0.14, 0.59, "$H$", COLORS["blue"])
    arrow(
        ax,
        (0.25, 0.59),
        (0.39, 0.59),
        color=COLORS["neutral"],
        linewidth=1.25,
        mutation_scale=9,
    )
    ax.text(0.32, 0.77, "replicate", ha="center", fontsize=7.0)
    for x, label, color in (
        (0.49, "$H_1$", COLORS["blue"]),
        (0.68, "$H_2$", COLORS["red"]),
        (0.91, "$H_m$", COLORS["green"]),
    ):
        mini_component(ax, x, 0.59, label, color)
    ax.text(0.795, 0.59, "$\\cdots$", ha="center", va="center", fontsize=10)

    ax.text(
        0.50,
        0.25,
        "Corridor: $|\\mathcal{P}|=12m$, $|\\mathcal{H}|=m$, removed $=0$",
        ha="center",
        fontsize=7.1,
        color=COLORS["blue"],
    )
    ax.text(
        0.50,
        0.10,
        "Transfer: $|\\mathcal{P}|=43m$, $|\\mathcal{H}|=m$, removed $=24m$",
        ha="center",
        fontsize=7.1,
        color=COLORS["red"],
    )
    ax.set_title("Independent operating-area copies", fontsize=8.5, pad=5)


def make_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.25),
        gridspec_kw={"width_ratios": [1.0, 1.08, 1.12]},
    )
    draw_corridor(axes[0])
    draw_diamond(axes[1])
    draw_replication(axes[2])

    for label, axis in zip(["a", "b", "c"], axes):
        axis.text(
            -0.06,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 0.92)
        axis.axis("off")

    figure.subplots_adjust(
        left=0.015,
        right=0.995,
        top=0.87,
        bottom=0.04,
        wspace=0.20,
    )
    return figure


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure = make_figure()
    figure.savefig(
        FIGURE_DIR / "scaling_instance_generation.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        FIGURE_DIR / "scaling_instance_generation.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        FIGURE_DIR / "scaling_instance_generation.svg",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
