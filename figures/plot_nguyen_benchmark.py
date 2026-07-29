"""Draw the complete Nguyen et al. timetable benchmark used in Section 5.3."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "results" / "figures"

COLORS = {
    "line1": "#3F73C5",
    "line2": "#D94B45",
    "line3": "#E0A100",
    "line4": "#27966A",
    "connector": "#343434",
    "priority": "#7A3E9D",
}

POSITIONS = {
    1: (0.40, 2.50),
    2: (0.40, 0.75),
    5: (1.50, 3.30),
    6: (1.18, 1.72),
    7: (1.25, 0.55),
    8: (2.05, 2.08),
    9: (3.00, 2.38),
    10: (4.55, 3.30),
    11: (3.78, 2.78),
    12: (3.48, 1.45),
    13: (5.82, 3.30),
    14: (5.55, 1.45),
    15: (7.15, 3.30),
    16: (7.03, 0.67),
    3: (6.62, 2.20),
    4: (8.28, 1.95),
}

TIMES = {
    5: "8:05",
    6: "8:15",
    7: "8:00",
    8: "8:20",
    9: "8:25",
    10: "8:35",
    11: "8:30",
    12: "8:30",
    13: "8:45",
    14: "8:50",
    15: "9:00",
    16: "9:00",
}


def add_arrow(
    ax: plt.Axes,
    start: int,
    end: int,
    *,
    color: str,
    linewidth: float = 2.2,
    zorder: int = 2,
) -> None:
    """Draw a directed segment while keeping arrowheads clear of the nodes."""
    patch = FancyArrowPatch(
        POSITIONS[start],
        POSITIONS[end],
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        color=color,
        shrinkA=12,
        shrinkB=12,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)


def add_priority_highlight(ax: plt.Axes, start: int, end: int) -> None:
    """Add a restrained halo behind a segment with a boarding-priority conflict."""
    patch = FancyArrowPatch(
        POSITIONS[start],
        POSITIONS[end],
        arrowstyle="-",
        linewidth=8.0,
        color=COLORS["priority"],
        alpha=0.17,
        shrinkA=9,
        shrinkB=9,
        capstyle="round",
        zorder=1,
    )
    ax.add_patch(patch)


def draw_event_node(ax: plt.Axes, node: int) -> None:
    x, y = POSITIONS[node]
    ax.add_patch(
        Circle(
            (x, y),
            radius=0.155,
            facecolor="white",
            edgecolor="#202020",
            linewidth=0.9,
            zorder=4,
        )
    )
    ax.text(x, y, str(node), ha="center", va="center", fontsize=8.2, zorder=5)


def draw_od_node(ax: plt.Axes, node: int) -> None:
    x, y = POSITIONS[node]
    size = 0.30
    ax.add_patch(
        Rectangle(
            (x - size / 2, y - size / 2),
            size,
            size,
            facecolor="#F7F7F7",
            edgecolor="#202020",
            linewidth=0.9,
            zorder=4,
        )
    )
    ax.text(x, y, str(node), ha="center", va="center", fontsize=8.2, zorder=5)


def annotate_times(ax: plt.Axes) -> None:
    offsets = {
        5: (0.00, 0.27),
        6: (-0.16, 0.25),
        7: (0.00, -0.27),
        8: (-0.02, -0.28),
        9: (-0.03, 0.28),
        10: (0.00, 0.27),
        11: (-0.08, 0.26),
        12: (0.00, -0.27),
        13: (0.00, 0.27),
        14: (0.00, -0.27),
        15: (0.00, 0.27),
        16: (0.00, -0.27),
    }
    for node, time in TIMES.items():
        x, y = POSITIONS[node]
        dx, dy = offsets[node]
        ax.text(
            x + dx,
            y + dy,
            time,
            ha="center",
            va="center",
            fontsize=7.6,
            color="#303030",
        )


def draw_network() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(7.25, 3.55))

    # Demand and destination connectors.
    for start, end in [
        (1, 5),
        (1, 8),
        (2, 6),
        (2, 7),
        (13, 3),
        (14, 3),
        (15, 4),
        (16, 4),
    ]:
        add_arrow(
            ax,
            start,
            end,
            color=COLORS["connector"],
            linewidth=1.35,
        )

    # Highlight the two shared segments that induce boarding priorities.
    add_priority_highlight(ax, 8, 9)
    add_priority_highlight(ax, 10, 13)

    routes = {
        "line1": [(5, 10), (10, 13), (13, 15)],
        "line2": [(6, 8), (8, 9), (9, 11), (11, 10)],
        "line3": [(9, 12), (12, 14)],
        "line4": [(7, 16)],
    }
    for line, segments in routes.items():
        for start, end in segments:
            add_arrow(ax, start, end, color=COLORS[line], linewidth=2.5)

    for node in range(5, 17):
        draw_event_node(ax, node)
    for node in [1, 2, 3, 4]:
        draw_od_node(ax, node)
    annotate_times(ax)

    # Direct labels avoid a separate legend and remain interpretable in grayscale.
    ax.text(2.82, 3.47, "Line 1", color=COLORS["line1"], fontsize=8.4)
    ax.text(1.73, 1.63, "Line 2", color=COLORS["line2"], fontsize=8.4)
    ax.text(4.25, 1.22, "Line 3", color=COLORS["line3"], fontsize=8.4)
    ax.text(4.15, 0.33, "Line 4", color=COLORS["line4"], fontsize=8.4)

    ax.text(
        2.54,
        2.00,
        "priority",
        color=COLORS["priority"],
        fontsize=7.0,
        ha="center",
        va="top",
    )
    ax.text(
        5.18,
        3.02,
        "priority",
        color=COLORS["priority"],
        fontsize=7.0,
        ha="center",
        va="top",
    )

    ax.set_xlim(0.05, 8.62)
    ax.set_ylim(0.08, 3.72)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.03)
    return fig


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure = draw_network()
    figure.savefig(
        FIGURE_DIR / "nguyen_complete_benchmark.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        FIGURE_DIR / "nguyen_complete_benchmark.svg",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        FIGURE_DIR / "nguyen_complete_benchmark.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
