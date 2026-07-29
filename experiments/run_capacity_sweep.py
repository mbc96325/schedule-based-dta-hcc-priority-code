"""Section 5.4: demand-capacity mechanisms for UE existence and efficiency.

The experiment has two controlled parts.

1. Hold the mixed path-priority relation of Figure 10 fixed while varying
   demand and bottleneck capacity. Classical UE is solved exactly in every cell.
2. Let two passenger groups compete for one departure. The group with the
   smaller fallback delay boards first. Vary capacity and the delay incurred by
   the lower-priority group to measure the welfare loss relative to SO.
"""

from __future__ import annotations

import csv
import json
import os

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import networkx as nx
import numpy as np

from dta import (
    classical_ue_assignment,
    diagnostics,
    enumerate_paths,
    quasi_ue_assignment,
    system_optimum_assignment,
)
from dta.priority_graph import build_priority_graph

from mechanism_instances import (
    mixed_component_instance,
    priority_bottleneck_instance,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results", "sensitivity")
FIGURE_PATH = os.path.join(
    os.path.dirname(ROOT),
    "manuscript",
    "Figures",
    "demand_capacity_mechanisms.pdf",
)

TOL = 1e-7
MIXED_DEMAND = np.linspace(0.5, 5.0, 19)
MIXED_CAPACITY = np.linspace(0.5, 5.0, 19)
GROUP_DEMAND = 10.0
HIGH_PRIORITY_DELAY = 5.0
BOTTLENECK_CAPACITY = np.arange(1.0, 21.0, 1.0)
LOW_PRIORITY_DELAY = np.arange(5.0, 30.1, 2.5)


def run_mixed_phase() -> tuple[np.ndarray, list[dict]]:
    """Solve exact UE over a fixed-relation demand-capacity grid."""
    regimes = np.zeros((len(MIXED_DEMAND), len(MIXED_CAPACITY)))
    rows = []
    for row, demand in enumerate(MIXED_DEMAND):
        for column, capacity in enumerate(MIXED_CAPACITY):
            instance = mixed_component_instance(demand, capacity)
            pathset = enumerate_paths(instance)
            graph = build_priority_graph(instance, pathset)
            if nx.is_directed_acyclic_graph(graph):
                raise AssertionError("mixed-component graph unexpectedly acyclic")

            so = system_optimum_assignment(instance, pathset)
            ue = classical_ue_assignment(instance, pathset)
            que = quasi_ue_assignment(instance, pathset)
            expected_exists = demand <= capacity + TOL

            if not so.feasible or not que.feasible:
                raise AssertionError("fallback path should make every cell feasible")
            if ue.feasible != expected_exists:
                raise AssertionError(
                    f"unexpected UE result at q={demand}, U={capacity}"
                )
            if abs(que.total_cost - so.total_cost) > TOL:
                raise AssertionError("QUE and SO should coincide in this family")

            regimes[row, column] = 1 if ue.feasible else 0
            rows.append(
                {
                    "demand": float(demand),
                    "capacity": float(capacity),
                    "demand_capacity_ratio": float(demand / capacity),
                    "ue_exists": ue.feasible,
                    "so_cost": so.total_cost,
                    "ue_cost": ue.total_cost if ue.feasible else float("nan"),
                    "que_cost": que.total_cost,
                    "removed_relations": len(que.removed_edges),
                }
            )
    return regimes, rows


def run_priority_welfare() -> tuple[np.ndarray, list[dict]]:
    """Measure the cost of priority allocation relative to SO."""
    gap_per_passenger = np.zeros(
        (len(LOW_PRIORITY_DELAY), len(BOTTLENECK_CAPACITY))
    )
    rows = []
    for row, low_delay in enumerate(LOW_PRIORITY_DELAY):
        for column, capacity in enumerate(BOTTLENECK_CAPACITY):
            instance = priority_bottleneck_instance(
                GROUP_DEMAND,
                capacity,
                HIGH_PRIORITY_DELAY,
                low_delay,
            )
            pathset = enumerate_paths(instance)
            graph = build_priority_graph(instance, pathset)
            if not nx.is_directed_acyclic_graph(graph):
                raise AssertionError("priority-bottleneck graph should be acyclic")

            so = system_optimum_assignment(instance, pathset)
            ue = classical_ue_assignment(instance, pathset)
            que = quasi_ue_assignment(instance, pathset)
            if not (so.feasible and ue.feasible and que.feasible):
                raise AssertionError("priority-bottleneck assignment must be feasible")
            if abs(que.total_cost - ue.total_cost) > TOL:
                raise AssertionError("QUE should equal UE on the acyclic graph")

            displaced_seats = max(
                0.0,
                min(capacity, 2 * GROUP_DEMAND - capacity),
            )
            expected_gap = displaced_seats * (
                low_delay - HIGH_PRIORITY_DELAY
            )
            actual_gap = que.total_cost - so.total_cost
            if abs(actual_gap - expected_gap) > TOL:
                raise AssertionError(
                    f"unexpected cost gap at U={capacity}, hB={low_delay}"
                )

            per_passenger = actual_gap / (2 * GROUP_DEMAND)
            gap_per_passenger[row, column] = per_passenger
            rows.append(
                {
                    "demand_per_group": GROUP_DEMAND,
                    "capacity": float(capacity),
                    "capacity_total_demand_ratio": float(
                        capacity / (2 * GROUP_DEMAND)
                    ),
                    "high_priority_delay": HIGH_PRIORITY_DELAY,
                    "low_priority_delay": float(low_delay),
                    "so_cost": so.total_cost,
                    "ue_cost": ue.total_cost,
                    "que_cost": que.total_cost,
                    "que_so_gap": actual_gap,
                    "gap_per_passenger": per_passenger,
                }
            )
    return gap_per_passenger, rows


def plot_results(
    mixed_regime: np.ndarray,
    welfare_gap: np.ndarray,
) -> None:
    """Create the three-panel publication figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.45),
        gridspec_kw={"width_ratios": [1.0, 1.08, 1.0]},
    )

    ax = axes[0]
    existence_cmap = ListedColormap(["#D95F59", "#3B8F6A"])
    existence_norm = BoundaryNorm([-0.5, 0.5, 1.5], existence_cmap.N)
    ax.imshow(
        mixed_regime,
        origin="lower",
        aspect="auto",
        extent=[
            MIXED_CAPACITY[0],
            MIXED_CAPACITY[-1],
            MIXED_DEMAND[0],
            MIXED_DEMAND[-1],
        ],
        cmap=existence_cmap,
        norm=existence_norm,
        interpolation="nearest",
    )
    ax.plot(
        [MIXED_CAPACITY[0], MIXED_CAPACITY[-1]],
        [MIXED_CAPACITY[0], MIXED_CAPACITY[-1]],
        color="white",
        linewidth=1.2,
    )
    ax.text(3.75, 1.35, "UE exists", color="white", ha="center")
    ax.text(1.55, 3.90, "No UE", color="white", ha="center")
    ax.set_xlabel("Bottleneck capacity, $U$")
    ax.set_ylabel("Demand, $q$")
    ax.set_title("Fixed mixed priority relation")

    ax = axes[1]
    image = ax.imshow(
        welfare_gap,
        origin="lower",
        aspect="auto",
        extent=[
            BOTTLENECK_CAPACITY[0] / (2 * GROUP_DEMAND),
            BOTTLENECK_CAPACITY[-1] / (2 * GROUP_DEMAND),
            LOW_PRIORITY_DELAY[0],
            LOW_PRIORITY_DELAY[-1],
        ],
        cmap="YlOrRd",
        interpolation="nearest",
    )
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Capacity / total demand")
    ax.set_ylabel("Low-priority fallback delay")
    ax.set_title("Priority cost per passenger")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.03)
    colorbar.set_label("Generalized-cost units", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5)

    ax = axes[2]
    selected_delays = [10.0, 20.0, 30.0]
    curve_colors = ["#3978B5", "#E2812C", "#2B8C6B"]
    ratios = BOTTLENECK_CAPACITY / (2 * GROUP_DEMAND)
    for delay, color in zip(selected_delays, curve_colors):
        index = int(np.argmin(np.abs(LOW_PRIORITY_DELAY - delay)))
        curve = welfare_gap[index]
        ax.plot(
            ratios,
            curve,
            marker="o",
            markersize=2.7,
            linewidth=1.5,
            color=color,
            label=f"Delay = {delay:g}",
        )
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    ax.legend(
        loc="upper right",
        fontsize=6.4,
        frameon=True,
        framealpha=0.90,
        facecolor="white",
        edgecolor="none",
        labelspacing=0.12,
        handlelength=1.25,
        handletextpad=0.35,
        borderpad=0.25,
        borderaxespad=0.35,
    )
    ax.set_xlabel("Capacity / total demand")
    ax.set_ylabel("Cost gap per passenger")
    ax.set_title("Nonmonotone priority loss")

    for label, ax in zip(["a", "b", "c"], axes):
        ax.text(
            -0.18,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.86,
        bottom=0.21,
        wspace=0.42,
    )
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    fig.savefig(FIGURE_PATH, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        os.path.join(RESULTS_DIR, "demand_capacity_mechanisms.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        os.path.join(RESULTS_DIR, "demand_capacity_mechanisms.svg"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def write_rows(filename: str, rows: list[dict]) -> None:
    with open(
        os.path.join(RESULTS_DIR, filename),
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mixed_regime, mixed_rows = run_mixed_phase()
    welfare_gap, welfare_rows = run_priority_welfare()
    plot_results(mixed_regime, welfare_gap)
    write_rows("mixed_ue_phase.csv", mixed_rows)
    write_rows("priority_welfare.csv", welfare_rows)

    no_ue_cells = sum(not row["ue_exists"] for row in mixed_rows)
    max_row = max(welfare_rows, key=lambda row: row["gap_per_passenger"])
    summary = {
        "mixed_cells": len(mixed_rows),
        "mixed_no_ue_cells": no_ue_cells,
        "verified_boundary": "UE exists if and only if q <= U",
        "priority_cells": len(welfare_rows),
        "maximum_gap_per_passenger": max_row["gap_per_passenger"],
        "maximum_gap_capacity": max_row["capacity"],
        "maximum_gap_capacity_total_demand_ratio": max_row[
            "capacity_total_demand_ratio"
        ],
        "maximum_gap_low_priority_delay": max_row["low_priority_delay"],
    }
    with open(
        os.path.join(RESULTS_DIR, "mechanism_summary.json"),
        "w",
    ) as file:
        json.dump(summary, file, indent=2)

    print(
        f"Mixed component: {len(mixed_rows)} cells, "
        f"{no_ue_cells} without UE; boundary q <= U verified."
    )
    print(
        "Priority bottleneck: maximum gap "
        f"{max_row['gap_per_passenger']:.2f} per passenger at "
        f"U/(2q)={max_row['capacity_total_demand_ratio']:.2f} and "
        f"low-priority delay={max_row['low_priority_delay']:.1f}."
    )
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
