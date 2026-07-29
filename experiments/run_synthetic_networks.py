"""Section 5.5: exact QUE computation with path-priority decomposition.

Two timetable families are replicated into independent operating areas:

* a two-line corridor with an acyclic path-priority graph;
* a transfer diamond with mixed cost-boarding components.

For each size, the script compares the full-network implementation of
Algorithm 1 with an implementation that solves each path-priority component
independently. Both methods use the same exact sequential linear programs.
"""

from __future__ import annotations

import csv
import json
import os
import time

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

from dta import (
    componentwise_quasi_ue_assignment,
    diagnostics,
    enumerate_paths,
    quasi_ue_assignment,
)
from dta.priority_graph import (
    COST_PRIORITY,
    build_priority_graph,
    relation_types,
)

from timespace import (
    build,
    family_corridor_two_lines,
    family_transfer_diamond,
    replicate_schedule,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results", "synthetic")
FIGURE_PATH = os.path.join(
    os.path.dirname(ROOT),
    "manuscript",
    "Figures",
    "componentwise_que_scaling.pdf",
)

REPEATS = 5
MONOLITHIC_PATH_LIMIT = 400
FAMILIES = {
    "Acyclic corridor": {
        "builder": family_corridor_two_lines,
        "copies": [1, 2, 4, 8, 16, 32, 64],
        "color": "#3978B5",
        "marker": "o",
    },
    "Mixed transfer diamond": {
        "builder": family_transfer_diamond,
        "copies": [1, 2, 4, 8, 16, 32],
        "color": "#D95F59",
        "marker": "s",
    },
}


def median_runtime(function, repeats: int = REPEATS):
    """Return median wall-clock time and the final result."""
    function()
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2], result


def priority_class(graph: nx.DiGraph) -> str:
    """Classify cycles from the relation types inside nontrivial SCCs."""
    if nx.is_directed_acyclic_graph(graph):
        return "acyclic"
    for component in nx.strongly_connected_components(graph):
        if len(component) <= 1:
            continue
        for start, end in graph.edges:
            if (
                start in component
                and end in component
                and COST_PRIORITY in relation_types(graph, start, end)
            ):
                return "mixed"
    return "pure boarding"


def run_one(family: str, factory, copies: int) -> dict:
    """Build and solve one replicated schedule instance."""
    spec = replicate_schedule(factory(), copies)
    instance = build(spec)

    start = time.perf_counter()
    pathset = enumerate_paths(instance)
    enumeration_time = time.perf_counter() - start
    graph = build_priority_graph(instance, pathset)
    component_count = nx.number_weakly_connected_components(graph)

    component_time, component_result = median_runtime(
        lambda: componentwise_quasi_ue_assignment(instance, pathset)
    )
    component_report = diagnostics.diagnose(
        instance,
        pathset,
        component_result,
    )
    if not component_result.feasible:
        raise AssertionError(f"componentwise QUE infeasible for {spec.name}")
    if not (
        component_report.demand_conserved
        and component_report.capacity_feasible
    ):
        raise AssertionError(f"componentwise QUE diagnostics fail for {spec.name}")

    if len(pathset) <= MONOLITHIC_PATH_LIMIT:
        monolithic_time, monolithic_result = median_runtime(
            lambda: quasi_ue_assignment(instance, pathset)
        )
        if not monolithic_result.feasible:
            raise AssertionError(f"monolithic QUE infeasible for {spec.name}")
        objective_difference = abs(
            monolithic_result.total_cost - component_result.total_cost
        )
        flow_difference = max(
            abs(
                monolithic_result.flow[path]
                - component_result.flow[path]
            )
            for path in pathset.indices()
        )
        if objective_difference > 1e-7 or flow_difference > 1e-7:
            raise AssertionError(
                f"componentwise and monolithic QUE differ for {spec.name}"
            )
        speedup = monolithic_time / component_time
    else:
        monolithic_time = float("nan")
        objective_difference = float("nan")
        flow_difference = float("nan")
        speedup = float("nan")

    return {
        "family": family,
        "copies": copies,
        "nodes": instance.G.number_of_nodes(),
        "edges": instance.G.number_of_edges(),
        "paths": len(pathset),
        "priority_edges": graph.number_of_edges(),
        "components": component_count,
        "priority_class": priority_class(graph),
        "removed_relations": len(component_result.removed_edges),
        "enumeration_time_s": enumeration_time,
        "componentwise_time_s": component_time,
        "monolithic_time_s": monolithic_time,
        "speedup": speedup,
        "objective_difference": objective_difference,
        "max_flow_difference": flow_difference,
        "demand_residual": component_report.demand_residual,
        "capacity_violation": component_report.max_capacity_violation,
        "que_cost": component_result.total_cost,
    }


def plot_results(rows: list[dict]) -> None:
    """Create runtime and speedup panels for the manuscript."""
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
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.95))

    for family, settings in FAMILIES.items():
        family_rows = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: row["paths"],
        )
        paths = [row["paths"] for row in family_rows]
        component_times = [
            row["componentwise_time_s"] for row in family_rows
        ]
        axes[0].plot(
            paths,
            component_times,
            color=settings["color"],
            marker=settings["marker"],
            linewidth=1.6,
            markersize=4,
            label=f"{family}, componentwise",
        )

        monolithic_rows = [
            row
            for row in family_rows
            if row["monolithic_time_s"] == row["monolithic_time_s"]
        ]
        axes[0].plot(
            [row["paths"] for row in monolithic_rows],
            [row["monolithic_time_s"] for row in monolithic_rows],
            color=settings["color"],
            marker=settings["marker"],
            linestyle="--",
            linewidth=1.3,
            markersize=3.6,
            alpha=0.85,
            label=f"{family}, full network",
        )

        axes[1].plot(
            [row["paths"] for row in monolithic_rows],
            [row["speedup"] for row in monolithic_rows],
            color=settings["color"],
            marker=settings["marker"],
            linewidth=1.6,
            markersize=4,
            label=family,
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of paths, $|\\mathcal{P}|$")
    axes[0].set_ylabel("QUE solution time (seconds)")
    axes[0].set_title("Exact lexicographic QUE")

    axes[1].set_xscale("log")
    axes[1].set_xticks([10, 30, 100, 300])
    axes[1].set_xticklabels(["10", "30", "100", "300"])
    axes[1].axhline(1.0, color="#777777", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel("Number of paths, $|\\mathcal{P}|$")
    axes[1].set_ylabel("Full-network / componentwise time")
    axes[1].set_title("Speedup from priority components")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=FAMILIES["Acyclic corridor"]["color"],
            marker=FAMILIES["Acyclic corridor"]["marker"],
            linewidth=1.6,
            label="Acyclic corridor",
        ),
        Line2D(
            [0],
            [0],
            color=FAMILIES["Mixed transfer diamond"]["color"],
            marker=FAMILIES["Mixed transfer diamond"]["marker"],
            linewidth=1.6,
            label="Mixed transfer diamond",
        ),
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=1.6,
            label="Componentwise",
        ),
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=1.4,
            linestyle="--",
            label="Full network",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.2,
    )

    for label, axis in zip(["a", "b"], axes):
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        axis.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.86,
        bottom=0.29,
        wspace=0.34,
    )
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    fig.savefig(FIGURE_PATH, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        os.path.join(RESULTS_DIR, "componentwise_que_scaling.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(
        os.path.join(RESULTS_DIR, "componentwise_que_scaling.svg"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for family, settings in FAMILIES.items():
        for copies in settings["copies"]:
            row = run_one(
                family,
                settings["builder"],
                copies,
            )
            rows.append(row)
            monolithic = (
                f"{row['monolithic_time_s']:.3f}"
                if row["monolithic_time_s"] == row["monolithic_time_s"]
                else "not run"
            )
            speedup = (
                f"{row['speedup']:.1f}x"
                if row["speedup"] == row["speedup"]
                else "-"
            )
            print(
                f"{family:<24} copies={copies:>2} "
                f"paths={row['paths']:>4} components={row['components']:>2} "
                f"removed={row['removed_relations']:>3} "
                f"component={row['componentwise_time_s']:.3f}s "
                f"full={monolithic}s speedup={speedup}"
            )

    with open(
        os.path.join(RESULTS_DIR, "component_scaling.csv"),
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(
        os.path.join(RESULTS_DIR, "component_scaling.json"),
        "w",
    ) as file:
        json.dump(rows, file, indent=2)
    plot_results(rows)

    comparable = [
        row for row in rows if row["speedup"] == row["speedup"]
    ]
    maximum = max(comparable, key=lambda row: row["speedup"])
    print(
        "Maximum verified speedup: "
        f"{maximum['speedup']:.1f}x for {maximum['family']} "
        f"with {maximum['paths']} paths."
    )
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
