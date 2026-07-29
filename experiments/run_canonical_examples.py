"""Rerun and validate the exact manuscript Figure 4 and 6--11 instances."""

from __future__ import annotations

import csv
import os

import networkx as nx

import _bootstrap  # noqa: F401
from dta import (
    classical_ue_assignment,
    diagnostics,
    enumerate_paths,
    io,
    quasi_ue_assignment,
    system_optimum_assignment,
)
from dta.priority_graph import (
    BOARDING_PRIORITY,
    COST_PRIORITY,
    build_priority_graph,
    relation_types,
)
from figure_instances import FIGURE_BUILDERS

TOL = 1e-7
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "canonical",
)


def _labeled_flow(pathset, result):
    return {
        pathset.label[path]: value for path, value in result.flow.items()
    }


def _labeled_result(pathset, result, report):
    payload = io.result_to_dict(result, report)
    payload["flow"] = _labeled_flow(pathset, result)
    payload["order"] = [pathset.label[p] for p in result.order]
    payload["removed_edges"] = [
        {
            "higher_path": pathset.label[item.edge[0]],
            "lower_path": pathset.label[item.edge[1]],
            "type": item.type,
            "reason": item.reason,
        }
        for item in result.removed_edges
    ]
    payload["details"] = result.details
    return payload


def _graph_type(graph):
    if nx.is_directed_acyclic_graph(graph):
        return "acyclic"
    has_mixed_loop = False
    for loop in nx.simple_cycles(graph):
        loop_relations = set()
        for u, v in zip(loop, loop[1:] + loop[:1]):
            loop_relations.update(relation_types(graph, u, v))
        if COST_PRIORITY in loop_relations:
            has_mixed_loop = True
            break
    return "mixed" if has_mixed_loop else "pure boarding"


def _matches(value, expected):
    if expected is None:
        return True
    return abs(float(value) - float(expected)) <= TOL


def _analyze(build):
    instance = build()
    pathset = enumerate_paths(instance)
    graph = build_priority_graph(instance, pathset)
    so = system_optimum_assignment(instance, pathset)
    ue = classical_ue_assignment(instance, pathset)
    que = quasi_ue_assignment(instance, pathset)
    so_report = diagnostics.diagnose(instance, pathset, so)
    ue_report = (
        diagnostics.diagnose(instance, pathset, ue) if ue.feasible else None
    )
    que_report = diagnostics.diagnose(instance, pathset, que)
    expected = instance.metadata["expected"]

    checks = {
        "path_costs": all(
            _matches(pathset.cost[pathset.index(label)], cost)
            for label, cost in expected["path_costs"].items()
        ),
        "so_cost": so.feasible and _matches(so.total_cost, expected["so_cost"]),
        "ue_existence": ue.feasible is expected["ue_exists"],
        "ue_cost": (
            not expected["ue_exists"]
            or _matches(ue.total_cost, expected["ue_cost"])
        ),
        "que_cost": que.feasible
        and _matches(que.total_cost, expected["que_cost"]),
        "so_feasibility": so_report.is_ok(),
        "ue_equilibrium": (
            not ue.feasible
            or (
                ue_report is not None
                and ue_report.is_ok()
                and ue_report.no_improving_switch
            )
        ),
        "que_feasibility": que_report.is_ok(),
        "full_demand": que.left_behind <= TOL,
    }
    return {
        "instance": instance,
        "pathset": pathset,
        "graph": graph,
        "so": so,
        "ue": ue,
        "que": que,
        "so_report": so_report,
        "ue_report": ue_report,
        "que_report": que_report,
        "checks": checks,
    }


def main() -> None:
    io.ensure_dir(RESULTS_DIR)
    rows = []
    payload = {}
    failures = []

    for build in FIGURE_BUILDERS:
        analysis = _analyze(build)
        instance = analysis["instance"]
        pathset = analysis["pathset"]
        so = analysis["so"]
        ue = analysis["ue"]
        que = analysis["que"]
        figure = instance.metadata["figure"]
        checks = analysis["checks"]
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            failures.append((figure, failed_checks))

        rows.append(
            {
                "figure": figure,
                "instance": instance.name,
                "graph_type": _graph_type(analysis["graph"]),
                "paths": len(pathset),
                "groups": len(instance.demand),
                "broken": len(que.removed_edges),
                "so_cost": so.total_cost,
                "ue_exists": ue.feasible,
                "ue_cost": ue.total_cost if ue.feasible else "none",
                "que_cost": que.total_cost,
                "full_demand": que.left_behind <= TOL,
                "all_checks_pass": not failed_checks,
            }
        )

        payload[figure] = {
            "instance": instance.name,
            "graph_type": _graph_type(analysis["graph"]),
            "demand": {str(od): value for od, value in instance.demand.items()},
            "finite_capacities": {
                str(edge): capacity
                for edge, capacity in instance.metadata[
                    "finite_capacities"
                ].items()
            },
            "boarding_relations": [
                {
                    "higher_path": higher,
                    "lower_path": lower,
                    "contested_edge": list(edge),
                }
                for higher, lower, edge in instance.metadata[
                    "boarding_relations"
                ]
            ],
            "path_catalog": [
                {
                    "index": path,
                    "label": pathset.label[path],
                    "od": list(pathset.od[path]),
                    "nodes": list(pathset.path[path]),
                    "cost": pathset.cost[path],
                }
                for path in pathset.indices()
            ],
            "system_optimum": _labeled_result(
                pathset, so, analysis["so_report"]
            ),
            "classical_ue": (
                _labeled_result(pathset, ue, analysis["ue_report"])
                if ue.feasible
                else {
                    "feasible": False,
                    "status": ue.status,
                    "details": ue.details,
                }
            ),
            "quasi_ue": _labeled_result(
                pathset, que, analysis["que_report"]
            ),
            "expected": instance.metadata["expected"],
            "checks": checks,
        }

    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    io.write_json(os.path.join(RESULTS_DIR, "canonical.json"), payload)

    header = (
        f"{'Figure':<9}{'Graph':<15}{'|P|':>5}{'|G|':>5}{'Broken':>8}"
        f"{'SO':>9}{'UE':>9}{'QUE':>9}{'Verified':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ue_cost = (
            f"{row['ue_cost']:.2f}"
            if isinstance(row["ue_cost"], float)
            else str(row["ue_cost"])
        )
        print(
            f"{row['figure']:<9}{row['graph_type']:<15}"
            f"{row['paths']:>5}{row['groups']:>5}{row['broken']:>8}"
            f"{row['so_cost']:>9.2f}{ue_cost:>9}{row['que_cost']:>9.2f}"
            f"{str(row['all_checks_pass']):>11}"
        )

    if failures:
        detail = "; ".join(
            f"Figure {figure}: {', '.join(checks)}"
            for figure, checks in failures
        )
        raise SystemExit(f"\nFigure validation failed: {detail}")
    print(f"\nAll figure checks passed. Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
