"""Reproduce and compare the complete Nguyen et al. (2001) example.

The script uses the four OD pairs, nine paths, and capacity 20 reported in the
source article. It compares the published exact equilibrium and system optimum
with the SO, classical UE, and QUE produced under the definitions in the
manuscript.

Run:
    python experiments/run_nguyen_comparison.py
"""

from __future__ import annotations

import os

import _bootstrap  # noqa: F401
from dta import (
    AssignmentResult,
    classical_ue_assignment,
    enumerate_paths,
    system_optimum_assignment,
    quasi_ue_assignment,
    diagnostics,
    io,
)
from dta.priority_graph import build_priority_graph

from instances import nguyen_full_instance

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "nguyen"
)


def _published_result(inst, ps, metadata_key: str, method: str) -> AssignmentResult:
    labeled_flow = inst.metadata[metadata_key]
    flow = {
        p: float(labeled_flow[ps.label[p]])
        for p in ps.indices()
    }
    edge_flow = {
        edge: float(sum(flow[p] for p in paths))
        for edge, paths in ps.edge_paths.items()
    }
    return AssignmentResult(
        name=inst.name,
        method=method,
        feasible=True,
        total_cost=float(sum(ps.cost[p] * flow[p] for p in ps.indices())),
        flow=flow,
        edge_flow=edge_flow,
    )


def main() -> None:
    inst = nguyen_full_instance()
    ps = enumerate_paths(inst)

    print(f"Instance '{inst.name}': {len(ps)} paths, "
          f"{len(inst.demand)} OD groups, total demand {inst.total_demand():g}")
    for p in ps.indices():
        print(
            f"  {ps.label[p]}: {' -> '.join(map(str, ps.path[p]))}  "
            f"(cost {ps.cost[p]:g})"
        )

    pg = build_priority_graph(inst, ps)
    print(f"\nPriority graph: {pg.number_of_nodes()} nodes, "
          f"{pg.number_of_edges()} edges")

    published_eq = _published_result(
        inst, ps, "published_equilibrium_flow", "nguyen_exact_equilibrium"
    )
    published_so = _published_result(
        inst, ps, "published_system_optimum_flow", "nguyen_system_optimum"
    )
    so = system_optimum_assignment(inst, ps)
    ue = classical_ue_assignment(inst, ps)
    que = quasi_ue_assignment(inst, ps)

    print(
        f"\nNguyen exact equilibrium: cost={published_eq.total_cost:g}"
    )
    print(f"Nguyen system optimum:    cost={published_so.total_cost:g}")
    print(f"Our system optimum:       cost={so.total_cost:g}")
    print(f"Our classical UE:         feasible={ue.feasible}, cost={ue.total_cost:g}")
    print(f"Our QUE:                  feasible={que.feasible}, cost={que.total_cost:g}")
    print(f"Cycles broken (edges removed): {len(que.removed_edges)}")
    for r in que.removed_edges:
        labels = (ps.label[r.edge[0]], ps.label[r.edge[1]])
        print(f"  removed {labels} [{r.type}] -- {r.reason}")

    published_eq_diag = diagnostics.diagnose(inst, ps, published_eq)
    published_so_diag = diagnostics.diagnose(inst, ps, published_so)
    so_diag = diagnostics.diagnose(inst, ps, so)
    ue_diag = diagnostics.diagnose(inst, ps, ue)
    que_diag = diagnostics.diagnose(inst, ps, que)
    gap = diagnostics.so_que_gap(so, que)
    for label, report in (
        ("Nguyen equilibrium", published_eq_diag),
        ("Nguyen SO", published_so_diag),
        ("Our SO", so_diag),
        ("Our UE", ue_diag),
        ("Our QUE", que_diag),
    ):
        print(
            f"{label:18s}: demand_ok={report.demand_conserved}, "
            f"capacity_ok={report.capacity_feasible}, "
            f"no_improving_switch={report.no_improving_switch}"
        )
    print(f"SO-QUE gap: absolute={gap['absolute']:g}, relative={gap['relative']:.4f}")

    print("\nPer-path flow:")
    results = {
        "nguyen_equilibrium": published_eq,
        "nguyen_system_optimum": published_so,
        "our_system_optimum": so,
        "our_ue": ue,
        "our_que": que,
    }
    for p in ps.indices():
        values = ", ".join(
            f"{name}={result.flow[p]:g}"
            for name, result in results.items()
        )
        print(f"  {ps.label[p]}: {values}")

    io.ensure_dir(RESULTS_DIR)
    io.write_flow_csv(
        os.path.join(RESULTS_DIR, "flows.csv"), ps,
        results,
    )
    io.write_json(
        os.path.join(RESULTS_DIR, "nguyen.json"),
        {
            "instance": inst.name,
            "num_paths": len(ps),
            "num_od_groups": len(inst.demand),
            "demand": {str(od): volume for od, volume in inst.demand.items()},
            "vehicle_capacity": 20,
            "nguyen_exact_equilibrium": io.result_to_dict(
                published_eq, published_eq_diag
            ),
            "nguyen_system_optimum": io.result_to_dict(
                published_so, published_so_diag
            ),
            "system_optimum": io.result_to_dict(so, so_diag),
            "classical_ue": io.result_to_dict(ue, ue_diag),
            "quasi_ue": io.result_to_dict(que, que_diag),
            "so_que_gap": gap,
        },
    )
    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
