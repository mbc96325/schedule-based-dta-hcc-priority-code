"""
Result writers: persist assignment results and diagnostics to CSV / JSON instead
of only printing them.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

from .network import TransitInstance
from .paths import PathSet
from .assignment import AssignmentResult
from .diagnostics import DiagnosticReport


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _path_str(path) -> str:
    return " -> ".join(str(n) for n in path)


def write_flow_csv(
    out_path: str,
    pathset: PathSet,
    results: Dict[str, AssignmentResult],
) -> str:
    """One row per path with cost and the flow under each method in ``results``."""
    methods = list(results.keys())
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path_index", "od", "path", "cost"] + [f"flow_{m}" for m in methods])
        for p in pathset.indices():
            row = [
                p,
                str(pathset.od[p]),
                _path_str(pathset.path[p]),
                pathset.cost[p],
            ]
            row += [results[m].flow.get(p, 0.0) for m in methods]
            w.writerow(row)
    return out_path


def result_to_dict(
    result: AssignmentResult, report: Optional[DiagnosticReport] = None
) -> dict:
    d = {
        "name": result.name,
        "method": result.method,
        "feasible": result.feasible,
        "total_cost": result.total_cost,
        "served_demand": result.served_demand(),
        "left_behind": result.left_behind,
        "status": result.status,
        "flow": {str(k): v for k, v in result.flow.items()},
        "edge_flow": {str(k): v for k, v in result.edge_flow.items()},
        "order": result.order,
        "removed_edges": [
            {
                "edge": list(r.edge),
                "type": r.type,
                "reason": r.reason,
                "cycle": [list(e) for e in r.cycle],
            }
            for r in result.removed_edges
        ],
    }
    if report is not None:
        d["diagnostics"] = {
            "demand_residual": report.demand_residual,
            "demand_conserved": report.demand_conserved,
            "max_capacity_violation": report.max_capacity_violation,
            "capacity_feasible": report.capacity_feasible,
            "no_improving_switch": report.no_improving_switch,
            "improving_switches": report.improving_switches,
        }
    return d


def write_json(out_path: str, payload: dict) -> str:
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return out_path
