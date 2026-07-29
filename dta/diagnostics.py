"""
Validation / diagnostic checks on an assignment result.

These implement the validation functions called for in the experiment plan:
demand conservation, capacity feasibility, path-cost ordering, the
no-improving-switch (equilibrium) check, and the SO--QUE gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Hashable

from .network import TransitInstance
from .paths import PathSet
from .assignment import AssignmentResult, improving_switches

Node = Hashable
TOL = 1e-6


@dataclass
class DiagnosticReport:
    demand_residual: float
    demand_conserved: bool
    max_capacity_violation: float
    capacity_feasible: bool
    improving_switches: List[dict]   # passenger groups that could improve
    no_improving_switch: bool

    def is_ok(self) -> bool:
        return self.demand_conserved and self.capacity_feasible


def check_demand(
    instance: TransitInstance, pathset: PathSet, result: AssignmentResult
) -> Tuple[float, bool]:
    """Largest OD imbalance under full-demand assignment."""
    served: Dict[Tuple[Node, Node], float] = {}
    for p, f in result.flow.items():
        served[pathset.od[p]] = served.get(pathset.od[p], 0.0) + f

    max_residual = 0.0
    conserved = True
    for od, demand in instance.demand.items():
        s = served.get(od, 0.0)
        max_residual = max(max_residual, abs(s - demand))
        if abs(s - demand) > TOL:
            conserved = False
    if result.left_behind > TOL:
        conserved = False
    if abs(result.served_demand() - instance.total_demand()) > TOL:
        conserved = False
    return max_residual, conserved


def check_capacity(
    instance: TransitInstance, result: AssignmentResult
) -> Tuple[float, bool]:
    """Largest capacity overflow over all edges."""
    max_violation = 0.0
    for edge, flow in result.edge_flow.items():
        cap = instance.edge_capacity(*edge)
        max_violation = max(max_violation, flow - cap)
    return max_violation, (max_violation <= TOL)


def check_no_improving_switch(
    instance: TransitInstance, pathset: PathSet, result: AssignmentResult
) -> Tuple[List[dict], bool]:
    """Check cost-reducing switches after boarding-priority displacement."""
    switches = improving_switches(instance, pathset, result.flow)
    return switches, (len(switches) == 0)


def diagnose(
    instance: TransitInstance, pathset: PathSet, result: AssignmentResult
) -> DiagnosticReport:
    demand_residual, conserved = check_demand(instance, pathset, result)
    violation, feasible = check_capacity(instance, result)
    switches, no_switch = check_no_improving_switch(instance, pathset, result)
    return DiagnosticReport(
        demand_residual=demand_residual,
        demand_conserved=conserved,
        max_capacity_violation=violation,
        capacity_feasible=feasible,
        improving_switches=switches,
        no_improving_switch=no_switch,
    )


def so_que_gap(so: AssignmentResult, que: AssignmentResult) -> dict:
    """Absolute and relative SO--QUE cost gap (QUE relative to SO)."""
    if not (so.feasible and so.total_cost == so.total_cost):  # not NaN
        return {"absolute": float("nan"), "relative": float("nan")}
    abs_gap = que.total_cost - so.total_cost
    rel = abs_gap / so.total_cost if so.total_cost else float("inf")
    return {"absolute": abs_gap, "relative": rel}
