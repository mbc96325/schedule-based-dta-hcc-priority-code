"""SO, exact QUE, and small-instance classical UE assignment routines."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Hashable

import networkx as nx
import numpy as np
from scipy.optimize import linprog

from .network import TransitInstance
from .paths import PathSet, restrict_pathset
from .priority_graph import (
    BOARDING_PRIORITY,
    RemovedEdge,
    break_cycles,
    build_priority_graph,
    linear_extension,
)

Node = Hashable
TOL = 1e-8


@dataclass
class AssignmentResult:
    """Structured output shared by all assignment methods."""

    name: str
    method: str
    feasible: bool
    total_cost: float
    flow: Dict[int, float]
    edge_flow: Dict[Tuple[Node, Node], float]
    left_behind: float = 0.0
    removed_edges: List[RemovedEdge] = field(default_factory=list)
    order: List[int] = field(default_factory=list)
    status: str = "ok"
    details: Dict[str, Any] = field(default_factory=dict)

    def served_demand(self) -> float:
        return float(sum(self.flow.values()))


def _edge_flows(
    pathset: PathSet, flow: Dict[int, float]
) -> Dict[Tuple[Node, Node], float]:
    return {
        edge: float(sum(flow.get(p, 0.0) for p in paths))
        for edge, paths in pathset.edge_paths.items()
    }


def _base_lp(instance: TransitInstance, pathset: PathSet):
    indices = pathset.indices()
    position = {p: k for k, p in enumerate(indices)}
    n_paths = len(indices)

    capacity_rows = []
    capacity_rhs = []
    for edge, paths in pathset.edge_paths.items():
        capacity = instance.edge_capacity(*edge)
        if np.isinf(capacity):
            continue
        row = np.zeros(n_paths)
        for p in paths:
            row[position[p]] = 1.0
        capacity_rows.append(row)
        capacity_rhs.append(float(capacity))

    demand_rows = []
    demand_rhs = []
    for od, paths in pathset.od_paths.items():
        row = np.zeros(n_paths)
        for p in paths:
            row[position[p]] = 1.0
        demand_rows.append(row)
        demand_rhs.append(float(instance.demand[od]))

    return (
        indices,
        position,
        np.asarray(capacity_rows) if capacity_rows else None,
        np.asarray(capacity_rhs) if capacity_rhs else None,
        demand_rows,
        demand_rhs,
    )


def _result_from_vector(
    instance: TransitInstance,
    pathset: PathSet,
    method: str,
    vector: np.ndarray,
    *,
    removed_edges: Optional[List[RemovedEdge]] = None,
    order: Optional[List[int]] = None,
    status: str = "ok",
    details: Optional[Dict[str, Any]] = None,
) -> AssignmentResult:
    indices = pathset.indices()
    flow = {
        p: 0.0 if abs(float(vector[k])) <= TOL else float(vector[k])
        for k, p in enumerate(indices)
    }
    return AssignmentResult(
        name=instance.name,
        method=method,
        feasible=True,
        total_cost=float(sum(pathset.cost[p] * flow[p] for p in indices)),
        flow=flow,
        edge_flow=_edge_flows(pathset, flow),
        left_behind=0.0,
        removed_edges=list(removed_edges or []),
        order=list(order or []),
        status=status,
        details=dict(details or {}),
    )


def _infeasible_result(
    instance: TransitInstance,
    pathset: PathSet,
    method: str,
    status: str,
    *,
    removed_edges: Optional[List[RemovedEdge]] = None,
    order: Optional[List[int]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> AssignmentResult:
    return AssignmentResult(
        name=instance.name,
        method=method,
        feasible=False,
        total_cost=float("nan"),
        flow={p: 0.0 for p in pathset.indices()},
        edge_flow={},
        removed_edges=list(removed_edges or []),
        order=list(order or []),
        status=status,
        details=dict(details or {}),
    )


def system_optimum_assignment(
    instance: TransitInstance, pathset: PathSet
) -> AssignmentResult:
    """Minimize total path cost over the full-demand feasible set."""
    indices, _, A_ub, b_ub, demand_rows, demand_rhs = _base_lp(instance, pathset)
    objective = np.asarray([pathset.cost[p] for p in indices], dtype=float)
    result = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=np.asarray(demand_rows) if demand_rows else None,
        b_eq=np.asarray(demand_rhs) if demand_rhs else None,
        bounds=[(0, None)] * len(indices),
        method="highs",
    )
    if not result.success:
        return _infeasible_result(
            instance, pathset, "system_optimum", result.message
        )
    return _result_from_vector(
        instance, pathset, "system_optimum", result.x
    )


def que_order(instance: TransitInstance, pathset: PathSet):
    """Return the raw graph, acyclic graph, removals, and linear extension."""
    priority_graph = build_priority_graph(instance, pathset)
    acyclic_graph, removed = break_cycles(priority_graph)
    order = linear_extension(acyclic_graph)
    return priority_graph, acyclic_graph, removed, order


def lexicographic_assignment(
    instance: TransitInstance,
    pathset: PathSet,
    order: List[int],
    *,
    removed_edges: Optional[List[RemovedEdge]] = None,
    method: str = "quasi_ue",
) -> AssignmentResult:
    """Solve the exact sequential LP in Algorithm 1.

    At step ``h``, the LP maximizes the flow on the ``h``-th path in the
    selected linear extension while fixing all preceding path flows at their
    previously attained optima. Demand is an equality in every step, so a
    successful solve always assigns all passengers.
    """
    indices, position, A_ub, b_ub, demand_rows, demand_rhs = _base_lp(
        instance, pathset
    )
    if sorted(order) != sorted(indices):
        raise ValueError("QUE order must contain every path exactly once")

    fixed_rows = list(demand_rows)
    fixed_rhs = list(demand_rhs)
    vector = None
    optimum_sequence = []
    for path in order:
        objective = np.zeros(len(indices))
        objective[position[path]] = -1.0
        result = linprog(
            objective,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=np.asarray(fixed_rows) if fixed_rows else None,
            b_eq=np.asarray(fixed_rhs) if fixed_rhs else None,
            bounds=[(0, None)] * len(indices),
            method="highs",
        )
        if not result.success:
            return _infeasible_result(
                instance,
                pathset,
                method,
                result.message,
                removed_edges=removed_edges,
                order=order,
                details={"failed_at_path": pathset.label[path]},
            )
        vector = result.x
        optimum = float(result.x[position[path]])
        if abs(optimum) <= TOL:
            optimum = 0.0
        optimum_sequence.append((pathset.label[path], optimum))
        row = np.zeros(len(indices))
        row[position[path]] = 1.0
        fixed_rows.append(row)
        fixed_rhs.append(optimum)

    if vector is None:
        vector = np.zeros(len(indices))
    return _result_from_vector(
        instance,
        pathset,
        method,
        vector,
        removed_edges=removed_edges,
        order=order,
        details={"lexicographic_optima": optimum_sequence},
    )


def quasi_ue_assignment(
    instance: TransitInstance, pathset: PathSet
) -> AssignmentResult:
    """Construct an exact QUE from the cycle-broken path-priority graph."""
    _, _, removed, order = que_order(instance, pathset)
    return lexicographic_assignment(
        instance,
        pathset,
        order,
        removed_edges=removed,
        method="quasi_ue",
    )


def componentwise_quasi_ue_assignment(
    instance: TransitInstance, pathset: PathSet
) -> AssignmentResult:
    """Solve the exact QUE independently on each path-priority component.

    Proposition 5 implies that distinct weakly connected components of the
    path-priority graph share neither demand constraints nor finite-capacity
    conflicts. The local lexicographic solutions can therefore be combined
    without changing the full-network QUE.
    """
    priority_graph = build_priority_graph(instance, pathset)
    components = sorted(
        nx.weakly_connected_components(priority_graph),
        key=lambda nodes: min(
            (str(pathset.label[path]), path) for path in nodes
        ),
    )

    combined_flow = {path: 0.0 for path in pathset.indices()}
    combined_removed: List[RemovedEdge] = []
    combined_order: List[int] = []
    component_details = []

    for component_id, nodes in enumerate(components, start=1):
        selected = set(nodes)
        local_pathset = restrict_pathset(pathset, selected)
        local_graph = priority_graph.subgraph(selected).copy()
        local_acyclic, local_removed = break_cycles(local_graph)
        local_order = linear_extension(local_acyclic)
        local_result = lexicographic_assignment(
            instance,
            local_pathset,
            local_order,
            removed_edges=local_removed,
            method="componentwise_quasi_ue",
        )
        if not local_result.feasible:
            return _infeasible_result(
                instance,
                pathset,
                "componentwise_quasi_ue",
                local_result.status,
                removed_edges=combined_removed + local_removed,
                order=combined_order + local_order,
                details={
                    "failed_component": component_id,
                    "component_paths": [
                        pathset.label[path] for path in local_order
                    ],
                },
            )

        combined_flow.update(local_result.flow)
        combined_removed.extend(local_removed)
        combined_order.extend(local_order)
        component_details.append(
            {
                "component": component_id,
                "paths": len(selected),
                "removed_relations": len(local_removed),
            }
        )

    return AssignmentResult(
        name=instance.name,
        method="componentwise_quasi_ue",
        feasible=True,
        total_cost=float(
            sum(
                pathset.cost[path] * combined_flow[path]
                for path in pathset.indices()
            )
        ),
        flow=combined_flow,
        edge_flow=_edge_flows(pathset, combined_flow),
        removed_edges=combined_removed,
        order=combined_order,
        details={
            "component_count": len(components),
            "components": component_details,
        },
    )


def _finite_edges(instance: TransitInstance, pathset: PathSet, path: int):
    return [
        edge
        for edge in pathset.edges_of(path)
        if not np.isinf(instance.edge_capacity(*edge))
    ]


def _paths_displaceable_by(
    instance: TransitInstance,
    pathset: PathSet,
    target: int,
    edge: Tuple[Node, Node],
) -> set[int]:
    if not instance.boarding_conflicts:
        return set()
    target_label = pathset.label[target]
    return {
        pathset.index(conflict.lower_path)
        for conflict in instance.boarding_conflicts
        if conflict.higher_path == target_label
        and conflict.contested_edge == edge
    }


def switch_availability(
    instance: TransitInstance,
    pathset: PathSet,
    flow: Dict[int, float],
    source: int,
    target: int,
) -> Tuple[float, Optional[Tuple[Node, Node]]]:
    """Return the maximum directly feasible improvement and its blocking edge.

    The calculation implements the displacement mechanism used by the paper
    figures. On a target-path edge, usable capacity equals residual capacity,
    flow released by the source path, and flow on paths that have lower
    boarding priority than the target at that edge.
    """
    source_flow = max(0.0, flow.get(source, 0.0))
    availability = source_flow
    blocking_edge = None
    for edge in _finite_edges(instance, pathset, target):
        edge_flow = sum(flow.get(p, 0.0) for p in pathset.edge_paths[edge])
        available = instance.edge_capacity(*edge) - edge_flow
        displaceable = _paths_displaceable_by(
            instance, pathset, target, edge
        )
        available += sum(flow.get(p, 0.0) for p in displaceable)
        if source in pathset.edge_paths[edge] and source not in displaceable:
            available += source_flow
        if available < availability:
            availability = available
            blocking_edge = edge
    return max(0.0, min(source_flow, availability)), blocking_edge


def improving_switches(
    instance: TransitInstance,
    pathset: PathSet,
    flow: Dict[int, float],
) -> List[dict]:
    """List every positive cost-reducing switch under boarding displacement."""
    switches = []
    for source, source_flow in flow.items():
        if source_flow <= TOL:
            continue
        for target in pathset.od_paths[pathset.od[source]]:
            if pathset.cost[target] + TOL >= pathset.cost[source]:
                continue
            amount, blocking_edge = switch_availability(
                instance, pathset, flow, source, target
            )
            if amount > TOL:
                switches.append(
                    {
                        "from_path": source,
                        "from_label": pathset.label[source],
                        "to_path": target,
                        "to_label": pathset.label[target],
                        "from_cost": pathset.cost[source],
                        "to_cost": pathset.cost[target],
                        "improvable_flow": amount,
                        "blocking_edge": blocking_edge,
                    }
                )
    return switches


def classical_ue_assignment(
    instance: TransitInstance, pathset: PathSet
) -> AssignmentResult:
    """Find a classical UE for a small canonical instance, if one exists.

    For every cheaper-target/used-source pair, UE requires either zero flow on
    the source or a saturated target-path edge whose occupied flow cannot be
    displaced by the target. These alternatives are linear. The routine
    enumerates the alternatives and solves one feasibility LP per combination,
    then returns the minimum-cost UE among all feasible combinations.

    This exact disjunctive enumeration is intended for the manuscript figures;
    its case count grows exponentially and it is not used for large networks.
    """
    indices, position, A_ub, b_ub, demand_rows, demand_rhs = _base_lp(
        instance, pathset
    )
    pairs = []
    alternatives = []
    for source in indices:
        for target in pathset.od_paths[pathset.od[source]]:
            if pathset.cost[target] + TOL >= pathset.cost[source]:
                continue
            pair = (source, target)
            pairs.append(pair)
            options = [("zero", None)]
            options.extend(
                ("block", edge)
                for edge in _finite_edges(instance, pathset, target)
            )
            alternatives.append(options)

    objective = np.asarray([pathset.cost[p] for p in indices], dtype=float)
    best = None
    best_choice = None
    cases = 0
    for choices in product(*alternatives) if alternatives else [()]:
        cases += 1
        equality_rows = list(demand_rows)
        equality_rhs = list(demand_rhs)
        for (source, target), (kind, edge) in zip(pairs, choices):
            row = np.zeros(len(indices))
            if kind == "zero":
                row[position[source]] = 1.0
                rhs = 0.0
            else:
                for path in pathset.edge_paths[edge]:
                    row[position[path]] += 1.0
                displaceable = _paths_displaceable_by(
                    instance, pathset, target, edge
                )
                for path in displaceable:
                    row[position[path]] -= 1.0
                if (
                    source in pathset.edge_paths[edge]
                    and source not in displaceable
                ):
                    row[position[source]] -= 1.0
                rhs = float(instance.edge_capacity(*edge))
            equality_rows.append(row)
            equality_rhs.append(rhs)

        result = linprog(
            objective,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=np.asarray(equality_rows) if equality_rows else None,
            b_eq=np.asarray(equality_rhs) if equality_rhs else None,
            bounds=[(0, None)] * len(indices),
            method="highs",
        )
        if result.success and (
            best is None or float(result.fun) < float(best.fun) - TOL
        ):
            best = result
            best_choice = choices

    if best is None:
        return _infeasible_result(
            instance,
            pathset,
            "classical_ue",
            "no UE satisfies the boarding-displacement blocking conditions",
            details={"enumerated_cases": cases},
        )

    candidate = _result_from_vector(
        instance,
        pathset,
        "classical_ue",
        best.x,
        details={
            "enumerated_cases": cases,
            "selected_disjunction": [
                {
                    "source": pathset.label[source],
                    "target": pathset.label[target],
                    "condition": kind,
                    "edge": edge,
                }
                for (source, target), (kind, edge) in zip(pairs, best_choice)
            ],
        },
    )
    switches = improving_switches(instance, pathset, candidate.flow)
    if switches:
        return _infeasible_result(
            instance,
            pathset,
            "classical_ue",
            "internal UE verification failed",
            details={"enumerated_cases": cases, "switches": switches},
        )
    return candidate


# Compatibility names retained for old experiment scripts. Both now invoke the
# exact sequential LP instead of the former greedy or geometric-weight models.
def greedy_fill(
    instance: TransitInstance,
    pathset: PathSet,
    order: List[int],
    removed=None,
) -> AssignmentResult:
    return lexicographic_assignment(
        instance, pathset, order, removed_edges=removed, method="quasi_ue"
    )


def lp_solve(
    instance: TransitInstance,
    pathset: PathSet,
    broken,
    order: List[int],
    max_weight_log10: float = 250.0,
) -> AssignmentResult:
    del broken, max_weight_log10
    return lexicographic_assignment(
        instance, pathset, order, method="lp_priority"
    )


def lp_priority_assignment(
    instance: TransitInstance,
    pathset: PathSet,
    order: Optional[List[int]] = None,
    max_weight_log10: float = 250.0,
) -> AssignmentResult:
    del order, max_weight_log10
    _, broken, _, selected_order = que_order(instance, pathset)
    return lp_solve(instance, pathset, broken, selected_order)
