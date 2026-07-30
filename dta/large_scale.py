"""Sparse SO and scalable QUE routines for prescribed large path sets."""

from __future__ import annotations

from typing import Dict, List, Tuple, Hashable

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .assignment import AssignmentResult, TOL
from .network import TransitInstance
from .paths import PathSet
from .priority_graph import (
    RemovedEdge,
    break_cycles_by_scc,
    build_compact_priority_graph,
    cost_compatible_projection_order,
    linear_extension,
)

Node = Hashable


def _finite_edges(
    instance: TransitInstance, pathset: PathSet
) -> List[Tuple[Node, Node]]:
    return [
        edge
        for edge in pathset.edge_paths
        if not np.isinf(instance.edge_capacity(*edge))
    ]


def _finite_edge_flows(
    pathset: PathSet,
    flow: Dict[int, float],
    finite_edges: List[Tuple[Node, Node]],
) -> Dict[Tuple[Node, Node], float]:
    return {
        edge: float(sum(flow.get(path, 0.0) for path in pathset.edge_paths[edge]))
        for edge in finite_edges
    }


def _sparse_assignment_matrices(
    instance: TransitInstance, pathset: PathSet
):
    indices = pathset.indices()
    position = {path: column for column, path in enumerate(indices)}

    finite_edges = _finite_edges(instance, pathset)
    capacity_row = []
    capacity_col = []
    for row, edge in enumerate(finite_edges):
        for path in pathset.edge_paths[edge]:
            capacity_row.append(row)
            capacity_col.append(position[path])
    capacity_data = np.ones(len(capacity_row), dtype=float)
    A_ub = coo_matrix(
        (capacity_data, (capacity_row, capacity_col)),
        shape=(len(finite_edges), len(indices)),
    ).tocsr()
    b_ub = np.asarray(
        [instance.edge_capacity(*edge) for edge in finite_edges], dtype=float
    )

    od_items = list(pathset.od_paths.items())
    demand_row = []
    demand_col = []
    for row, (_, paths) in enumerate(od_items):
        for path in paths:
            demand_row.append(row)
            demand_col.append(position[path])
    demand_data = np.ones(len(demand_row), dtype=float)
    A_eq = coo_matrix(
        (demand_data, (demand_row, demand_col)),
        shape=(len(od_items), len(indices)),
    ).tocsr()
    b_eq = np.asarray(
        [instance.demand[od] for od, _ in od_items], dtype=float
    )
    return indices, finite_edges, A_ub, b_ub, A_eq, b_eq


def sparse_system_optimum_assignment(
    instance: TransitInstance, pathset: PathSet
) -> AssignmentResult:
    """Solve the path-based SO with sparse demand and capacity matrices."""
    (
        indices,
        finite_edges,
        A_ub,
        b_ub,
        A_eq,
        b_eq,
    ) = _sparse_assignment_matrices(instance, pathset)
    objective = np.asarray([pathset.cost[path] for path in indices], dtype=float)
    result = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        return AssignmentResult(
            name=instance.name,
            method="sparse_system_optimum",
            feasible=False,
            total_cost=float("nan"),
            flow={path: 0.0 for path in indices},
            edge_flow={},
            status=result.message,
            details={
                "solver_status": int(result.status),
                "iterations": int(getattr(result, "nit", 0)),
            },
        )

    flow = {
        path: 0.0 if abs(float(result.x[column])) <= TOL
        else float(result.x[column])
        for column, path in enumerate(indices)
    }
    return AssignmentResult(
        name=instance.name,
        method="sparse_system_optimum",
        feasible=True,
        total_cost=float(result.fun),
        flow=flow,
        edge_flow=_finite_edge_flows(pathset, flow, finite_edges),
        details={
            "solver": "scipy-highs",
            "iterations": int(getattr(result, "nit", 0)),
            "num_variables": len(indices),
            "num_capacity_constraints": len(finite_edges),
            "num_demand_constraints": len(pathset.od_paths),
        },
    )


def greedy_lexicographic_assignment(
    instance: TransitInstance,
    pathset: PathSet,
    order: List[int],
    *,
    removed_edges: List[RemovedEdge] | None = None,
    method: str = "large_scale_quasi_ue",
) -> AssignmentResult:
    """Load paths in a linear extension and certify the result when complete.

    Each path receives the largest amount allowed by its remaining group demand
    and residual capacities after all preceding path flows have been fixed. If
    the pass assigns every group in full, the resulting vector is the same
    lexicographic optimum obtained by the sequential LPs: the value fixed at
    each step reaches either the remaining demand bound or a residual-capacity
    bound, and the final flow is a feasible completion of every prefix.
    """
    indices = pathset.indices()
    if sorted(order) != sorted(indices):
        raise ValueError("large-scale QUE order must contain every path once")

    finite_edges = _finite_edges(instance, pathset)
    finite_edge_set = set(finite_edges)
    residual_capacity = {
        edge: float(instance.edge_capacity(*edge)) for edge in finite_edges
    }
    remaining_demand = {
        od: float(volume) for od, volume in instance.demand.items()
    }
    flow = {path: 0.0 for path in indices}

    for path in order:
        od = pathset.od[path]
        amount = remaining_demand[od]
        if amount <= TOL:
            continue
        path_finite_edges = [
            edge for edge in pathset.edges_of(path) if edge in finite_edge_set
        ]
        if path_finite_edges:
            amount = min(
                amount,
                min(residual_capacity[edge] for edge in path_finite_edges),
            )
        if amount <= TOL:
            continue
        flow[path] = float(amount)
        remaining_demand[od] -= amount
        for edge in path_finite_edges:
            residual_capacity[edge] -= amount
            if abs(residual_capacity[edge]) <= TOL:
                residual_capacity[edge] = 0.0

    unassigned = {
        od: volume
        for od, volume in remaining_demand.items()
        if volume > TOL
    }
    full_demand = not unassigned
    edge_flow = {
        edge: float(instance.edge_capacity(*edge) - residual_capacity[edge])
        for edge in finite_edges
    }
    return AssignmentResult(
        name=instance.name,
        method=method,
        feasible=full_demand,
        total_cost=float(
            sum(pathset.cost[path] * flow[path] for path in indices)
        ),
        flow=flow,
        edge_flow=edge_flow,
        removed_edges=list(removed_edges or []),
        order=list(order),
        status="ok" if full_demand else "incomplete full-demand loading",
        details={
            "full_demand_certificate": full_demand,
            "unassigned_demand": float(sum(unassigned.values())),
            "unassigned_groups": len(unassigned),
            "unassigned_by_group": {
                str(od): volume for od, volume in list(unassigned.items())[:100]
            },
        },
    )


def compact_quasi_ue_assignment(
    instance: TransitInstance,
    pathset: PathSet,
    *,
    projection_threshold: int = 50_000,
) -> AssignmentResult:
    """Construct QUE with compact precedence relations and certified loading."""
    priority_graph = build_compact_priority_graph(instance, pathset)
    if len(pathset) >= projection_threshold:
        order, removed_count, removed = cost_compatible_projection_order(
            priority_graph
        )
        cycle_breaking = "cost-compatible order projection"
        acyclic = True
    else:
        acyclic_graph, removed = break_cycles_by_scc(priority_graph)
        order = linear_extension(acyclic_graph)
        removed_count = len(removed)
        cycle_breaking = "iterative timely-last SCC removal"
        acyclic = nx.is_directed_acyclic_graph(acyclic_graph)
    result = greedy_lexicographic_assignment(
        instance,
        pathset,
        order,
        removed_edges=removed,
        method="large_scale_quasi_ue",
    )
    result.details.update(
        {
            "priority_nodes": priority_graph.number_of_nodes(),
            "priority_edges": priority_graph.number_of_edges(),
            "removed_relations": removed_count,
            "stored_removed_relation_samples": len(removed),
            "acyclic": acyclic,
            "compact_transitive_order": True,
            "cycle_breaking": cycle_breaking,
        }
    )
    return result
