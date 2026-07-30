"""Nguyen et al. (2001) boarding-penalty equilibrium approximation.

The model adds a soft capacity penalty to a passenger when the passenger
boards a vehicle. Continuing passengers and earlier arrivals have boarding
priority over later arrivals. The resulting path-cost mapping is asymmetric
and need not be monotone, so the successive-averages routine below is a
numerical approximation rather than a convergence-certified exact solver.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from .network import TransitInstance
from .paths import PathSet


@dataclass
class NguyenPenaltyIndex:
    """Sparse indices used to evaluate the boarding-penalty cost mapping."""

    path_ids: np.ndarray
    base_cost: np.ndarray
    group_starts: np.ndarray
    group_code: np.ndarray
    group_demand: np.ndarray
    finite_edges: List[tuple]
    capacity: np.ndarray
    incidence: sparse.csr_matrix
    event_path: np.ndarray
    event_bucket: np.ndarray
    bucket_edge: np.ndarray
    bucket_starts: np.ndarray
    bucket_counts: np.ndarray
    cost_scale: float

    @property
    def number_of_paths(self) -> int:
        return int(self.path_ids.size)

    @property
    def number_of_edges(self) -> int:
        return int(self.capacity.size)

    @property
    def number_of_buckets(self) -> int:
        return int(self.bucket_edge.size)


@dataclass
class NguyenPenaltyResult:
    """Best iterate returned by the Nguyen penalty-equilibrium solver."""

    flow: np.ndarray
    adjusted_cost: np.ndarray
    path_penalty: np.ndarray
    edge_flow: np.ndarray
    relative_gap: float
    best_iteration: int
    evaluated_iterations: int
    converged: bool
    history: List[dict]


def _group_index(
    instance: TransitInstance,
    pathset: PathSet,
    path_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return contiguous group starts, path group codes, and group demands."""
    starts = []
    demands = []
    previous_od = None
    seen = set()
    for position, path_id in enumerate(path_ids):
        od = pathset.od[int(path_id)]
        if od != previous_od:
            if od in seen:
                raise ValueError(
                    "Nguyen solver requires each demand group's paths to be "
                    "contiguous in PathSet order"
                )
            seen.add(od)
            starts.append(position)
            demands.append(float(instance.demand[od]))
            previous_od = od

    starts_array = np.asarray(starts, dtype=np.int64)
    ends = np.r_[starts_array[1:], path_ids.size]
    group_code = np.repeat(
        np.arange(starts_array.size, dtype=np.int64),
        ends - starts_array,
    )
    return (
        starts_array,
        group_code,
        np.asarray(demands, dtype=float),
    )


def build_nguyen_penalty_index(
    instance: TransitInstance,
    pathset: PathSet,
    *,
    cost_scale: float = 1.0,
) -> NguyenPenaltyIndex:
    """Build sparse path-edge and boarding-event indices.

    ``cost_scale`` converts the stored path-cost unit to the unit used by the
    penalty function. For the MTR application, costs are stored in seconds and
    ``cost_scale=60`` evaluates both fixed costs and penalties in minutes.
    """
    if cost_scale <= 0:
        raise ValueError("cost_scale must be positive")

    path_ids = np.asarray(pathset.indices(), dtype=np.int64)
    path_position = {
        int(path_id): position
        for position, path_id in enumerate(path_ids)
    }
    base_cost = np.asarray(
        [pathset.cost[int(path_id)] / cost_scale for path_id in path_ids],
        dtype=float,
    )
    group_starts, group_code, group_demand = _group_index(
        instance, pathset, path_ids
    )

    finite_edges = [
        edge
        for edge in pathset.edge_paths
        if (
            instance.edge_type(*edge) == "in-vehicle"
            and np.isfinite(instance.edge_capacity(*edge))
        )
    ]
    edge_position = {
        edge: position for position, edge in enumerate(finite_edges)
    }
    capacity = np.asarray(
        [instance.edge_capacity(*edge) for edge in finite_edges],
        dtype=float,
    )

    number_of_incidence_entries = sum(
        len(pathset.edge_paths[edge]) for edge in finite_edges
    )
    incidence_rows = np.empty(number_of_incidence_entries, dtype=np.int32)
    incidence_cols = np.empty(number_of_incidence_entries, dtype=np.int32)
    cursor = 0
    for edge_row, edge in enumerate(finite_edges):
        paths = pathset.edge_paths[edge]
        next_cursor = cursor + len(paths)
        incidence_rows[cursor:next_cursor] = edge_row
        incidence_cols[cursor:next_cursor] = [
            path_position[int(path_id)] for path_id in paths
        ]
        cursor = next_cursor
    incidence = sparse.csr_matrix(
        (
            np.ones(number_of_incidence_entries, dtype=np.float64),
            (incidence_rows, incidence_cols),
        ),
        shape=(len(finite_edges), len(path_ids)),
    )

    event_paths = array("q")
    event_edges = array("q")
    event_ready_times = array("d")
    for path_id in path_ids:
        path_id_int = int(path_id)
        nodes = pathset.path[path_id_int]
        edges = list(zip(nodes, nodes[1:]))
        previous_in_vehicle = False
        for edge_order, edge in enumerate(edges):
            current_in_vehicle = instance.edge_type(*edge) == "in-vehicle"
            finite_in_vehicle = (
                current_in_vehicle
                and np.isfinite(instance.edge_capacity(*edge))
            )
            if finite_in_vehicle and not previous_in_vehicle:
                ready_node = (
                    nodes[edge_order - 1]
                    if edge_order > 0
                    else nodes[edge_order]
                )
                event_paths.append(path_position[path_id_int])
                event_edges.append(edge_position[edge])
                event_ready_times.append(instance.time_of(ready_node))
            previous_in_vehicle = current_in_vehicle

    if event_paths:
        event_path = np.frombuffer(event_paths, dtype=np.int64).copy()
        event_edge = np.frombuffer(event_edges, dtype=np.int64).copy()
        ready_time = np.frombuffer(event_ready_times, dtype=np.float64).copy()
        ordering = np.lexsort((ready_time, event_edge))
        event_path = event_path[ordering]
        event_edge = event_edge[ordering]
        ready_time = ready_time[ordering]

        new_bucket = np.ones(event_path.size, dtype=bool)
        new_bucket[1:] = (
            (event_edge[1:] != event_edge[:-1])
            | (ready_time[1:] != ready_time[:-1])
        )
        event_bucket = np.cumsum(new_bucket, dtype=np.int64) - 1
        bucket_event_starts = np.flatnonzero(new_bucket)
        bucket_edge = event_edge[bucket_event_starts]
        unique_edge_bucket = np.ones(bucket_edge.size, dtype=bool)
        unique_edge_bucket[1:] = bucket_edge[1:] != bucket_edge[:-1]
        bucket_starts = np.flatnonzero(unique_edge_bucket)
        bucket_counts = np.diff(np.r_[bucket_starts, bucket_edge.size])
    else:
        event_path = np.empty(0, dtype=np.int64)
        event_bucket = np.empty(0, dtype=np.int64)
        bucket_edge = np.empty(0, dtype=np.int64)
        bucket_starts = np.empty(0, dtype=np.int64)
        bucket_counts = np.empty(0, dtype=np.int64)

    return NguyenPenaltyIndex(
        path_ids=path_ids,
        base_cost=base_cost,
        group_starts=group_starts,
        group_code=group_code,
        group_demand=group_demand,
        finite_edges=finite_edges,
        capacity=capacity,
        incidence=incidence,
        event_path=event_path,
        event_bucket=event_bucket,
        bucket_edge=bucket_edge,
        bucket_starts=bucket_starts,
        bucket_counts=bucket_counts,
        cost_scale=float(cost_scale),
    )


def evaluate_nguyen_cost(
    index: NguyenPenaltyIndex,
    flow: np.ndarray,
    *,
    alpha: float = 1.0,
    rho: float = 0.8,
    theta: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate adjusted path costs, path penalties, and finite-edge flows."""
    edge_flow, _, priority_load = _priority_load(index, flow)
    if index.number_of_buckets == 0:
        path_penalty = np.zeros(index.number_of_paths, dtype=float)
        return index.base_cost.copy(), path_penalty, edge_flow

    bucket_penalty = alpha * np.maximum(
        priority_load - rho * index.capacity[index.bucket_edge],
        0.0,
    ) ** theta
    path_penalty = np.bincount(
        index.event_path,
        weights=bucket_penalty[index.event_bucket],
        minlength=index.number_of_paths,
    )
    return index.base_cost + path_penalty, path_penalty, edge_flow


def _priority_load(
    index: NguyenPenaltyIndex,
    flow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return edge flow, bucket boarding flow, and bucket priority load."""
    edge_flow = np.asarray(index.incidence @ flow).ravel()
    if index.number_of_buckets == 0:
        empty = np.empty(0, dtype=float)
        return edge_flow, empty, empty
    board_flow = np.bincount(
        index.event_bucket,
        weights=flow[index.event_path],
        minlength=index.number_of_buckets,
    )
    total_boarding_by_edge = np.bincount(
        index.bucket_edge,
        weights=board_flow,
        minlength=index.number_of_edges,
    )
    continuing_flow = np.maximum(edge_flow - total_boarding_by_edge, 0.0)
    global_cumulative = np.cumsum(board_flow)
    offsets = np.zeros(index.bucket_starts.size, dtype=float)
    if index.bucket_starts.size > 1:
        offsets[1:] = global_cumulative[index.bucket_starts[1:] - 1]
    cumulative_boarding = global_cumulative - np.repeat(
        offsets, index.bucket_counts
    )
    priority_load = (
        continuing_flow[index.bucket_edge] + cumulative_boarding
    )
    return edge_flow, board_flow, priority_load


def _all_or_nothing(
    index: NguyenPenaltyIndex,
    adjusted_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    minimum_cost = np.minimum.reduceat(
        adjusted_cost, index.group_starts
    )
    is_minimum = (
        adjusted_cost
        <= minimum_cost[index.group_code] + 1e-12
    )
    positions = np.arange(index.number_of_paths, dtype=np.int64)
    selected = np.minimum.reduceat(
        np.where(is_minimum, positions, index.number_of_paths),
        index.group_starts,
    )
    flow = np.zeros(index.number_of_paths, dtype=float)
    flow[selected] = index.group_demand
    return flow, minimum_cost


def _selected_paths(
    index: NguyenPenaltyIndex,
    adjusted_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one minimum-cost path position per group and group minima."""
    minimum_cost = np.minimum.reduceat(
        adjusted_cost, index.group_starts
    )
    is_minimum = (
        adjusted_cost
        <= minimum_cost[index.group_code] + 1e-12
    )
    positions = np.arange(index.number_of_paths, dtype=np.int64)
    selected = np.minimum.reduceat(
        np.where(is_minimum, positions, index.number_of_paths),
        index.group_starts,
    )
    return selected, minimum_cost


def _flow_from_selected(
    index: NguyenPenaltyIndex,
    selected: np.ndarray,
) -> np.ndarray:
    flow = np.zeros(index.number_of_paths, dtype=float)
    flow[selected] = index.group_demand
    return flow


def _master_problem(
    base_totals: np.ndarray,
    boarding_columns: np.ndarray,
    load_columns: np.ndarray,
    initial_weights: np.ndarray,
    capacity_by_bucket: np.ndarray,
    *,
    alpha: float,
    rho: float,
    theta: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, object]:
    """Solve the restricted Smith-merit master problem."""
    number_of_columns = base_totals.size
    if number_of_columns == 1:
        return np.ones(1, dtype=float), None

    normalizer = max(float(np.mean(base_totals)), 1.0)
    cache = {}

    def objective_and_gradient(weights):
        cached_weights = cache.get("weights")
        if (
            cached_weights is not None
            and np.array_equal(cached_weights, weights)
        ):
            return cache["objective"], cache["gradient"]

        priority_load = load_columns @ weights
        excess = np.maximum(
            priority_load - rho * capacity_by_bucket,
            0.0,
        )
        penalty = alpha * excess ** theta
        column_cost = base_totals + boarding_columns.T @ penalty
        current_cost = float(np.dot(weights, column_cost))
        smith_difference = current_cost - column_cost
        positive = np.maximum(smith_difference, 0.0)
        objective = float(np.dot(positive, positive)) / (normalizer ** 2)

        derivative = np.zeros_like(excess)
        active = excess > 0
        derivative[active] = (
            alpha * theta * excess[active] ** (theta - 1.0)
        )
        cost_jacobian = boarding_columns.T @ (
            derivative[:, None] * load_columns
        )
        weighted_jacobian = weights @ cost_jacobian
        difference_jacobian = (
            column_cost[None, :]
            + weighted_jacobian[None, :]
            - cost_jacobian
        )
        gradient = (
            2.0 * positive @ difference_jacobian / (normalizer ** 2)
        )
        cache["weights"] = weights.copy()
        cache["objective"] = objective
        cache["gradient"] = gradient
        return objective, gradient

    initial_logits = np.log(np.maximum(initial_weights, 1e-12))
    initial_logits -= initial_logits.mean()

    def softmax(logits):
        shifted = logits - np.max(logits)
        weights = np.exp(shifted)
        return weights / weights.sum()

    def logit_objective_and_gradient(logits):
        weights = softmax(logits)
        objective, weight_gradient = objective_and_gradient(weights)
        logit_gradient = weights * (
            weight_gradient - np.dot(weights, weight_gradient)
        )
        return objective, logit_gradient

    result = minimize(
        fun=lambda logits: logit_objective_and_gradient(logits)[0],
        x0=initial_logits,
        jac=lambda logits: logit_objective_and_gradient(logits)[1],
        method="L-BFGS-B",
        options={
            "maxiter": maximum_iterations,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 40,
        },
    )
    weights = softmax(np.asarray(result.x, dtype=float))
    return weights, result


def nguyen_penalty_equilibrium_sd(
    index: NguyenPenaltyIndex,
    *,
    alpha: float = 1.0,
    rho: float = 0.8,
    theta: float = 2.0,
    max_columns: int = 30,
    master_iterations: int = 100,
    tolerance: float = 1e-4,
) -> NguyenPenaltyResult:
    """Approximate the VI by restricted simplicial decomposition.

    This follows the column-generation structure proposed by Nguyen et al.
    (2001). The nonconvex restricted master is solved locally with SLSQP, so
    the returned VI gap remains the relevant numerical certificate.
    """
    if max_columns < 1:
        raise ValueError("max_columns must be positive")
    if not (0 < rho <= 1):
        raise ValueError("rho must lie in (0, 1]")
    if theta <= 0 or alpha < 0:
        raise ValueError("theta must be positive and alpha nonnegative")

    first_selected, _ = _selected_paths(index, index.base_cost)
    selected_columns = [first_selected]
    selected_keys = {first_selected.tobytes()}
    base_totals = []
    boarding_columns = []
    load_columns = []
    weights = np.ones(1, dtype=float)
    history = []
    best_gap = float("inf")
    best_iteration = 0
    best_state = None
    converged = False

    for column_iteration in range(1, max_columns + 1):
        selected = selected_columns[-1]
        extreme_flow = _flow_from_selected(index, selected)
        _, board_flow, priority_load = _priority_load(
            index, extreme_flow
        )
        base_totals.append(
            float(np.dot(index.group_demand, index.base_cost[selected]))
        )
        boarding_columns.append(board_flow)
        load_columns.append(priority_load)

        if column_iteration > 1:
            weights = np.r_[0.99 * weights, 0.01]
        boarding_matrix = np.column_stack(boarding_columns)
        load_matrix = np.column_stack(load_columns)
        weights, master_result = _master_problem(
            np.asarray(base_totals),
            boarding_matrix,
            load_matrix,
            weights,
            index.capacity[index.bucket_edge],
            alpha=alpha,
            rho=rho,
            theta=theta,
            maximum_iterations=master_iterations,
        )

        flow = np.zeros(index.number_of_paths, dtype=float)
        for column_weight, column_selected in zip(
            weights, selected_columns
        ):
            if column_weight > 1e-12:
                flow[column_selected] += (
                    column_weight * index.group_demand
                )
        adjusted_cost, path_penalty, final_edge_flow = evaluate_nguyen_cost(
            index,
            flow,
            alpha=alpha,
            rho=rho,
            theta=theta,
        )
        next_selected, minimum_cost = _selected_paths(
            index, adjusted_cost
        )
        current_cost = float(np.dot(flow, adjusted_cost))
        lower_bound = float(np.dot(index.group_demand, minimum_cost))
        gap = max(current_cost - lower_bound, 0.0) / max(
            abs(current_cost), 1.0
        )
        master_success = (
            True if master_result is None else bool(master_result.success)
        )
        history.append(
            {
                "iteration": column_iteration,
                "relative_gap": gap,
                "best_relative_gap": min(best_gap, gap),
                "columns": len(selected_columns),
                "master_success": master_success,
                "master_status": (
                    "initial column"
                    if master_result is None
                    else str(master_result.message)
                ),
                "positive_column_weights": int(
                    np.count_nonzero(weights > 1e-8)
                ),
            }
        )
        if gap < best_gap:
            best_gap = gap
            best_iteration = column_iteration
            best_state = (
                flow.copy(),
                adjusted_cost.copy(),
                path_penalty.copy(),
                final_edge_flow.copy(),
            )
        if gap <= tolerance:
            converged = True
            break
        key = next_selected.tobytes()
        if key in selected_keys:
            break
        selected_columns.append(next_selected)
        selected_keys.add(key)

    if best_state is None:
        raise RuntimeError("Nguyen SD solver did not evaluate an iterate")
    best_flow, best_cost, best_penalty, best_edge_flow = best_state
    return NguyenPenaltyResult(
        flow=best_flow,
        adjusted_cost=best_cost,
        path_penalty=best_penalty,
        edge_flow=best_edge_flow,
        relative_gap=best_gap,
        best_iteration=best_iteration,
        evaluated_iterations=len(history),
        converged=converged,
        history=history,
    )


def nguyen_penalty_equilibrium(
    index: NguyenPenaltyIndex,
    *,
    alpha: float = 1.0,
    rho: float = 0.8,
    theta: float = 2.0,
    max_iterations: int = 2000,
    tolerance: float = 1e-4,
    minimum_iterations: int = 20,
    report_every: int = 10,
    step_exponent: float = 1.0,
    initial_flow: np.ndarray | None = None,
    step_offset: int = 0,
) -> NguyenPenaltyResult:
    """Approximate the asymmetric VI with deterministic successive averages."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not (0 < rho <= 1):
        raise ValueError("rho must lie in (0, 1]")
    if theta <= 0 or alpha < 0:
        raise ValueError("theta must be positive and alpha nonnegative")
    if not (0.5 < step_exponent <= 1.0):
        raise ValueError("step_exponent must lie in (0.5, 1]")
    if step_offset < 0:
        raise ValueError("step_offset must be nonnegative")

    if initial_flow is None:
        flow, _ = _all_or_nothing(index, index.base_cost)
    else:
        flow = np.asarray(initial_flow, dtype=float).copy()
        if flow.shape != (index.number_of_paths,):
            raise ValueError("initial_flow has the wrong number of paths")
        if np.any(flow < -1e-9):
            raise ValueError("initial_flow must be nonnegative")
    best_gap = float("inf")
    best_iteration = 0
    best_state = None
    history = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        global_iteration = step_offset + iteration
        adjusted_cost, path_penalty, edge_flow = evaluate_nguyen_cost(
            index,
            flow,
            alpha=alpha,
            rho=rho,
            theta=theta,
        )
        target, minimum_cost = _all_or_nothing(index, adjusted_cost)
        current_cost = float(np.dot(flow, adjusted_cost))
        lower_bound = float(np.dot(index.group_demand, minimum_cost))
        gap = max(current_cost - lower_bound, 0.0) / max(
            abs(current_cost), 1.0
        )
        if gap < best_gap:
            best_gap = gap
            best_iteration = global_iteration
            best_state = (
                flow.copy(),
                adjusted_cost.copy(),
                path_penalty.copy(),
                edge_flow.copy(),
            )
        if iteration == 1 or iteration % report_every == 0:
            history.append(
                {
                    "iteration": global_iteration,
                    "relative_gap": gap,
                    "best_relative_gap": best_gap,
                }
            )
        if global_iteration >= minimum_iterations and gap <= tolerance:
            converged = True
            break
        step = 1.0 / ((global_iteration + 1.0) ** step_exponent)
        flow += step * (target - flow)

    if best_state is None:
        raise RuntimeError("Nguyen solver did not evaluate an iterate")
    best_flow, best_cost, best_penalty, best_edge_flow = best_state
    return NguyenPenaltyResult(
        flow=best_flow,
        adjusted_cost=best_cost,
        path_penalty=best_penalty,
        edge_flow=best_edge_flow,
        relative_gap=best_gap,
        best_iteration=best_iteration,
        evaluated_iterations=step_offset + iteration,
        converged=converged,
        history=history,
    )
