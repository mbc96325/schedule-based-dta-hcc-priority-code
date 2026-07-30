"""Approximate the Nguyen et al. boarding-penalty equilibrium for MTR."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from dta import (
    build_nguyen_penalty_index,
    nguyen_penalty_equilibrium,
    nguyen_penalty_equilibrium_sd,
)
from mtr_case import build_mtr_case


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "mtr_data")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results" / "mtr_nguyen",
    )
    parser.add_argument("--demand-start", type=int, default=64800)
    parser.add_argument("--demand-end", type=int, default=68400)
    parser.add_argument("--timetable-end", type=int, default=82800)
    parser.add_argument("--max-left-behind", type=int, default=5)
    parser.add_argument("--paths-per-route", type=int, default=50)
    parser.add_argument("--passengers-per-car", type=float, default=248.0)
    parser.add_argument("--transfer-time", type=int, default=180)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--theta", type=float, default=2.0)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--step-exponent", type=float, default=0.6)
    parser.add_argument(
        "--solver", choices=("sd", "msa"), default="msa"
    )
    parser.add_argument("--max-columns", type=int, default=30)
    parser.add_argument("--master-iterations", type=int, default=100)
    parser.add_argument("--initial-flow", type=Path)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--refinement-iterations", type=int, default=5000)
    parser.add_argument("--refinement-tolerance", type=float, default=0.005)
    return parser.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("Building MTR paths...", flush=True)
    build_start = time.perf_counter()
    build = build_mtr_case(
        args.data_dir,
        demand_start=args.demand_start,
        demand_end=args.demand_end,
        timetable_end=args.timetable_end,
        max_left_behind=args.max_left_behind,
        paths_per_route=args.paths_per_route,
        passengers_per_car=args.passengers_per_car,
        transfer_time=args.transfer_time,
    )
    build_seconds = time.perf_counter() - build_start
    print(
        f"Built {len(build.pathset):,} paths for "
        f"{len(build.instance.demand):,} groups in {build_seconds:.1f}s.",
        flush=True,
    )

    print("Building Nguyen sparse penalty index...", flush=True)
    index_start = time.perf_counter()
    index = build_nguyen_penalty_index(
        build.instance,
        build.pathset,
        cost_scale=60.0,
    )
    index_seconds = time.perf_counter() - index_start
    print(
        f"Indexed {index.number_of_edges:,} finite train segments and "
        f"{index.number_of_buckets:,} boarding-priority buckets in "
        f"{index_seconds:.1f}s.",
        flush=True,
    )

    print(
        f"Solving Nguyen boarding-penalty VI with {args.solver.upper()}...",
        flush=True,
    )
    solve_start = time.perf_counter()
    initial_flow = None
    if args.initial_flow is not None:
        with np.load(args.initial_flow) as saved:
            initial_flow = np.asarray(saved["flow"], dtype=float)
    if args.solver == "sd":
        result = nguyen_penalty_equilibrium_sd(
            index,
            alpha=args.alpha,
            rho=args.rho,
            theta=args.theta,
            max_columns=args.max_columns,
            master_iterations=args.master_iterations,
            tolerance=args.tolerance,
        )
    else:
        result = nguyen_penalty_equilibrium(
            index,
            alpha=args.alpha,
            rho=args.rho,
            theta=args.theta,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            report_every=args.report_every,
            step_exponent=args.step_exponent,
            initial_flow=initial_flow,
            step_offset=args.step_offset,
        )
        if (
            args.initial_flow is None
            and args.refinement_iterations > 0
            and result.relative_gap > args.refinement_tolerance
        ):
            first_result = result
            refined_result = nguyen_penalty_equilibrium(
                index,
                alpha=args.alpha,
                rho=args.rho,
                theta=args.theta,
                max_iterations=args.refinement_iterations,
                tolerance=args.refinement_tolerance,
                report_every=args.report_every,
                step_exponent=args.step_exponent,
                initial_flow=first_result.flow,
                step_offset=first_result.evaluated_iterations,
            )
            combined_history = (
                first_result.history + refined_result.history
            )
            result = (
                refined_result
                if refined_result.relative_gap < first_result.relative_gap
                else first_result
            )
            result.history = combined_history
            result.evaluated_iterations = (
                refined_result.evaluated_iterations
            )
    solve_seconds = time.perf_counter() - solve_start

    total_demand = build.instance.total_demand()
    fixed_total_cost = float(np.dot(result.flow, index.base_cost))
    adjusted_total_cost = float(np.dot(result.flow, result.adjusted_cost))
    overload = np.maximum(result.edge_flow - index.capacity, 0.0)
    load_factor = np.divide(
        result.edge_flow,
        index.capacity,
        out=np.zeros_like(result.edge_flow),
        where=index.capacity > 0,
    )
    path_left = np.asarray(
        [
            int(record["total_left_behind"])
            for record in build.path_records
        ],
        dtype=int,
    )
    left_flow = float(result.flow[path_left > 0].sum())
    positive = result.flow > 1e-8

    summary = {
        "method": (
            "Nguyen boarding-penalty equilibrium "
            f"({args.solver.upper()} approximation)"
        ),
        "parameters": {
            "alpha": args.alpha,
            "rho": args.rho,
            "theta": args.theta,
            "maximum_iterations": args.max_iterations,
            "tolerance": args.tolerance,
            "step_exponent": args.step_exponent,
            "solver": args.solver,
            "maximum_columns": args.max_columns,
            "master_iterations": args.master_iterations,
            "initial_flow": (
                str(args.initial_flow)
                if args.initial_flow is not None
                else None
            ),
            "step_offset": args.step_offset,
            "refinement_iterations": args.refinement_iterations,
            "refinement_tolerance": args.refinement_tolerance,
        },
        "build": build.quality_control,
        "number_of_paths": index.number_of_paths,
        "number_of_finite_train_segments": index.number_of_edges,
        "number_of_boarding_priority_buckets": index.number_of_buckets,
        "total_demand": total_demand,
        "served_demand": float(result.flow.sum()),
        "relative_vi_gap": result.relative_gap,
        "converged": result.converged,
        "best_iteration": result.best_iteration,
        "evaluated_iterations": result.evaluated_iterations,
        "average_fixed_generalized_cost_minutes": (
            fixed_total_cost / total_demand
        ),
        "average_penalty_adjusted_cost_minutes": (
            adjusted_total_cost / total_demand
        ),
        "capacity_violation_passenger_segments": float(overload.sum()),
        "maximum_capacity_violation": float(overload.max(initial=0.0)),
        "violated_train_segments": int((overload > 1e-7).sum()),
        "maximum_load_factor": float(load_factor.max(initial=0.0)),
        "positive_flow_paths": int(positive.sum()),
        "average_left_behind": float(
            np.dot(result.flow, path_left) / total_demand
        ),
        "passengers_left_behind": left_flow,
        "share_passengers_left_behind": left_flow / total_demand,
        "maximum_used_left_behind": int(
            path_left[positive].max() if np.any(positive) else 0
        ),
        "runtime_seconds": {
            "path_generation": build_seconds,
            "index_construction": index_seconds,
            "equilibrium_solution": solve_seconds,
            "total": build_seconds + index_seconds + solve_seconds,
        },
        "maximum_resident_set_kb": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
    }

    with open(args.results_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    pd.DataFrame(result.history).to_csv(
        args.results_dir / "iteration_history.csv", index=False
    )
    positive_positions = np.flatnonzero(positive)
    pd.DataFrame(
        {
            "path_index": index.path_ids[positive_positions],
            "path_label": [
                build.pathset.label[int(index.path_ids[position])]
                for position in positive_positions
            ],
            "flow": result.flow[positive_positions],
            "fixed_cost_minutes": index.base_cost[positive_positions],
            "boarding_penalty_minutes": result.path_penalty[
                positive_positions
            ],
            "adjusted_cost_minutes": result.adjusted_cost[
                positive_positions
            ],
            "total_left_behind": path_left[positive_positions],
        }
    ).to_csv(args.results_dir / "positive_path_flows.csv", index=False)
    np.savez_compressed(
        args.results_dir / "flow.npz",
        path_ids=index.path_ids,
        flow=result.flow,
        adjusted_cost=result.adjusted_cost,
        edge_flow=result.edge_flow,
        capacity=index.capacity,
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
