"""Run SO and QUE on the 18:00--19:00 Hong Kong MTR demand window."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from dta import diagnostics
from dta.large_scale import (
    compact_quasi_ue_assignment,
    sparse_system_optimum_assignment,
)
from mtr_case import build_mtr_case


def _weighted_path_metric(path_records, flow, field, total_demand):
    value_by_path = {
        int(record["path_index"]): float(record[field])
        for record in path_records
    }
    return (
        sum(flow.get(path, 0.0) * value_by_path[path] for path in value_by_path)
        / total_demand
    )


def _assignment_summary(build, result, elapsed):
    total_demand = build.instance.total_demand()
    demand_residual, demand_ok = diagnostics.check_demand(
        build.instance, build.pathset, result
    )
    capacity_violation, capacity_ok = diagnostics.check_capacity(
        build.instance, result
    )
    positive_path_count = sum(value > 1e-8 for value in result.flow.values())
    average_left_behind = _weighted_path_metric(
        build.path_records,
        result.flow,
        "total_left_behind",
        total_demand,
    )
    path_left = {
        int(record["path_index"]): int(record["total_left_behind"])
        for record in build.path_records
    }
    passengers_left_behind = sum(
        flow for path, flow in result.flow.items() if path_left[path] > 0
    )
    max_used_left_behind = max(
        (
            path_left[path]
            for path, flow in result.flow.items()
            if flow > 1e-8
        ),
        default=0,
    )
    load_factors = []
    for edge, edge_flow in result.edge_flow.items():
        capacity = build.instance.edge_capacity(*edge)
        if np.isinf(capacity):
            continue
        load_factors.append(edge_flow / capacity)
    saturated = sum(value >= 1.0 - 1e-7 for value in load_factors)
    return {
        "method": result.method,
        "feasible": result.feasible,
        "status": result.status,
        "runtime_seconds": elapsed,
        "total_cost_seconds": result.total_cost,
        "average_generalized_cost_minutes": (
            result.total_cost / total_demand / 60.0
            if result.feasible else None
        ),
        "served_demand": result.served_demand(),
        "demand_residual": demand_residual,
        "demand_conserved": demand_ok,
        "capacity_violation": capacity_violation,
        "capacity_feasible": capacity_ok,
        "positive_flow_paths": positive_path_count,
        "average_left_behind": average_left_behind,
        "passengers_left_behind": passengers_left_behind,
        "share_passengers_left_behind": passengers_left_behind / total_demand,
        "maximum_used_left_behind": max_used_left_behind,
        "saturated_train_segments": saturated,
        "maximum_load_factor": max(load_factors, default=0.0),
        "mean_positive_load_factor": float(
            np.mean([value for value in load_factors if value > 1e-8])
        )
        if any(value > 1e-8 for value in load_factors)
        else 0.0,
        "details": result.details,
    }


def _write_outputs(results_dir, build, so, que, payload):
    processed_dir = results_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(build.demand_records).to_csv(
        processed_dir / "demand.csv", index=False
    )
    pd.DataFrame(build.route_records).to_csv(
        processed_dir / "route_legs.csv", index=False
    )

    path_frame = pd.DataFrame(build.path_records)
    path_frame["flow_so"] = path_frame["path_index"].map(so.flow).fillna(0.0)
    path_frame["flow_que"] = path_frame["path_index"].map(que.flow).fillna(0.0)
    path_frame.to_csv(results_dir / "path_flows.csv", index=False)

    edge_rows = []
    all_edges = set(so.edge_flow) | set(que.edge_flow)
    for edge in sorted(all_edges, key=lambda value: (str(value[0]), str(value[1]))):
        capacity = build.instance.edge_capacity(*edge)
        edge_rows.append(
            {
                "edge_from": edge[0],
                "edge_to": edge[1],
                "capacity": capacity,
                "flow_so": so.edge_flow.get(edge, 0.0),
                "flow_que": que.edge_flow.get(edge, 0.0),
                "load_factor_so": so.edge_flow.get(edge, 0.0) / capacity,
                "load_factor_que": que.edge_flow.get(edge, 0.0) / capacity,
            }
        )
    pd.DataFrame(edge_rows).to_csv(
        results_dir / "train_segment_flows.csv", index=False
    )

    with open(results_dir / "summary.json", "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    pd.DataFrame(
        [
            {
                "metric": key,
                "so": payload["system_optimum"].get(key),
                "que": payload["quasi_ue"].get(key),
            }
            for key in (
                "feasible",
                "runtime_seconds",
                "average_generalized_cost_minutes",
                "served_demand",
                "demand_residual",
                "capacity_violation",
                "positive_flow_paths",
                "average_left_behind",
                "share_passengers_left_behind",
                "maximum_used_left_behind",
                "saturated_train_segments",
                "maximum_load_factor",
            )
        ]
    ).to_csv(results_dir / "summary.csv", index=False)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "mtr_data")
    parser.add_argument(
        "--results-dir", type=Path, default=root / "results" / "mtr"
    )
    parser.add_argument("--demand-start", type=int, default=64800)
    parser.add_argument("--demand-end", type=int, default=68400)
    parser.add_argument("--timetable-end", type=int, default=82800)
    parser.add_argument("--max-left-behind", type=int, default=5)
    parser.add_argument("--paths-per-route", type=int, default=50)
    parser.add_argument("--passengers-per-car", type=float, default=248.0)
    parser.add_argument("--transfer-time", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("Building the Hong Kong MTR time-space path set...", flush=True)
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
        f"{len(build.instance.demand):,} groups and "
        f"{build.instance.total_demand():,.0f} passengers in "
        f"{build_seconds:.1f} seconds.",
        flush=True,
    )

    print("Solving sparse system optimum...", flush=True)
    start = time.perf_counter()
    so = sparse_system_optimum_assignment(build.instance, build.pathset)
    so_seconds = time.perf_counter() - start
    print(
        f"SO feasible={so.feasible}, time={so_seconds:.1f}s, "
        f"cost={so.total_cost:,.1f}.",
        flush=True,
    )

    print("Constructing compact priority graph and solving QUE...", flush=True)
    start = time.perf_counter()
    que = compact_quasi_ue_assignment(build.instance, build.pathset)
    que_seconds = time.perf_counter() - start
    print(
        f"QUE feasible={que.feasible}, time={que_seconds:.1f}s, "
        f"cost={que.total_cost:,.1f}, "
        f"removed={que.details.get('removed_relations', 0):,}.",
        flush=True,
    )

    so_summary = _assignment_summary(build, so, so_seconds)
    que_summary = _assignment_summary(build, que, que_seconds)
    total_demand = build.instance.total_demand()
    observed_average = (
        sum(
            record["volume"] * record["observed_mean_journey"]
            for record in build.demand_records
        )
        / total_demand
        / 60.0
    )
    gap = (
        que.total_cost - so.total_cost
        if so.feasible and que.feasible
        else float("nan")
    )
    relative_gap = (
        gap / so.total_cost
        if so.feasible
        and que.feasible
        and abs(so.total_cost) > 1e-9
        else float("nan")
    )
    within_group_pairs = sum(
        len(paths) * (len(paths) - 1) // 2
        for paths in build.pathset.od_paths.values()
    )
    payload = {
        "instance": build.instance.name,
        "parameters": build.instance.metadata,
        "build_runtime_seconds": build_seconds,
        "quality_control": build.quality_control,
        "observed_average_journey_minutes": observed_average,
        "system_optimum": so_summary,
        "quasi_ue": que_summary,
        "so_que_gap": {
            "total_seconds": gap,
            "average_minutes_per_passenger": (
                gap / total_demand / 60.0
                if not math.isnan(gap) else None
            ),
            "relative": relative_gap if not math.isnan(relative_gap) else None,
        },
        "classical_ue": {
            "status": "not_available_for_large_case",
            "reason": (
                "The exact Nguyen/canonical UE routine enumerates blocking "
                "alternatives and is exponential in the number of "
                "cost-reducing path pairs."
            ),
            "within_group_path_pairs": within_group_pairs,
        },
        "peak_memory_gb": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1.0 if os.uname().sysname == "Darwin" else 1024.0)
            / 1024.0**3
        ),
    }
    _write_outputs(args.results_dir, build, so, que, payload)
    print(f"Results written to {args.results_dir}", flush=True)


if __name__ == "__main__":
    main()
