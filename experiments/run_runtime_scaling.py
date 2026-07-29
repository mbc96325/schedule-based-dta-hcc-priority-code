"""
Experiment 3: algorithm validation and runtime scaling.

Compares the greedy quasi-UE assignment to the LP-based priority assignment
(same priority order, solved by scipy/HiGHS) over synthetic instances of growing
size, validating that the two agree and measuring how each scales with the
number of (hyper)paths.

Metrics per instance (experiment-plan list): number of paths, number of priority
edges, number of cycles removed, greedy runtime, LP runtime, objective
difference, demand-conservation residual, capacity violation, and the
no-improving-switch diagnostic.

Outputs ``results/runtime/``: ``runtime.csv``, ``runtime.json`` and the
log-log runtime-vs-paths figure ``runtime_scaling.png``.

Run:
    python experiments/run_runtime_scaling.py
"""

from __future__ import annotations

import csv
import os
import time

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dta import (
    enumerate_paths,
    que_order,
    greedy_fill,
    lp_solve,
    diagnostics,
)
from dta.priority_graph import build_priority_graph

from synthetic import make_corridor_instance, make_cycle_instance

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "runtime"
)

REPEATS = 5            # timing repeats; report the median
LP_PATH_CAP = 1500     # skip the LP above this many paths (it is the slow method)


def _median_time(fn, repeats=REPEATS):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], result


def sweep_configs():
    """(family, builder) pairs spanning a range of path counts."""
    configs = []
    # acyclic corridor: fixed gadget, scale number of disjoint copies
    for copies in (1, 2, 4, 8, 16, 32, 64):
        configs.append(
            ("corridor", lambda c=copies: make_corridor_instance(c, n_lines=3, n_segments=3))
        )
    # acyclic corridor: scale within-gadget path explosion
    for lines, segs in ((2, 3), (3, 3), (4, 3), (3, 4)):
        configs.append(
            ("corridor", lambda l=lines, s=segs: make_corridor_instance(4, n_lines=l, n_segments=s))
        )
    # cycle gadget: scale number of pure-boarding cycles to break
    for copies in (1, 4, 16, 32, 64, 128):
        configs.append(("cycle", lambda c=copies: make_cycle_instance(c)))
    return configs


def run_one(family, build):
    inst = build()
    ps = enumerate_paths(inst)
    n_paths = len(ps)

    # Shared setup (priority graph + cycle break + linear extension): common to
    # both methods, timed once.
    setup_time, (pg, broken, removed, order) = _median_time(lambda: que_order(inst, ps))

    # The distinguishing step of each method, on the same precomputed order.
    greedy_time, que = _median_time(lambda: greedy_fill(inst, ps, order, removed))

    if n_paths <= LP_PATH_CAP:
        lp_time, lp = _median_time(lambda: lp_solve(inst, ps, broken, order))
    else:
        lp_time, lp = float("nan"), None

    que_diag = diagnostics.diagnose(inst, ps, que)

    if lp is not None and lp.status == "ok":
        obj_diff = abs(que.total_cost - lp.total_cost)
        flow_max_diff = max(abs(que.flow[p] - lp.flow[p]) for p in ps.indices())
        lp_ran = True
    else:
        obj_diff = float("nan")
        flow_max_diff = float("nan")
        lp_ran = False

    # Totals = shared setup + the method's own step.
    greedy_total = setup_time + greedy_time
    lp_total = (setup_time + lp_time) if lp_ran else float("nan")

    return {
        "family": family,
        "instance": inst.name,
        "paths": n_paths,
        "priority_edges": pg.number_of_edges(),
        "cycles_removed": len(removed),
        "setup_time_s": setup_time,
        "greedy_fill_s": greedy_time,
        "lp_solve_s": lp_time,
        "greedy_total_s": greedy_total,
        "lp_total_s": lp_total,
        "lp_ran": lp_ran,
        "solve_speedup": (lp_time / greedy_time) if lp_ran and greedy_time > 0 else float("nan"),
        "total_speedup": (lp_total / greedy_total) if lp_ran and greedy_total > 0 else float("nan"),
        "obj_diff": obj_diff,
        "flow_max_diff": flow_max_diff,
        "demand_residual": que_diag.demand_residual,
        "capacity_violation": que_diag.max_capacity_violation,
        "no_improving_switch": que_diag.no_improving_switch,
        "left_behind": que.left_behind,
    }


def make_plot(rows, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for family, marker, color in (("corridor", "o", "tab:blue"), ("cycle", "s", "tab:green")):
        fam = sorted((r for r in rows if r["family"] == family), key=lambda r: r["paths"])
        if not fam:
            continue
        xs = [r["paths"] for r in fam]
        ax1.plot(xs, [r["greedy_fill_s"] for r in fam], marker=marker, color=color,
                 label=f"{family}: greedy fill")
        lp = [(r["paths"], r["lp_solve_s"]) for r in fam if r["lp_ran"]]
        if lp:
            ax1.plot([x for x, _ in lp], [t for _, t in lp], marker=marker, linestyle="--",
                     color=color, alpha=0.6, label=f"{family}: LP solve")

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("number of paths (hyperpaths)")
    ax1.set_ylabel("runtime [s] (median of %d)" % REPEATS)
    ax1.set_title("Greedy fill vs LP solve (shared setup excluded)")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend(fontsize=8)

    # speedup panel (solve-step ratio)
    sp = sorted((r for r in rows if r["lp_ran"]), key=lambda r: r["paths"])
    if sp:
        ax2.plot([r["paths"] for r in sp], [r["solve_speedup"] for r in sp], "o-",
                 color="tab:red")
        ax2.set_xscale("log")
        ax2.set_xlabel("number of paths (hyperpaths)")
        ax2.set_ylabel("LP solve / greedy fill")
        ax2.set_title("Greedy fill speedup over LP solve")
        ax2.grid(True, which="both", ls=":", alpha=0.5)
        ax2.axhline(1.0, color="gray", ls=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    print(f"{'instance':<26}{'paths':>7}{'pedges':>8}{'cyc':>5}"
          f"{'setup_s':>10}{'greedy_s':>10}{'lp_s':>10}{'spdup':>7}{'objdiff':>9}{'noSw':>6}")
    print("-" * 98)
    for family, build in sweep_configs():
        r = run_one(family, build)
        rows.append(r)
        lp_s = f"{r['lp_solve_s']:.5f}" if r["lp_ran"] else "skip"
        sp = f"{r['solve_speedup']:.1f}x" if r["lp_ran"] else "-"
        od = f"{r['obj_diff']:.1e}" if r["lp_ran"] else "-"
        print(f"{r['instance']:<26}{r['paths']:>7}{r['priority_edges']:>8}{r['cycles_removed']:>5}"
              f"{r['setup_time_s']:>10.5f}{r['greedy_fill_s']:>10.5f}{lp_s:>10}{sp:>7}{od:>9}"
              f"{str(r['no_improving_switch']):>6}")

    # write CSV + JSON
    csv_path = os.path.join(RESULTS_DIR, "runtime.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import json
    with open(os.path.join(RESULTS_DIR, "runtime.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=str)

    plot_path = os.path.join(RESULTS_DIR, "runtime_scaling.png")
    make_plot(rows, plot_path)

    # validation summary
    validated = [r for r in rows if r["lp_ran"]]
    max_obj = max((r["obj_diff"] for r in validated), default=float("nan"))
    all_ok = all(r["no_improving_switch"] and r["capacity_violation"] <= 1e-6 for r in rows)
    print("-" * 95)
    print(f"Validation: {len(validated)} instances cross-checked against the LP; "
          f"max |greedy-LP objective| = {max_obj:.2e}")
    print(f"All instances: no improving switch & capacity-feasible = {all_ok}")
    print(f"Results written to {RESULTS_DIR}/ "
          f"(runtime.csv, runtime.json, runtime_scaling.png)")


if __name__ == "__main__":
    main()
