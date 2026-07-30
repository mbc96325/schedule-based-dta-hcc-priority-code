# Schedule-based DTA with hard capacity and boarding priority

This repository contains the Python implementation and experiment scripts for
the manuscript *Schedule-Based Dynamic Transit Assignment with Capacity and
Boarding Priority*. The validated scope includes the manuscript examples and
the complete four-OD Nguyen et al. (2001) benchmark. Larger synthetic
experiments remain separate from these exact regression checks.

## Repository layout

```text
dta/
  network.py          network, demand, explicit boarding conflicts
  paths.py            stable path labels and path-link incidence
  priority_graph.py   cost/boarding relations and timely-last deletion
  assignment.py       SO LP, exact sequential QUE LP, canonical UE search
  diagnostics.py      full-demand, capacity, and displacement checks
experiments/
  figure_instances.py            manuscript example specifications
  instances.py                   complete Nguyen benchmark specification
  mtr_case.py                    Hong Kong MTR data and path construction
  run_canonical_examples.py      rerun and verify all example results
  run_mtr_case.py                one-hour MTR SO and QUE experiment
  run_nguyen_comparison.py       reproduce published and computed flows
figures/                         manuscript figure-generation scripts
tests/test_dta.py                structural and numerical regressions
results/                         generated locally by the runner scripts
```

## Setup

Run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The implementation uses SciPy HiGHS and NetworkX. It does not require Gurobi
or Sage.

## Validate the manuscript examples

```bash
.venv/bin/python tests/test_dta.py
.venv/bin/python experiments/run_canonical_examples.py
.venv/bin/python experiments/run_nguyen_comparison.py
```

The tests fail if any path cost, capacity, UE existence result, SO cost, or QUE
cost differs from the validated specifications. The runners write:

- `results/canonical/summary.csv`
- `results/canonical/canonical.json`
- `results/nguyen/flows.csv`
- `results/nguyen/nguyen.json`

Additional synthetic and mechanism experiments are available as
`experiments/run_*.py`. The scripts in `figures/` write their outputs to
`results/figures/`.

## Hong Kong MTR case

The real-data runner reads `mtr_data/tb_txn.csv`,
`mtr_data/mtr_network_operation_assignment.csv`,
`mtr_data/Timetable_Test1.csv`, and `mtr_data/tb_carrier.csv`. It selects
tap-ins from 18:00 to 19:00, aggregates demand by OD and observed 15-minute
tap-out interval, extends the timetable to 23:00, and limits each generated
itinerary to five denied boardings. Vehicle capacity is
`248 * carrier_car_no`. Timetable car counts are used only for extended
horizon runs outside the temporal coverage of `tb_carrier.csv`.
For each physical route, path generation also retains boundary realizations
that attain the left-behind limit on each train leg.

```bash
python experiments/run_mtr_case.py \
  --demand-start 64800 \
  --demand-end 68400 \
  --timetable-end 82800 \
  --max-left-behind 5 \
  --paths-per-route 50
```

The large case uses sparse HiGHS matrices for SO. QUE uses a compact
precedence graph and a sequential capacity-loading shortcut. A full-demand
result certifies the same lexicographic path flows as Algorithm 1 because
every path is loaded to its remaining-demand or residual-capacity bound and
the final assignment provides a feasible completion of every prefix.

## Implementation notes

- Every manuscript path has a stable label such as `p1` or `p1_bar`. The code
  does not infer figure labels from NetworkX path-enumeration order.
- Every boarding relation records its higher-priority path, lower-priority
  path, contested finite-capacity edge, and timely-last ordering time.
- A directed path pair can retain both cost and boarding relations. One
  relation never overwrites the other.
- QUE is solved by the exact sequence of `|\mathcal P|` linear programs in
  Algorithm 1. Demand conservation is an equality in every LP, so a successful
  result has no residual or unassigned demand.
- Classical UE for these small figures is computed by enumerating the linear
  blocking alternatives induced by boarding-priority displacement. This
  solver is intended as a theory-example verifier, not a scalable network
  algorithm.
