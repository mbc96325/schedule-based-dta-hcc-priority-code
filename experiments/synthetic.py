"""
Scalable synthetic instance generators for the runtime-scaling experiment
(Experiment 3).

Two gadget families, each replicated into ``n_copies`` disjoint sub-networks so
the number of paths / hyperpaths grows in a controlled way:

``corridor``
    An OD group connected by a corridor of ``n_segments`` segments, each offering
    ``n_lines`` parallel capacitated lines of slightly different cost. This yields
    ``n_lines ** n_segments`` cost-distinct paths per copy and an **acyclic**
    (cost-priority) graph -- the "acyclic graph" regime of the experiment plan.

``boarding_cycle``
    The pure boarding-priority cycle gadget from the canonical examples (4 paths,
    one 2-cycle), giving the **cycle-broken** regime so the timely-last rule is
    exercised at scale.

Replication keeps each gadget's priority sub-graph small (so the within-OD
cost-priority order stays O(paths_per_copy^2) rather than O(total^2)) while the
total path count scales linearly in ``n_copies``.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple

import _bootstrap  # noqa: F401
from dta import TransitInstance, Edge

INF = float("inf")


def _corridor_gadget(
    prefix: str,
    n_lines: int,
    n_segments: int,
    capacity: float,
    demand: float,
    cost_step: float,
    node_time: Dict,
):
    """Edges for one corridor copy. Returns (edges, (origin, dest))."""
    edges: List[Edge] = []
    hubs = [f"{prefix}_H{i}" for i in range(n_segments + 1)]
    for i in range(n_segments):
        for l in range(n_lines):
            mid = f"{prefix}_m{i}_{l}"
            # boarding edge into line l (line-dependent cost spreads path costs)
            edges.append(Edge(hubs[i], mid, cost=l * cost_step, capacity=INF, type="boarding"))
            # capacitated in-vehicle edge of this line/segment
            edges.append(Edge(mid, hubs[i + 1], cost=1.0, capacity=capacity, type="in-vehicle"))
            node_time[mid] = float(i)
    return edges, (hubs[0], hubs[-1])


def make_corridor_instance(
    n_copies: int,
    n_lines: int = 3,
    n_segments: int = 3,
    capacity: float = 8.0,
    demand: float = 10.0,
    cost_step: float = 0.5,
) -> TransitInstance:
    """``n_copies`` disjoint corridors; ``n_copies * n_lines**n_segments`` paths."""
    all_edges: List[Edge] = []
    demand_map: Dict[Tuple[str, str], float] = {}
    node_time: Dict = {}
    for c in range(n_copies):
        edges, (o, d) = _corridor_gadget(
            f"c{c}", n_lines, n_segments, capacity, demand, cost_step, node_time
        )
        all_edges.extend(edges)
        demand_map[(o, d)] = demand
    name = f"corridor_x{n_copies}_L{n_lines}_S{n_segments}"
    return TransitInstance(name, all_edges, demand_map, node_time)


def _boarding_cycle_gadget(prefix: str, capacity: float):
    """One pure-boarding-cycle copy (see canonical_boarding_cycle)."""
    p = prefix
    edges = [
        Edge(f"{p}_m1", f"{p}_m2", 1, capacity, "in-vehicle"),
        Edge(f"{p}_n1", f"{p}_n2", 1, capacity, "in-vehicle"),
        Edge(f"{p}_OA", f"{p}_aIn", 0, INF, "boarding"),
        Edge(f"{p}_aIn", f"{p}_m1", 0, INF, "boarding"),
        Edge(f"{p}_m2", f"{p}_aMid", 0, INF, "boarding"),
        Edge(f"{p}_aMid", f"{p}_n1", 0, INF, "boarding"),
        Edge(f"{p}_n2", f"{p}_DA", 0, INF, "alighting"),
        Edge(f"{p}_OB", f"{p}_bIn", 0, INF, "boarding"),
        Edge(f"{p}_bIn", f"{p}_m1", 0, INF, "boarding"),
        Edge(f"{p}_m2", f"{p}_bMid", 0, INF, "boarding"),
        Edge(f"{p}_bMid", f"{p}_n1", 0, INF, "boarding"),
        Edge(f"{p}_n2", f"{p}_DB", 0, INF, "alighting"),
    ]
    node_time = {
        f"{p}_aIn": 0, f"{p}_bIn": 1, f"{p}_bMid": 2, f"{p}_aMid": 3,
    }
    demand = {(f"{p}_OA", f"{p}_DA"): capacity, (f"{p}_OB", f"{p}_DB"): capacity}
    return edges, node_time, demand


def make_cycle_instance(n_copies: int, capacity: float = 12.0) -> TransitInstance:
    """``n_copies`` disjoint boarding-cycle gadgets; ``4 * n_copies`` paths,
    ``n_copies`` cycles to break."""
    all_edges: List[Edge] = []
    demand_map: Dict = {}
    node_time: Dict = {}
    for c in range(n_copies):
        edges, nt, dem = _boarding_cycle_gadget(f"g{c}", capacity)
        all_edges.extend(edges)
        node_time.update(nt)
        demand_map.update(dem)
    name = f"boarding_cycle_x{n_copies}"
    return TransitInstance(name, all_edges, demand_map, node_time)
