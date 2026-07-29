"""
Schedule-based time-space network generator (Experiment 5).

A clean, dependency-light reimplementation of the essential mechanics from the
original prototype: timetabled trips, in-vehicle
capacity, boarding (with left-behind to later trips), transfers, and demand by
departure time. It emits a :class:`dta.TransitInstance`, so the same SO / QUE /
diagnostics pipeline used for the small examples applies unchanged.

Boarding priority is *not* declared explicitly here -- it falls out of the
automatic per-in-vehicle-edge rule in :mod:`dta.priority_graph`: on a vehicle
segment ``V(line,trip,i) -> V(line,trip,i+1)`` the paths already on board (whose
predecessor is the previous in-vehicle node, an earlier time) outrank the paths
that board fresh at stop ``i`` (predecessor an origin/transfer node at the
departure time). Interleaved trips in the loop/fork family make these relations
form cycles, which the timely-last rule then breaks.

Node naming
    V|line|trip|stopidx   being on a trip at one of its stops (carries a time)
    O|stop|t              a demand origin (stop, departure time)
    D|stop                a destination (alighting sink)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import _bootstrap  # noqa: F401
from dta import TransitInstance, Edge


@dataclass
class Line:
    line_id: str
    stops: List[str]            # ordered stop ids
    headway: float
    n_trips: int
    seg_time: float = 5.0
    offset: float = 0.0         # first-trip departure offset


@dataclass
class ScheduleSpec:
    name: str
    lines: List[Line]
    demand: List[Tuple[str, str, float, float]]  # (origin, dest, depart_time, volume)
    capacity: float = 10.0
    transfer_time: float = 2.0
    max_left_behind: int = 1
    boarding_cost: float = 0.0  # fixed penalty per boarding (keeps paths distinct)
    # Stops at which inter-line transfers are allowed. ``None`` means no
    # transfers (parallel/competing lines are alternatives, not transfer points);
    # supply the hub stops for networks where transferring is intended.
    transfer_stops: Optional[Sequence[str]] = None


def _vnode(line: str, trip: int, idx: int) -> str:
    return f"V|{line}|{trip}|{idx}"


class TimeSpaceBuilder:
    """Builds a :class:`dta.TransitInstance` from a :class:`ScheduleSpec`."""

    def __init__(self, spec: ScheduleSpec):
        self.spec = spec
        self.node_time: Dict[str, float] = {}
        self.edges: List[Edge] = []
        # departures[(line, stop)] -> sorted list of (depart_time, trip, idx)
        self.departures: Dict[Tuple[str, str], List[Tuple[float, int, int]]] = {}
        self._build()

    # -- timetable -----------------------------------------------------------
    def _vtime(self, line: Line, trip: int, idx: int) -> float:
        return line.offset + trip * line.headway + idx * line.seg_time

    def _index_departures(self) -> None:
        for line in self.spec.lines:
            for trip in range(line.n_trips):
                for idx, stop in enumerate(line.stops):
                    t = self._vtime(line, trip, idx)
                    node = _vnode(line.line_id, trip, idx)
                    self.node_time[node] = t
                    self.departures.setdefault((line.line_id, stop), []).append((t, trip, idx))
        for key in self.departures:
            self.departures[key].sort()

    def _next_trips(self, line_id: str, stop: str, after: float) -> List[Tuple[float, int, int]]:
        """Up to ``max_left_behind + 1`` trips of ``line_id`` departing ``stop``
        at time ``>= after`` (the first boardable trip plus its left-behind
        alternatives)."""
        feasible = [d for d in self.departures.get((line_id, stop), []) if d[0] >= after - 1e-9]
        return feasible[: self.spec.max_left_behind + 1]

    # -- graph construction --------------------------------------------------
    def _build(self) -> None:
        self._index_departures()
        bc = self.spec.boarding_cost
        cap = self.spec.capacity

        # 1. in-vehicle + alighting edges
        for line in self.spec.lines:
            for trip in range(line.n_trips):
                for idx in range(len(line.stops)):
                    node = _vnode(line.line_id, trip, idx)
                    if idx + 1 < len(line.stops):
                        nxt = _vnode(line.line_id, trip, idx + 1)
                        self.edges.append(
                            Edge(node, nxt, cost=line.seg_time, capacity=cap, type="in-vehicle")
                        )
                    # alight to destination sink for this stop
                    self.edges.append(
                        Edge(node, f"D|{line.stops[idx]}", cost=0.0, capacity=float("inf"),
                             type="alighting")
                    )

        # 2. transfer edges: alight line A at stop s, board line B departing later
        #    (only at designated transfer stops)
        transfer_stops = set(self.spec.transfer_stops or [])
        stop_lines: Dict[str, List[Line]] = {}
        for line in self.spec.lines:
            for stop in line.stops:
                stop_lines.setdefault(stop, []).append(line)
        for line_a in self.spec.lines:
            for trip_a in range(line_a.n_trips):
                for idx_a, stop in enumerate(line_a.stops):
                    if stop not in transfer_stops:
                        continue
                    arrive = self._vtime(line_a, trip_a, idx_a)
                    src = _vnode(line_a.line_id, trip_a, idx_a)
                    for line_b in stop_lines.get(stop, []):
                        if line_b.line_id == line_a.line_id:
                            continue
                        for dep, trip_b, idx_b in self._next_trips(
                            line_b.line_id, stop, arrive + self.spec.transfer_time
                        ):
                            dst = _vnode(line_b.line_id, trip_b, idx_b)
                            self.edges.append(
                                Edge(src, dst, cost=(dep - arrive) + bc, capacity=float("inf"),
                                     type="transfer")
                            )

        # 3. demand origins + access (boarding) edges
        self.demand: Dict[Tuple[str, str], float] = {}
        for origin, dest, t, vol in self.spec.demand:
            onode = f"O|{origin}|{t:g}"
            self.node_time.setdefault(onode, t)
            self.demand[(onode, f"D|{dest}")] = self.demand.get((onode, f"D|{dest}"), 0.0) + vol
            for line in self.spec.lines:
                if origin not in line.stops:
                    continue
                for dep, trip, idx in self._next_trips(line.line_id, origin, t):
                    dst = _vnode(line.line_id, trip, idx)
                    self.edges.append(
                        Edge(onode, dst, cost=(dep - t) + bc, capacity=float("inf"),
                             type="demand")
                    )

    def instance(self) -> TransitInstance:
        return TransitInstance(self.spec.name, self.edges, self.demand, self.node_time)


def build(spec: ScheduleSpec) -> TransitInstance:
    return TimeSpaceBuilder(spec).instance()


# --------------------------------------------------------------------------- #
# Network families (Experiment 5).
# Each returns a ScheduleSpec; `build(spec)` turns it into a TransitInstance.
# --------------------------------------------------------------------------- #


def family_corridor_two_lines(capacity: float = 6.0) -> ScheduleSpec:
    """Single corridor served by two competing lines (a fast/expensive and a
    slow/cheap one). No transfers: the lines are alternatives. Exercises cost
    priority and left-behind to later trips."""
    return ScheduleSpec(
        name="corridor_two_lines",
        lines=[
            Line("A", ["s1", "s2", "s3"], headway=10, n_trips=4, seg_time=5, offset=0),
            Line("B", ["s1", "s2", "s3"], headway=10, n_trips=4, seg_time=6, offset=3),
        ],
        demand=[("s1", "s3", 0, 8), ("s1", "s3", 1, 8), ("s1", "s3", 11, 6)],
        capacity=capacity, max_left_behind=1,
    )


def family_transfer_diamond(capacity: float = 6.0) -> ScheduleSpec:
    """Transfer diamond: origin sA to destination sD via either hub h1 or h2,
    each requiring a transfer between a feeder and a delivery line."""
    return ScheduleSpec(
        name="transfer_diamond",
        lines=[
            Line("F1", ["sA", "h1"], headway=10, n_trips=4, seg_time=5, offset=0),
            Line("F2", ["sA", "h2"], headway=10, n_trips=4, seg_time=6, offset=2),
            Line("G1", ["h1", "sD"], headway=10, n_trips=4, seg_time=5, offset=8),
            Line("G2", ["h2", "sD"], headway=10, n_trips=4, seg_time=5, offset=9),
        ],
        demand=[("sA", "sD", 0, 8), ("sA", "sD", 1, 6)],
        capacity=capacity, transfer_time=2, max_left_behind=1,
        transfer_stops=["h1", "h2"],
    )


def family_fork_join(capacity: float = 6.0) -> ScheduleSpec:
    """Fork-join: two feeder lines from different origins join onto one shared
    trunk line and compete for its capacity (boarding priority by arrival)."""
    return ScheduleSpec(
        name="fork_join",
        lines=[
            Line("F1", ["sA", "sJ"], headway=12, n_trips=3, seg_time=5, offset=0),
            Line("F2", ["sB", "sJ"], headway=12, n_trips=3, seg_time=5, offset=1),
            Line("T", ["sJ", "sD"], headway=12, n_trips=3, seg_time=5, offset=8),
        ],
        demand=[("sA", "sD", 0, 6), ("sB", "sD", 0, 6)],
        capacity=capacity, transfer_time=2, max_left_behind=1,
        transfer_stops=["sJ"],
    )


def family_loop_fork(capacity: float = 5.0) -> ScheduleSpec:
    """Loop/fork network tuned to create boarding-priority cycles: two groups
    each ride two shared trunk trips, with timetable offsets arranged so the
    boarding order flips between the two shared segments."""
    return ScheduleSpec(
        name="loop_fork",
        lines=[
            # feeders set up arrival order at the two trunks
            Line("FA", ["oa", "p"], headway=20, n_trips=3, seg_time=4, offset=0),
            Line("FB", ["ob", "p"], headway=20, n_trips=3, seg_time=4, offset=2),
            # first shared trunk P: p -> q
            Line("P", ["p", "q"], headway=20, n_trips=3, seg_time=5, offset=7),
            # second shared trunk Q: q -> r, offset so the order flips
            Line("Q", ["q", "r"], headway=20, n_trips=3, seg_time=5, offset=9),
        ],
        demand=[("oa", "r", 0, 5), ("ob", "r", 0, 5)],
        capacity=capacity, transfer_time=1, max_left_behind=1,
        transfer_stops=["p", "q"],
    )


def family_high_frequency(capacity: float = 6.0) -> ScheduleSpec:
    """High-frequency service: short headway, many trips (low left-behind)."""
    return ScheduleSpec(
        name="high_frequency",
        lines=[Line("H", ["s1", "s2", "s3"], headway=4, n_trips=6, seg_time=5, offset=0)],
        demand=[("s1", "s3", 0, 8), ("s1", "s3", 2, 8), ("s1", "s3", 4, 8)],
        capacity=capacity, max_left_behind=2,
    )


def family_low_frequency(capacity: float = 6.0) -> ScheduleSpec:
    """Low-frequency service: long headway, few trips (more left-behind)."""
    return ScheduleSpec(
        name="low_frequency",
        lines=[Line("L", ["s1", "s2", "s3"], headway=15, n_trips=3, seg_time=5, offset=0)],
        demand=[("s1", "s3", 0, 8), ("s1", "s3", 2, 8), ("s1", "s3", 4, 8)],
        capacity=capacity, max_left_behind=2,
    )


def replicate_schedule(spec: ScheduleSpec, copies: int) -> ScheduleSpec:
    """Create independent copies of one timetable in a common assignment."""
    if copies < 1:
        raise ValueError("copies must be positive")

    lines: List[Line] = []
    demand: List[Tuple[str, str, float, float]] = []
    transfer_stops: List[str] = []
    original_transfer_stops = set(spec.transfer_stops or [])

    for copy in range(copies):
        prefix = f"c{copy}|"
        for line in spec.lines:
            lines.append(
                Line(
                    line_id=f"{prefix}{line.line_id}",
                    stops=[f"{prefix}{stop}" for stop in line.stops],
                    headway=line.headway,
                    n_trips=line.n_trips,
                    seg_time=line.seg_time,
                    offset=line.offset,
                )
            )
        for origin, destination, departure, volume in spec.demand:
            demand.append(
                (
                    f"{prefix}{origin}",
                    f"{prefix}{destination}",
                    departure,
                    volume,
                )
            )
        transfer_stops.extend(
            f"{prefix}{stop}" for stop in original_transfer_stops
        )

    return ScheduleSpec(
        name=f"{spec.name}_x{copies}",
        lines=lines,
        demand=demand,
        capacity=spec.capacity,
        transfer_time=spec.transfer_time,
        max_left_behind=spec.max_left_behind,
        boarding_cost=spec.boarding_cost,
        transfer_stops=transfer_stops,
    )


FAMILIES = [
    family_corridor_two_lines,
    family_transfer_diamond,
    family_fork_join,
    family_loop_fork,
    family_high_frequency,
    family_low_frequency,
]
