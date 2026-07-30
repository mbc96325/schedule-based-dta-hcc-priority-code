"""Hong Kong MTR data preparation and prescribed time-space path generation."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from dta import Edge, PathSet, TransitInstance

INF = float("inf")


@dataclass(frozen=True)
class DemandGroup:
    group_id: int
    origin: int
    destination: int
    desired_start: int
    desired_end: int
    platform_arrival: float
    volume: float
    observed_mean_journey: float


@dataclass(frozen=True)
class RouteLeg:
    line: int
    direction: int
    board_station: int
    alight_station: int
    walk_before: float = 0.0


@dataclass(frozen=True)
class PhysicalRoute:
    route_id: str
    origin: int
    destination: int
    harbour_code: str
    evening_peak_share: float
    legs: Tuple[RouteLeg, ...]
    final_walk: float = 0.0


@dataclass(frozen=True)
class RideOption:
    run_id: str
    line: int
    direction: int
    board_station: int
    alight_station: int
    departure: float
    arrival: float
    nodes: Tuple[str, ...]
    edges: Tuple[Edge, ...]
    transfer_node: str


@dataclass(frozen=True)
class Itinerary:
    route_id: str
    rides: Tuple[RideOption, ...]
    left_behind_by_leg: Tuple[int, ...]
    walk_before_by_leg: Tuple[float, ...]
    start_time: float
    final_arrival: float
    final_walk: float
    generalized_cost: float
    schedule_penalty: float

    @property
    def departure(self) -> float:
        return self.rides[0].departure if self.rides else self.start_time

    @property
    def arrival(self) -> float:
        return self.final_arrival

    @property
    def total_left_behind(self) -> int:
        return int(sum(self.left_behind_by_leg))


@dataclass
class MTRBuildResult:
    instance: TransitInstance
    pathset: PathSet
    demand_records: List[dict]
    route_records: List[dict]
    path_records: List[dict]
    quality_control: dict


def _seconds(value: str) -> int:
    hour, minute, second = (int(part) for part in str(value).split(":"))
    return 3600 * hour + 60 * minute + second


def _adjust_run_times(values: Sequence[int]) -> List[int]:
    adjusted = []
    day_offset = 0
    previous = None
    for value in values:
        candidate = int(value) + day_offset
        if previous is not None and candidate + 12 * 3600 < previous:
            day_offset += 24 * 3600
            candidate = int(value) + day_offset
        adjusted.append(candidate)
        previous = candidate
    return adjusted


def load_demand_groups(
    transaction_path: Path,
    *,
    demand_start: int,
    demand_end: int,
    desired_interval: int = 900,
    maximum_observed_journey: int = 3 * 3600,
) -> Tuple[List[DemandGroup], dict]:
    """Filter one hour and aggregate by OD and desired arrival interval."""
    usecols = [
        "pax_origin",
        "pax_destination",
        "pax_tapin_time",
        "pax_tapout_time",
        "tapout_ti",
    ]
    frame = pd.read_csv(transaction_path, usecols=usecols)
    frame = frame[
        (frame["pax_tapin_time"] >= demand_start)
        & (frame["pax_tapin_time"] < demand_end)
    ].copy()
    raw_transactions = len(frame)

    tapout_adjusted = np.where(
        frame["pax_tapout_time"] < frame["pax_tapin_time"],
        frame["pax_tapout_time"] + 24 * 3600,
        frame["pax_tapout_time"],
    )
    frame["tapout_adjusted"] = tapout_adjusted
    frame["observed_journey"] = (
        frame["tapout_adjusted"] - frame["pax_tapin_time"]
    )
    valid = (
        (frame["observed_journey"] > 0)
        & (frame["observed_journey"] <= maximum_observed_journey)
    )
    excluded_transactions = int((~valid).sum())
    frame = frame[valid].copy()

    desired = frame["tapout_ti"].astype(int)
    desired = np.where(desired < demand_start, desired + 24 * 3600, desired)
    frame["desired_start"] = desired

    grouped = (
        frame.groupby(
            ["pax_origin", "pax_destination", "desired_start"],
            sort=True,
        )
        .agg(
            volume=("pax_tapin_time", "size"),
            platform_arrival=("pax_tapin_time", "mean"),
            observed_mean_journey=("observed_journey", "mean"),
        )
        .reset_index()
    )
    groups = []
    records = []
    for group_id, row in enumerate(grouped.itertuples(index=False), start=1):
        group = DemandGroup(
            group_id=group_id,
            origin=int(row.pax_origin),
            destination=int(row.pax_destination),
            desired_start=int(row.desired_start),
            desired_end=int(row.desired_start) + desired_interval,
            platform_arrival=float(row.platform_arrival),
            volume=float(row.volume),
            observed_mean_journey=float(row.observed_mean_journey),
        )
        groups.append(group)
        records.append(
            {
                "group_id": group.group_id,
                "origin": group.origin,
                "destination": group.destination,
                "desired_start": group.desired_start,
                "desired_end": group.desired_end,
                "platform_arrival": group.platform_arrival,
                "volume": group.volume,
                "observed_mean_journey": group.observed_mean_journey,
            }
        )
    quality = {
        "raw_transactions_in_window": raw_transactions,
        "excluded_invalid_journeys": excluded_transactions,
        "retained_transactions": int(len(frame)),
        "demand_groups": len(groups),
        "od_pairs": int(
            grouped.groupby(["pax_origin", "pax_destination"]).ngroups
        ),
    }
    return groups, {"records": records, "quality": quality}


def load_physical_routes(
    route_path: Path,
    *,
    service_keys: Iterable[Tuple[int, int, int, int]],
    share_column: str = "EP_SHARE",
    default_transfer_time: int = 180,
) -> Tuple[Dict[Tuple[int, int], List[PhysicalRoute]], List[dict]]:
    """Parse train legs and station connectors from each prescribed OD route."""
    frame = pd.read_csv(route_path)
    frame = frame[frame[share_column] > 0].copy()
    keys = ["ORI_STN_NO", "DES_STN_NO", "HARBOUR_CO"]
    service_keys = set(service_keys)

    by_od: Dict[Tuple[int, int], List[PhysicalRoute]] = {}
    records = []
    for (origin, destination, harbour_code), rows in frame.groupby(
        keys, sort=False
    ):
        legs = []
        current_leg = None
        pending_walk = 0.0
        row_list = list(rows.itertuples(index=False))
        for row_number, row in enumerate(row_list):
            leg_key = (int(row.PASS_LINE_NO), int(row.PASS_DIRECTION))
            link_start = int(row.LINK_START)
            link_end = int(row.LINK_END)
            next_delay = (
                float(row_list[row_number + 1].DELAY_TIME)
                if row_number + 1 < len(row_list)
                else 0.0
            )
            connector_time = max(
                0.0, (float(row.DELAY_TIME) - next_delay) * 60.0
            )
            movement_key = (
                leg_key[0],
                leg_key[1],
                link_start,
                link_end,
            )
            is_train_movement = (
                link_start != link_end and movement_key in service_keys
            )

            if not is_train_movement:
                if current_leg is not None:
                    legs.append(current_leg)
                    current_leg = None
                pending_walk += connector_time
                continue

            if (
                current_leg is not None
                and current_leg.line == leg_key[0]
                and current_leg.direction == leg_key[1]
                and current_leg.alight_station == link_start
                and pending_walk <= 1e-9
            ):
                current_leg = RouteLeg(
                    line=current_leg.line,
                    direction=current_leg.direction,
                    board_station=current_leg.board_station,
                    alight_station=link_end,
                    walk_before=current_leg.walk_before,
                )
                continue

            if current_leg is not None:
                legs.append(current_leg)
            walk_before = pending_walk
            if legs and walk_before <= 1e-9:
                walk_before = float(default_transfer_time)
            current_leg = RouteLeg(
                line=leg_key[0],
                direction=leg_key[1],
                board_station=link_start,
                alight_station=link_end,
                walk_before=walk_before,
            )
            pending_walk = 0.0
        if current_leg is not None:
            legs.append(current_leg)

        route_id = f"{int(origin)}-{int(destination)}-{harbour_code}"
        route = PhysicalRoute(
            route_id=route_id,
            origin=int(origin),
            destination=int(destination),
            harbour_code=str(harbour_code),
            evening_peak_share=float(rows.iloc[0][share_column]),
            legs=tuple(legs),
            final_walk=pending_walk,
        )
        by_od.setdefault((route.origin, route.destination), []).append(route)
        if not route.legs:
            records.append(
                {
                    "route_id": route.route_id,
                    "origin": route.origin,
                    "destination": route.destination,
                    "harbour_code": route.harbour_code,
                    "evening_peak_share": route.evening_peak_share,
                    "leg_order": 0,
                    "line": None,
                    "direction": None,
                    "board_station": route.origin,
                    "alight_station": route.destination,
                    "walk_before": route.final_walk,
                    "final_walk": route.final_walk,
                }
            )
        else:
            for leg_order, leg in enumerate(route.legs, start=1):
                records.append(
                    {
                        "route_id": route.route_id,
                        "origin": route.origin,
                        "destination": route.destination,
                        "harbour_code": route.harbour_code,
                        "evening_peak_share": route.evening_peak_share,
                        "leg_order": leg_order,
                        "line": leg.line,
                        "direction": leg.direction,
                        "board_station": leg.board_station,
                        "alight_station": leg.alight_station,
                        "walk_before": leg.walk_before,
                        "final_walk": route.final_walk,
                    }
                )
    for routes in by_od.values():
        routes.sort(key=lambda route: (-route.evening_peak_share, route.route_id))
    return by_od, records


def load_timetable_options(
    timetable_path: Path,
    *,
    carrier_path: Path,
    demand_start: int,
    timetable_end: int,
    passengers_per_car: float,
    transfer_time: int,
) -> Tuple[
    Dict[Tuple[int, int, int, int], List[RideOption]],
    Dict[str, float],
    dict,
]:
    """Build indexed train-run options for every ordered station pair."""
    frame = pd.read_csv(timetable_path)
    frame = frame[frame["Revenue_Y_N"] == "Y"].copy()
    carrier = pd.read_csv(
        carrier_path,
        usecols=[
            "carrier_line",
            "carrier_direction",
            "carrier_trip",
            "carrier_car_no",
        ],
    )
    carrier_keys = [
        "carrier_line",
        "carrier_direction",
        "carrier_trip",
    ]
    if carrier.duplicated(carrier_keys).any():
        raise ValueError("tb_carrier contains duplicate line-direction-trip rows")
    carrier_lookup = {
        (
            int(row.carrier_line),
            int(row.carrier_direction),
            str(row.carrier_trip),
        ): float(row.carrier_car_no)
        for row in carrier.itertuples(index=False)
    }
    ride_index: Dict[Tuple[int, int, int, int], List[RideOption]] = {}
    node_time: Dict[str, float] = {}
    retained_runs = 0
    retained_segments = 0
    carrier_matched_runs = 0
    timetable_fallback_runs = 0
    carrier_timetable_mismatches = 0
    capacity_values = []

    for (line, trip_no), rows in frame.groupby(
        ["LINE_CODE", "Trip_No"], sort=False
    ):
        rows = rows.reset_index(drop=True)
        raw_sequence = []
        for row in rows.itertuples(index=False):
            raw_sequence.extend(
                [_seconds(row.Dep_From), _seconds(row.Arr_To)]
            )
        adjusted = _adjust_run_times(raw_sequence)
        departures = adjusted[0::2]
        arrivals = adjusted[1::2]
        if max(arrivals) < demand_start or min(departures) > timetable_end:
            continue
        if len(set(rows["Direction_ID"].astype(int))) != 1:
            raise ValueError(f"trip {line}-{trip_no} changes direction")

        stations = [int(rows.iloc[0]["From_ID"])]
        stations.extend(int(value) for value in rows["To_ID"])
        if any(
            int(rows.iloc[i]["To_ID"]) != int(rows.iloc[i + 1]["From_ID"])
            for i in range(len(rows) - 1)
        ):
            raise ValueError(f"trip {line}-{trip_no} has a broken station chain")

        run_id = f"{int(line)}:{trip_no}"
        direction = int(rows.iloc[0]["Direction_ID"])
        timetable_car_num = float(rows.iloc[0]["Car_Num"])
        carrier_car_num = carrier_lookup.get(
            (int(line), direction, str(trip_no))
        )
        if carrier_car_num is None:
            car_num = timetable_car_num
            timetable_fallback_runs += 1
        else:
            car_num = carrier_car_num
            carrier_matched_runs += 1
            if abs(carrier_car_num - timetable_car_num) > 1e-9:
                carrier_timetable_mismatches += 1
        capacity = car_num * passengers_per_car
        capacity_values.append(capacity)
        retained_runs += 1
        retained_segments += len(rows)

        departure_nodes = []
        arrival_nodes = [None]
        segment_edges = []
        dwell_edges = [None]
        for index, (departure, arrival) in enumerate(
            zip(departures, arrivals)
        ):
            departure_node = f"V|{run_id}|{index}"
            arrival_node = f"A|{run_id}|{index + 1}"
            departure_nodes.append(departure_node)
            arrival_nodes.append(arrival_node)
            node_time[departure_node] = float(departure)
            node_time[arrival_node] = float(arrival)
            segment_edges.append(
                Edge(
                    departure_node,
                    arrival_node,
                    cost=float(arrival - departure),
                    capacity=capacity,
                    type="in-vehicle",
                )
            )
            if index + 1 < len(rows):
                next_departure = departures[index + 1]
                next_departure_node = f"V|{run_id}|{index + 1}"
                dwell_edges.append(
                    Edge(
                        arrival_node,
                        next_departure_node,
                        cost=float(next_departure - arrival),
                        capacity=INF,
                        type="in-vehicle",
                    )
                )

        for board_index in range(len(stations) - 1):
            for alight_index in range(board_index + 1, len(stations)):
                nodes = [departure_nodes[board_index]]
                edges = []
                for segment_index in range(board_index, alight_index):
                    edges.append(segment_edges[segment_index])
                    nodes.append(arrival_nodes[segment_index + 1])
                    if segment_index + 1 < alight_index:
                        edges.append(dwell_edges[segment_index + 1])
                        nodes.append(departure_nodes[segment_index + 1])
                transfer_node = (
                    f"X|{run_id}|{alight_index}|{transfer_time}"
                )
                node_time[transfer_node] = (
                    float(arrivals[alight_index - 1]) + transfer_time
                )
                option = RideOption(
                    run_id=run_id,
                    line=int(line),
                    direction=direction,
                    board_station=stations[board_index],
                    alight_station=stations[alight_index],
                    departure=float(departures[board_index]),
                    arrival=float(arrivals[alight_index - 1]),
                    nodes=tuple(nodes),
                    edges=tuple(edges),
                    transfer_node=transfer_node,
                )
                key = (
                    option.line,
                    option.direction,
                    option.board_station,
                    option.alight_station,
                )
                ride_index.setdefault(key, []).append(option)

    for options in ride_index.values():
        options.sort(key=lambda option: (option.departure, option.run_id))
    stats = {
        "retained_runs": retained_runs,
        "retained_segments": retained_segments,
        "minimum_train_capacity": min(capacity_values),
        "maximum_train_capacity": max(capacity_values),
        "passengers_per_car": passengers_per_car,
        "carrier_matched_runs": carrier_matched_runs,
        "timetable_car_count_fallback_runs": timetable_fallback_runs,
        "carrier_timetable_car_count_mismatches": (
            carrier_timetable_mismatches
        ),
        "timetable_end": timetable_end,
    }
    return ride_index, node_time, stats


def _enumerate_route_itineraries(
    group: DemandGroup,
    route: PhysicalRoute,
    ride_index: Dict[Tuple[int, int, int, int], List[RideOption]],
    *,
    max_left_behind: int,
    paths_per_route: int,
    early_penalty: float,
    late_penalty: float,
) -> List[Itinerary]:
    if not route.legs:
        arrival = group.platform_arrival + route.final_walk
        schedule_penalty = (
            early_penalty * max(0.0, group.desired_start - arrival)
            + late_penalty * max(0.0, arrival - group.desired_end)
        )
        return [
            Itinerary(
                route_id=route.route_id,
                rides=tuple(),
                left_behind_by_leg=tuple(),
                walk_before_by_leg=tuple(),
                start_time=group.platform_arrival,
                final_arrival=arrival,
                final_walk=route.final_walk,
                generalized_cost=route.final_walk + schedule_penalty,
                schedule_penalty=schedule_penalty,
            )
        ]

    states = [(group.platform_arrival, tuple(), tuple(), 0)]
    for leg_order, leg in enumerate(route.legs):
        key = (
            leg.line,
            leg.direction,
            leg.board_station,
            leg.alight_station,
        )
        options = ride_index.get(key, [])
        departures = [option.departure for option in options]
        next_states = []
        for ready, rides, left_counts, total_left in states:
            platform_ready = ready + leg.walk_before
            first = bisect_left(departures, platform_ready - 1e-9)
            remaining = max_left_behind - total_left
            for left_count in range(remaining + 1):
                option_index = first + left_count
                if option_index >= len(options):
                    break
                option = options[option_index]
                next_ready = option.arrival
                next_states.append(
                    (
                        next_ready,
                        rides + (option,),
                        left_counts + (left_count,),
                        total_left + left_count,
                    )
                )
        states = next_states
        if not states:
            return []

    candidates = []
    for _, rides, left_counts, _ in states:
        arrival = rides[-1].arrival + route.final_walk
        schedule_penalty = (
            early_penalty * max(0.0, group.desired_start - arrival)
            + late_penalty * max(0.0, arrival - group.desired_end)
        )
        generalized_cost = (
            arrival - group.platform_arrival + schedule_penalty
        )
        candidates.append(
            Itinerary(
                route_id=route.route_id,
                rides=rides,
                left_behind_by_leg=left_counts,
                walk_before_by_leg=tuple(
                    leg.walk_before for leg in route.legs
                ),
                start_time=group.platform_arrival,
                final_arrival=arrival,
                final_walk=route.final_walk,
                generalized_cost=float(generalized_cost),
                schedule_penalty=float(schedule_penalty),
            )
        )
    candidates.sort(
        key=lambda itinerary: (
            itinerary.generalized_cost,
            itinerary.arrival,
            tuple(ride.run_id for ride in itinerary.rides),
            itinerary.left_behind_by_leg,
        )
    )
    selected = list(candidates[:paths_per_route])
    selected_keys = {
        (
            tuple(ride.run_id for ride in itinerary.rides),
            itinerary.left_behind_by_leg,
        )
        for itinerary in selected
    }
    boundary_candidates = [
        max(
            candidates,
            key=lambda itinerary: (
                itinerary.arrival,
                itinerary.generalized_cost,
            ),
        )
    ]
    for leg_order in range(len(route.legs)):
        at_limit = [
            itinerary
            for itinerary in candidates
            if itinerary.left_behind_by_leg[leg_order] == max_left_behind
        ]
        if at_limit:
            boundary_candidates.append(
                min(
                    at_limit,
                    key=lambda itinerary: (
                        itinerary.generalized_cost,
                        itinerary.arrival,
                        tuple(
                            ride.run_id for ride in itinerary.rides
                        ),
                    ),
                )
            )
    for itinerary in boundary_candidates:
        key = (
            tuple(ride.run_id for ride in itinerary.rides),
            itinerary.left_behind_by_leg,
        )
        if key not in selected_keys:
            selected.append(itinerary)
            selected_keys.add(key)
    selected.sort(
        key=lambda itinerary: (
            itinerary.generalized_cost,
            itinerary.arrival,
            tuple(ride.run_id for ride in itinerary.rides),
            itinerary.left_behind_by_leg,
        )
    )
    return selected


def _insert_edge(edge_map: Dict[Tuple[str, str], Edge], edge: Edge) -> None:
    key = edge.as_tuple()
    existing = edge_map.get(key)
    if existing is None:
        edge_map[key] = edge
        return
    if (
        abs(existing.cost - edge.cost) > 1e-6
        or abs(existing.capacity - edge.capacity) > 1e-6
        or existing.type != edge.type
    ):
        raise ValueError(f"conflicting definitions for edge {key}")


def build_mtr_case(
    data_dir: Path,
    *,
    demand_start: int = 64800,
    demand_end: int = 64800 + 3600,
    timetable_end: int = 23 * 3600,
    max_left_behind: int = 5,
    paths_per_route: int = 50,
    passengers_per_car: float = 248.0,
    transfer_time: int = 180,
    early_penalty: float = 1.0,
    late_penalty: float = 1.0,
) -> MTRBuildResult:
    """Create the one-hour MTR assignment with an extended timetable."""
    groups, demand_payload = load_demand_groups(
        data_dir / "tb_txn.csv",
        demand_start=demand_start,
        demand_end=demand_end,
    )
    ride_index, node_time, timetable_stats = load_timetable_options(
        data_dir / "Timetable_Test1.csv",
        carrier_path=data_dir / "tb_carrier.csv",
        demand_start=demand_start,
        timetable_end=timetable_end,
        passengers_per_car=passengers_per_car,
        transfer_time=transfer_time,
    )
    routes_by_od, route_records = load_physical_routes(
        data_dir / "mtr_network_operation_assignment.csv",
        service_keys=ride_index.keys(),
        default_transfer_time=transfer_time,
    )

    edge_map: Dict[Tuple[str, str], Edge] = {}
    demand = {}
    pathset = PathSet()
    path_records = []
    missing_groups = []
    next_path = 1

    for group in groups:
        source = f"O|g{group.group_id}"
        sink = f"D|g{group.group_id}"
        node_time[source] = group.platform_arrival
        node_time[sink] = group.desired_end
        od = (source, sink)
        demand[od] = group.volume

        candidates = []
        for route in routes_by_od.get(
            (group.origin, group.destination), []
        ):
            candidates.extend(
                _enumerate_route_itineraries(
                    group,
                    route,
                    ride_index,
                    max_left_behind=max_left_behind,
                    paths_per_route=paths_per_route,
                    early_penalty=early_penalty,
                    late_penalty=late_penalty,
                )
            )
        deduplicated = {}
        for itinerary in candidates:
            signature = tuple(
                (
                    ride.run_id,
                    ride.board_station,
                    ride.alight_station,
                )
                for ride in itinerary.rides
            )
            if not signature:
                signature = (("walk", itinerary.route_id, 0),)
            current = deduplicated.get(signature)
            if current is None or (
                itinerary.generalized_cost,
                itinerary.route_id,
            ) < (
                current.generalized_cost,
                current.route_id,
            ):
                deduplicated[signature] = itinerary
        candidates = sorted(
            deduplicated.values(),
            key=lambda itinerary: (
                itinerary.generalized_cost,
                itinerary.route_id,
                tuple(ride.run_id for ride in itinerary.rides),
            ),
        )
        if not candidates:
            missing_groups.append(group)
            continue

        for candidate_number, itinerary in enumerate(candidates, start=1):
            nodes = [source]
            if not itinerary.rides:
                _insert_edge(
                    edge_map,
                    Edge(
                        source,
                        sink,
                        cost=itinerary.generalized_cost,
                        capacity=INF,
                        type="exit",
                    ),
                )
                nodes.append(sink)
            else:
                first_ride = itinerary.rides[0]
                first_board = first_ride.nodes[0]
                first_walk = itinerary.walk_before_by_leg[0]
                if first_walk > 1e-9:
                    access_node = (
                        f"B|g{group.group_id}|{itinerary.route_id}|"
                        f"{first_walk:g}"
                    )
                    node_time[access_node] = (
                        group.platform_arrival + first_walk
                    )
                    _insert_edge(
                        edge_map,
                        Edge(
                            source,
                            access_node,
                            cost=first_walk,
                            capacity=INF,
                            type="demand",
                        ),
                    )
                    _insert_edge(
                        edge_map,
                        Edge(
                            access_node,
                            first_board,
                            cost=first_ride.departure
                            - (group.platform_arrival + first_walk),
                            capacity=INF,
                            type="boarding",
                        ),
                    )
                    nodes.extend([access_node, first_board])
                else:
                    _insert_edge(
                        edge_map,
                        Edge(
                            source,
                            first_board,
                            cost=first_ride.departure
                            - group.platform_arrival,
                            capacity=INF,
                            type="demand",
                        ),
                    )
                    nodes.append(first_board)

                for leg_order, ride in enumerate(itinerary.rides):
                    if leg_order > 0:
                        previous = itinerary.rides[leg_order - 1]
                        walk_time = itinerary.walk_before_by_leg[leg_order]
                        transfer_node = (
                            f"X|{previous.run_id}|"
                            f"{previous.alight_station}|{walk_time:g}"
                        )
                        node_time[transfer_node] = (
                            previous.arrival + walk_time
                        )
                        _insert_edge(
                            edge_map,
                            Edge(
                                previous.nodes[-1],
                                transfer_node,
                                cost=walk_time,
                                capacity=INF,
                                type="transfer",
                            ),
                        )
                        _insert_edge(
                            edge_map,
                            Edge(
                                transfer_node,
                                ride.nodes[0],
                                cost=ride.departure
                                - (previous.arrival + walk_time),
                                capacity=INF,
                                type="boarding",
                            ),
                        )
                        nodes.extend([transfer_node, ride.nodes[0]])

                    for edge in ride.edges:
                        _insert_edge(edge_map, edge)
                        if nodes[-1] != edge.u:
                            raise ValueError(
                                "path assembly lost continuity at "
                                f"{edge.as_tuple()}"
                            )
                        nodes.append(edge.v)

                exit_cost = itinerary.final_walk + itinerary.schedule_penalty
                _insert_edge(
                    edge_map,
                    Edge(
                        nodes[-1],
                        sink,
                        cost=exit_cost,
                        capacity=INF,
                        type="exit",
                    ),
                )
                nodes.append(sink)

            label = f"g{group.group_id}_p{candidate_number}"
            pathset.path[next_path] = tuple(nodes)
            pathset.cost[next_path] = itinerary.generalized_cost
            pathset.od[next_path] = od
            pathset.od_paths.setdefault(od, []).append(next_path)
            pathset.label[next_path] = label
            pathset.index_by_label[label] = next_path
            for edge in zip(nodes, nodes[1:]):
                pathset.edge_paths.setdefault(edge, []).append(next_path)

            path_records.append(
                {
                    "path_index": next_path,
                    "path_label": label,
                    "group_id": group.group_id,
                    "origin": group.origin,
                    "destination": group.destination,
                    "route_id": itinerary.route_id,
                    "run_sequence": "|".join(
                        ride.run_id for ride in itinerary.rides
                    ),
                    "departure": itinerary.departure,
                    "arrival": itinerary.arrival,
                    "generalized_cost": itinerary.generalized_cost,
                    "schedule_penalty": itinerary.schedule_penalty,
                    "total_left_behind": itinerary.total_left_behind,
                    "left_behind_by_leg": "|".join(
                        str(value)
                        for value in itinerary.left_behind_by_leg
                    ),
                }
            )
            next_path += 1

    if missing_groups:
        missing_volume = sum(group.volume for group in missing_groups)
        sample = [
            (
                group.group_id,
                group.origin,
                group.destination,
                group.desired_start,
            )
            for group in missing_groups[:20]
        ]
        raise ValueError(
            f"{len(missing_groups)} demand groups with volume {missing_volume} "
            f"have no timetable path before {timetable_end}; sample={sample}"
        )

    instance = TransitInstance(
        name="hong_kong_mtr_1800_1900",
        edges=list(edge_map.values()),
        demand=demand,
        node_time=node_time,
        metadata={
            "demand_start": demand_start,
            "demand_end": demand_end,
            "timetable_end": timetable_end,
            "max_left_behind": max_left_behind,
            "paths_per_route": paths_per_route,
            "passengers_per_car": passengers_per_car,
            "transfer_time": transfer_time,
            "early_penalty": early_penalty,
            "late_penalty": late_penalty,
        },
    )
    quality_control = {
        **demand_payload["quality"],
        **timetable_stats,
        "physical_routes": sum(len(routes) for routes in routes_by_od.values()),
        "candidate_paths": len(pathset),
        "finite_capacity_segments": sum(
            not np.isinf(instance.edge_capacity(*edge))
            for edge in pathset.edge_paths
        ),
        "maximum_generated_left_behind": max(
            record["total_left_behind"] for record in path_records
        ),
        "latest_generated_arrival": max(
            record["arrival"] for record in path_records
        ),
    }
    return MTRBuildResult(
        instance=instance,
        pathset=pathset,
        demand_records=demand_payload["records"],
        route_records=route_records,
        path_records=path_records,
        quality_control=quality_control,
    )
