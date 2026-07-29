"""Path-priority graph construction and deterministic QUE cycle breaking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Hashable

import networkx as nx

from .network import TransitInstance
from .paths import PathSet

Node = Hashable
COST_PRIORITY = "cost_priority"
BOARDING_PRIORITY = "boarding_priority"


@dataclass
class RemovedEdge:
    """Record of one removed boarding-priority relation."""

    edge: Tuple[int, int]
    type: str
    reason: str
    cycle: List[Tuple[int, int]]


def relation_types(pg: nx.DiGraph, u: int, v: int) -> frozenset[str]:
    """Return all relation types represented by the directed pair ``u -> v``."""
    return frozenset(pg.edges[u, v].get("relation_types", ()))


def has_relation(pg: nx.DiGraph, u: int, v: int, relation: str) -> bool:
    return pg.has_edge(u, v) and relation in relation_types(pg, u, v)


def _add_relation(
    pg: nx.DiGraph, u: int, v: int, relation: str, **attributes
) -> None:
    """Add one relation without overwriting another relation on the same pair."""
    data = dict(pg.edges[u, v]) if pg.has_edge(u, v) else {}
    relations = set(data.get("relation_types", ()))
    relations.add(relation)
    data["relation_types"] = frozenset(relations)

    if relation == COST_PRIORITY:
        data["cost_gap"] = max(
            float(attributes.get("cost_gap", 0.0)),
            float(data.get("cost_gap", 0.0)),
        )
    else:
        times = list(data.get("board_times", ()))
        times.append(float(attributes.get("board_time", 0.0)))
        data["board_times"] = tuple(sorted(times))
        conflicts = list(data.get("boarding_conflicts", ()))
        conflict = attributes.get("conflict")
        if conflict is not None:
            conflicts.append(conflict)
        data["boarding_conflicts"] = tuple(conflicts)
    pg.add_edge(u, v, **data)


def _remove_relation(
    pg: nx.DiGraph, edge: Tuple[int, int], relation: str
) -> None:
    data = dict(pg.edges[edge])
    relations = set(data.get("relation_types", ()))
    relations.discard(relation)
    if not relations:
        pg.remove_edge(*edge)
        return
    data["relation_types"] = frozenset(relations)
    if relation == BOARDING_PRIORITY:
        data.pop("board_times", None)
        data.pop("boarding_conflicts", None)
    pg.edges[edge].clear()
    pg.edges[edge].update(data)


def _latest_boarding_node(
    instance: TransitInstance, path: Sequence[Node], link_source: Node
) -> Optional[Node]:
    """Return the latest path node from which the passenger entered a vehicle.

    The immediate predecessor of ``link_source`` is not necessarily a boarding
    event: it can be an earlier in-vehicle node. Scanning backward until the
    preceding arc is not in-vehicle distinguishes passengers already on board
    from passengers entering at the contested movement.
    """
    try:
        source_pos = path.index(link_source)
    except ValueError:
        return None
    pos = source_pos
    while pos > 0:
        edge = (path[pos - 1], path[pos])
        if instance.edge_type(*edge) != "in-vehicle":
            return path[pos - 1]
        pos -= 1
    return path[0]


def build_priority_graph(instance: TransitInstance, pathset: PathSet) -> nx.DiGraph:
    """Build the path-priority graph while preserving both relation types."""
    pg = nx.DiGraph(name=f"path_priority_{instance.name}")
    pg.add_nodes_from(
        (p, {"label": pathset.label[p], "cost": pathset.cost[p]})
        for p in pathset.indices()
    )

    for path_list in pathset.od_paths.values():
        ordered = sorted(path_list, key=lambda p: (pathset.cost[p], p))
        for i, higher in enumerate(ordered):
            for lower in ordered[i + 1 :]:
                gap = pathset.cost[lower] - pathset.cost[higher]
                if gap > 0:
                    _add_relation(
                        pg, higher, lower, COST_PRIORITY, cost_gap=gap
                    )

    if instance.boarding_conflicts:
        for conflict in instance.boarding_conflicts:
            higher = pathset.index(conflict.higher_path)
            lower = pathset.index(conflict.lower_path)
            _add_relation(
                pg,
                higher,
                lower,
                BOARDING_PRIORITY,
                board_time=conflict.board_time,
                conflict=conflict,
            )
    elif instance.boarding_scenarios:
        for higher_entry, lower_entry in instance.boarding_scenarios:
            for higher in pathset.edge_paths.get(higher_entry, []):
                for lower in pathset.edge_paths.get(lower_entry, []):
                    if higher != lower:
                        _add_relation(
                            pg,
                            higher,
                            lower,
                            BOARDING_PRIORITY,
                            board_time=instance.time_of(higher_entry[0]),
                        )
    else:
        for edge, paths in pathset.edge_paths.items():
            if instance.edge_type(*edge) != "in-vehicle":
                continue
            timed = [
                (
                    instance.time_of(
                        _latest_boarding_node(instance, pathset.path[p], edge[0])
                    ),
                    p,
                )
                for p in paths
            ]
            timed.sort(key=lambda item: (item[0], item[1]))
            for i, (higher_time, higher) in enumerate(timed):
                for lower_time, lower in timed[i + 1 :]:
                    if higher_time < lower_time:
                        _add_relation(
                            pg,
                            higher,
                            lower,
                            BOARDING_PRIORITY,
                            board_time=higher_time,
                        )
    return pg


def _select_edge_to_remove(
    pg: nx.DiGraph, cycle: List[Tuple[int, int]]
) -> Tuple[Tuple[int, int], str, str]:
    """Select the timely-last boarding-only relation on a directed loop."""
    candidates = [
        edge
        for edge in cycle
        if BOARDING_PRIORITY in relation_types(pg, *edge)
        and COST_PRIORITY not in relation_types(pg, *edge)
    ]
    if not candidates:
        labels = [
            (pg.nodes[u].get("label", u), pg.nodes[v].get("label", v))
            for u, v in cycle
        ]
        raise ValueError(
            "cyclic path-priority graph has no boarding-only relation that can "
            f"be removed while retaining every cost relation: {labels}"
        )
    edge = max(
        candidates,
        key=lambda e: (
            max(pg.edges[e].get("board_times", (0.0,))),
            str(pg.nodes[e[0]].get("label", e[0])),
            str(pg.nodes[e[1]].get("label", e[1])),
        ),
    )
    return edge, BOARDING_PRIORITY, "timely-last boarding-priority relation"


def break_cycles(pg: nx.DiGraph) -> Tuple[nx.DiGraph, List[RemovedEdge]]:
    """Remove boarding-priority relations until the graph is acyclic."""
    broken = pg.copy()
    removed: List[RemovedEdge] = []
    while True:
        try:
            raw_cycle = list(nx.find_cycle(broken, orientation="original"))
        except nx.NetworkXNoCycle:
            break
        cycle = [(u, v) for u, v, *_ in raw_cycle]
        edge, relation, reason = _select_edge_to_remove(broken, cycle)
        _remove_relation(broken, edge, relation)
        removed.append(
            RemovedEdge(
                edge=edge,
                type=relation,
                reason=reason,
                cycle=cycle,
            )
        )
    return broken, removed


def linear_extension(pg: nx.DiGraph) -> List[int]:
    """Return a deterministic linear extension of an acyclic priority graph."""
    if not nx.is_directed_acyclic_graph(pg):
        raise ValueError("priority graph is cyclic; call break_cycles first")
    return list(
        nx.lexicographical_topological_sort(
            pg, key=lambda p: (str(pg.nodes[p].get("label", p)), p)
        )
    )
