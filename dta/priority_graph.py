"""Path-priority graph construction and deterministic QUE cycle breaking."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Hashable

import networkx as nx
import numpy as np

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


def build_compact_priority_graph(
    instance: TransitInstance,
    pathset: PathSet,
    *,
    tie_break_equal_boarding_times: bool = True,
) -> nx.DiGraph:
    """Build a sparse precedence graph for a large prescribed path set.

    Cost and boarding priorities each induce an ordered list. The full graph
    connects every earlier path to every later path in the corresponding list.
    This routine stores only adjacent comparisons, whose transitive closure
    imposes the same order. Equal platform-arrival times are resolved by stable
    path labels when ``tie_break_equal_boarding_times`` is true.
    """
    pg = nx.DiGraph(name=f"compact_path_priority_{instance.name}")
    pg.add_nodes_from(
        (p, {"label": pathset.label[p], "cost": pathset.cost[p]})
        for p in pathset.indices()
    )

    for path_list in pathset.od_paths.values():
        ordered = sorted(
            path_list,
            key=lambda p: (pathset.cost[p], str(pathset.label[p]), p),
        )
        for higher, lower in zip(ordered, ordered[1:]):
            gap = pathset.cost[lower] - pathset.cost[higher]
            if gap > 0:
                _add_relation(
                    pg, higher, lower, COST_PRIORITY, cost_gap=gap
                )

    if instance.boarding_conflicts:
        for conflict in instance.boarding_conflicts:
            _add_relation(
                pg,
                pathset.index(conflict.higher_path),
                pathset.index(conflict.lower_path),
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
        edge_boarders: Dict[
            Tuple[Node, Node], List[Tuple[float, str, int]]
        ] = defaultdict(list)
        path_boarding_times: Dict[int, List[float]] = defaultdict(list)
        for path_ind in pathset.indices():
            path = pathset.path[path_ind]
            latest_entry = path[0]
            for u, v in zip(path, path[1:]):
                edge_type = instance.edge_type(u, v)
                if edge_type != "in-vehicle":
                    latest_entry = u
                    continue
                if np.isinf(instance.edge_capacity(u, v)):
                    continue
                entry_time = instance.time_of(latest_entry)
                edge_boarders[(u, v)].append(
                    (
                        entry_time,
                        str(pathset.label[path_ind]),
                        path_ind,
                    )
                )
                path_boarding_times[path_ind].append(entry_time)

        for path_ind, boarding_times in path_boarding_times.items():
            pg.nodes[path_ind]["earliest_boarding_time"] = min(
                boarding_times
            )

        for edge, boarders in edge_boarders.items():
            ordered = sorted(boarders)
            for higher, lower in zip(ordered, ordered[1:]):
                higher_time, _, higher_path = higher
                lower_time, _, lower_path = lower
                if (
                    higher_time < lower_time
                    or tie_break_equal_boarding_times
                ):
                    _add_relation(
                        pg,
                        higher_path,
                        lower_path,
                        BOARDING_PRIORITY,
                        board_time=higher_time,
                        conflict=edge,
                    )
    pg.graph["compact"] = True
    pg.graph["tie_break_equal_boarding_times"] = (
        tie_break_equal_boarding_times
    )
    return pg


def cost_compatible_projection_order(
    pg: nx.DiGraph,
    *,
    removed_sample_limit: int = 100,
) -> Tuple[List[int], int, List[RemovedEdge]]:
    """Project a large priority graph onto a cost-compatible total order.

    The cost-priority subgraph is acyclic because it consists of within-group
    strict cost orders. A deterministic topological sort of this subgraph
    preserves every cost relation and uses earliest boarding time to order
    paths that are otherwise incomparable. Retaining only boarding relations
    that point forward in the resulting total order yields an acyclic
    subrelation without repeated whole-graph SCC scans.
    """
    cost_graph = nx.DiGraph()
    cost_graph.add_nodes_from(pg.nodes(data=True))
    cost_graph.add_edges_from(
        (u, v)
        for u, v in pg.edges
        if COST_PRIORITY in relation_types(pg, u, v)
    )
    if not nx.is_directed_acyclic_graph(cost_graph):
        raise ValueError("cost-priority relations must be acyclic")

    order = list(
        nx.lexicographical_topological_sort(
            cost_graph,
            key=lambda path: (
                float(
                    pg.nodes[path].get(
                        "earliest_boarding_time", float("inf")
                    )
                ),
                float(pg.nodes[path].get("cost", 0.0)),
                str(pg.nodes[path].get("label", path)),
                path,
            ),
        )
    )
    position = {path: rank for rank, path in enumerate(order)}
    removed_count = 0
    removed_sample: List[RemovedEdge] = []
    for u, v in pg.edges:
        if position[u] < position[v]:
            continue
        relations = relation_types(pg, u, v)
        if COST_PRIORITY in relations:
            raise ValueError(
                "cost-compatible projection reversed a cost-priority relation"
            )
        if BOARDING_PRIORITY not in relations:
            continue
        removed_count += 1
        if len(removed_sample) < removed_sample_limit:
            removed_sample.append(
                RemovedEdge(
                    edge=(u, v),
                    type=BOARDING_PRIORITY,
                    reason=(
                        "boarding-priority relation conflicts with the "
                        "cost-compatible projection order"
                    ),
                    cycle=[],
                )
            )
    return order, removed_count, removed_sample


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


def break_cycles_by_scc(
    pg: nx.DiGraph,
) -> Tuple[nx.DiGraph, List[RemovedEdge]]:
    """Break cycles in batches using strongly connected components.

    Every edge internal to a nontrivial strongly connected component lies on a
    directed cycle. In each round, the timely-last boarding-only relation is
    removed from every cyclic component. This preserves all cost-priority
    relations and avoids repeatedly scanning unrelated acyclic regions.
    """
    broken = pg.copy()
    removed: List[RemovedEdge] = []

    while True:
        cyclic_components = [
            component
            for component in nx.strongly_connected_components(broken)
            if len(component) > 1
            or any(broken.has_edge(node, node) for node in component)
        ]
        if not cyclic_components:
            break

        selections = []
        for component in sorted(
            cyclic_components,
            key=lambda nodes: min(
                (str(broken.nodes[node].get("label", node)), node)
                for node in nodes
            ),
        ):
            candidates = [
                (u, v)
                for u in component
                for v in broken.successors(u)
                if v in component
                and BOARDING_PRIORITY in relation_types(broken, u, v)
                and COST_PRIORITY not in relation_types(broken, u, v)
            ]
            if not candidates:
                labels = sorted(
                    str(broken.nodes[node].get("label", node))
                    for node in component
                )
                raise ValueError(
                    "cyclic component has no boarding-only relation that can "
                    f"be removed while retaining cost priority: {labels[:20]}"
                )
            edge = max(
                candidates,
                key=lambda e: (
                    max(broken.edges[e].get("board_times", (0.0,))),
                    str(broken.nodes[e[0]].get("label", e[0])),
                    str(broken.nodes[e[1]].get("label", e[1])),
                ),
            )
            return_path = nx.shortest_path(
                broken.subgraph(component), source=edge[1], target=edge[0]
            )
            cycle = [edge] + list(zip(return_path, return_path[1:]))
            selections.append((edge, cycle))

        for edge, cycle in selections:
            _remove_relation(broken, edge, BOARDING_PRIORITY)
            removed.append(
                RemovedEdge(
                    edge=edge,
                    type=BOARDING_PRIORITY,
                    reason="timely-last boarding-priority relation in SCC",
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
