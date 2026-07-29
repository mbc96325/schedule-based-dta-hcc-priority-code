"""
Path enumeration and the derived index structures used by the assignment and
priority-graph code.

This mirrors ``DTA.path_enumeration`` from the prototype scripts but returns a
structured :class:`PathSet` instead of mutating instance attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Hashable

import networkx as nx

from .network import TransitInstance

Node = Hashable
Path = Tuple[Node, ...]


@dataclass
class PathSet:
    """All enumerated paths plus the lookup tables built over them.

    Path indices are 1-based to match the prototype's printouts.
    """

    path: Dict[int, Path] = field(default_factory=dict)
    cost: Dict[int, float] = field(default_factory=dict)
    od: Dict[int, Tuple[Node, Node]] = field(default_factory=dict)
    od_paths: Dict[Tuple[Node, Node], List[int]] = field(default_factory=dict)
    edge_paths: Dict[Tuple[Node, Node], List[int]] = field(default_factory=dict)
    label: Dict[int, str] = field(default_factory=dict)
    index_by_label: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.path)

    def indices(self) -> List[int]:
        return list(self.path.keys())

    def edges_of(self, path_ind: int) -> List[Tuple[Node, Node]]:
        p = self.path[path_ind]
        return [(p[i], p[i + 1]) for i in range(len(p) - 1)]

    def index(self, label: str) -> int:
        """Return the path index associated with a stable manuscript label."""
        return self.index_by_label[label]


def enumerate_paths(instance: TransitInstance) -> PathSet:
    """Enumerate every simple path for each OD pair with positive demand.

    Path cost is the sum of edge costs along the path. Builds the
    ``od -> [path]`` and ``edge -> [path]`` indices used downstream.
    """

    ps = PathSet()
    next_ind = 1
    G = instance.G

    if instance.explicit_paths is not None:
        labels = instance.explicit_path_labels or [
            f"path_{i}" for i in range(1, len(instance.explicit_paths) + 1)
        ]
        for label, path in zip(labels, instance.explicit_paths):
            path_t = tuple(path)
            od = (path_t[0], path_t[-1])
            if instance.demand.get(od, 0.0) <= 0:
                continue
            ps.path[next_ind] = path_t
            ps.label[next_ind] = label
            ps.index_by_label[label] = next_ind
            ps.od[next_ind] = od
            ps.od_paths.setdefault(od, []).append(next_ind)

            cost = 0.0
            for i in range(len(path_t) - 1):
                edge = (path_t[i], path_t[i + 1])
                if edge not in G.edges:
                    raise ValueError(
                        f"explicit path {path_t} uses missing edge {edge}"
                    )
                ps.edge_paths.setdefault(edge, []).append(next_ind)
                cost += G.edges[edge]["cost"]
            ps.cost[next_ind] = cost
            next_ind += 1
        return ps

    for (ori, dest), vol in instance.demand.items():
        if vol <= 0:
            continue
        if ori not in G or dest not in G:
            continue
        for path in nx.all_simple_paths(G, ori, dest):
            path_t = tuple(path)
            ps.path[next_ind] = path_t
            label = f"path_{next_ind}"
            ps.label[next_ind] = label
            ps.index_by_label[label] = next_ind
            ps.od[next_ind] = (ori, dest)
            ps.od_paths.setdefault((ori, dest), []).append(next_ind)

            cost = 0.0
            for i in range(len(path_t) - 1):
                edge = (path_t[i], path_t[i + 1])
                ps.edge_paths.setdefault(edge, []).append(next_ind)
                cost += G.edges[edge]["cost"]
            ps.cost[next_ind] = cost
            next_ind += 1

    return ps


def restrict_pathset(pathset: PathSet, selected: set[int]) -> PathSet:
    """Return the path-set restriction induced by ``selected`` path indices."""
    restricted = PathSet()
    for path_ind in pathset.indices():
        if path_ind not in selected:
            continue
        restricted.path[path_ind] = pathset.path[path_ind]
        restricted.cost[path_ind] = pathset.cost[path_ind]
        restricted.od[path_ind] = pathset.od[path_ind]
        restricted.label[path_ind] = pathset.label[path_ind]
        restricted.index_by_label[pathset.label[path_ind]] = path_ind
        restricted.od_paths.setdefault(pathset.od[path_ind], []).append(path_ind)
        for edge in pathset.edges_of(path_ind):
            restricted.edge_paths.setdefault(edge, []).append(path_ind)
    return restricted
