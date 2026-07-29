"""Network primitives for schedule-based dynamic transit assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Hashable

import networkx as nx

# Edge ``type`` values. ``in-vehicle`` links are the ones that carry a hard
# capacity and over which boarding priority is contested; the others are
# auxiliary (walking / waiting / entering / leaving the system).
SEGMENT_TYPES = (
    "in-vehicle",
    "boarding",
    "alighting",
    "transfer",
    "demand",
    "exit",
)

Node = Hashable


@dataclass(frozen=True)
class Edge:
    """A typed, costed, capacitated arc of the transit network."""

    u: Node
    v: Node
    cost: float
    capacity: float = float("inf")
    type: str = "in-vehicle"

    def as_tuple(self) -> Tuple[Node, Node]:
        return (self.u, self.v)


@dataclass(frozen=True)
class BoardingConflict:
    """One path-level boarding-priority relation at a contested link.

    ``higher_path`` and ``lower_path`` refer to labels in
    :attr:`TransitInstance.explicit_path_labels`. The two paths must both use
    ``contested_edge``. ``board_time`` is the time used by the timely-last QUE
    cycle-breaking rule; only its ordering matters in canonical examples.
    """

    higher_path: str
    lower_path: str
    contested_edge: Tuple[Node, Node]
    board_time: float = 0.0


@dataclass
class TransitInstance:
    """A single assignment instance.

    Parameters
    ----------
    name:
        Human-readable identifier used when writing results.
    edges:
        Iterable of :class:`Edge` (or ``(u, v, cost, capacity, type)`` tuples).
    demand:
        Mapping ``(origin, destination) -> volume``.
    node_time:
        Optional mapping ``node -> scalar time``. Used by the automatic
        boarding-priority rule and the timely-last cycle-breaking rule. Nodes
        without an entry are treated as having time ``0``.
    boarding_conflicts:
        Explicit path-level boarding-priority relations. This is the preferred
        representation for manuscript figures because it records the exact
        paths and contested link shown in each figure.
    boarding_scenarios:
        Optional explicit list of ``(link_a, link_b)`` pairs. When given, every
        path traversing ``link_a`` is declared to have boarding priority over
        every path traversing ``link_b`` (the encoding used by the Nguyen
        example). When omitted, boarding priority is derived automatically from
        ``node_time`` on each shared in-vehicle link (see
        :func:`dta.priority_graph.build_priority_graph`).
    explicit_paths:
        Optional path list for paper examples whose route alternatives are
        defined explicitly. When present, path enumeration uses only these
        routes rather than all simple graph paths.
    explicit_path_labels:
        Stable labels aligned one-to-one with ``explicit_paths``.
    metadata:
        Reproducibility information and expected figure properties.
    """

    name: str
    edges: Sequence
    demand: Dict[Tuple[Node, Node], float]
    node_time: Dict[Node, float] = field(default_factory=dict)
    boarding_conflicts: Optional[List[BoardingConflict]] = None
    boarding_scenarios: Optional[List[Tuple[Tuple[Node, Node], Tuple[Node, Node]]]] = None
    explicit_paths: Optional[List[Sequence[Node]]] = None
    explicit_path_labels: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    G: nx.DiGraph = field(init=False)

    def __post_init__(self) -> None:
        self.edges = [self._coerce_edge(e) for e in self.edges]
        if self.explicit_path_labels is not None:
            if self.explicit_paths is None:
                raise ValueError("explicit_path_labels requires explicit_paths")
            if len(self.explicit_path_labels) != len(self.explicit_paths):
                raise ValueError(
                    "explicit_path_labels and explicit_paths must have equal length"
                )
            if len(set(self.explicit_path_labels)) != len(self.explicit_path_labels):
                raise ValueError("explicit path labels must be unique")
        if self.boarding_conflicts and self.explicit_path_labels is None:
            raise ValueError(
                "boarding_conflicts requires stable explicit_path_labels"
            )
        self.G = self._build_graph()
        self._validate_boarding_conflicts()

    @staticmethod
    def _coerce_edge(e) -> Edge:
        if isinstance(e, Edge):
            return e
        # tuple form: (u, v, cost[, capacity[, type]])
        u, v, cost = e[0], e[1], e[2]
        capacity = e[3] if len(e) > 3 else float("inf")
        etype = e[4] if len(e) > 4 else "in-vehicle"
        return Edge(u, v, cost, capacity, etype)

    def _build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph(name=self.name)
        for e in self.edges:
            if e.type not in SEGMENT_TYPES:
                raise ValueError(
                    f"edge {e.as_tuple()} has unknown type {e.type!r}; "
                    f"expected one of {SEGMENT_TYPES}"
                )
            G.add_edge(e.u, e.v, cost=e.cost, capacity=e.capacity, type=e.type)
        for node, t in self.node_time.items():
            if node in G:
                G.nodes[node]["time"] = t
        return G

    def _validate_boarding_conflicts(self) -> None:
        if not self.boarding_conflicts:
            return
        labels = set(self.explicit_path_labels or [])
        paths = {
            label: tuple(path)
            for label, path in zip(
                self.explicit_path_labels or [], self.explicit_paths or []
            )
        }
        for conflict in self.boarding_conflicts:
            if conflict.higher_path not in labels or conflict.lower_path not in labels:
                raise ValueError(
                    "boarding conflict references an unknown path label: "
                    f"{conflict.higher_path} > {conflict.lower_path}"
                )
            edge = conflict.contested_edge
            if edge not in self.G.edges:
                raise ValueError(
                    f"boarding conflict uses missing contested edge {edge}"
                )
            for label in (conflict.higher_path, conflict.lower_path):
                path_edges = set(zip(paths[label], paths[label][1:]))
                if edge not in path_edges:
                    raise ValueError(
                        f"path {label} does not use contested edge {edge}"
                    )
            if self.edge_type(*edge) != "in-vehicle":
                raise ValueError(
                    f"contested edge {edge} must be an in-vehicle edge"
                )
            if self.edge_capacity(*edge) == float("inf"):
                raise ValueError(
                    f"contested edge {edge} must have finite capacity"
                )

    # -- convenience accessors -------------------------------------------------

    def edge_cost(self, u: Node, v: Node) -> float:
        return self.G.edges[(u, v)]["cost"]

    def edge_capacity(self, u: Node, v: Node) -> float:
        return self.G.edges[(u, v)]["capacity"]

    def edge_type(self, u: Node, v: Node) -> str:
        return self.G.edges[(u, v)]["type"]

    def time_of(self, node: Node) -> float:
        """Scalar time of ``node`` (0 when unknown)."""
        return float(self.node_time.get(node, 0.0))

    def total_demand(self) -> float:
        return float(sum(self.demand.values()))
