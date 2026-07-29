"""Exact computational specifications for manuscript Figures 4 and 6--11.

The node sequences below follow the auxiliary nodes shown in the figures.
Paths omitted graphically but required by the text are represented explicitly
as ``p*_bar`` fallback paths.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import _bootstrap  # noqa: F401
from dta import BoardingConflict, Edge, TransitInstance

INF = float("inf")
Node = str
Route = Sequence[Node]


def _edge_set_from_routes(
    routes: Mapping[str, Route],
    edge_data: Mapping[Tuple[Node, Node], Tuple[float, float, str]],
) -> List[Edge]:
    """Create one consistent edge object for every arc used by the routes."""
    built: Dict[Tuple[Node, Node], Edge] = {}
    for route in routes.values():
        for position, edge in enumerate(zip(route, route[1:])):
            if edge in edge_data:
                cost, capacity, edge_type = edge_data[edge]
            elif position == 0:
                cost, capacity, edge_type = 0.0, INF, "demand"
            elif position == len(route) - 2:
                cost, capacity, edge_type = 0.0, INF, "exit"
            else:
                cost, capacity, edge_type = 0.0, INF, "transfer"
            candidate = Edge(
                edge[0], edge[1], float(cost), float(capacity), edge_type
            )
            if edge in built and built[edge] != candidate:
                raise ValueError(f"inconsistent data for shared edge {edge}")
            built[edge] = candidate
    return list(built.values())


def _build(
    *,
    name: str,
    figure: str,
    routes: Mapping[str, Route],
    demand: Mapping[Tuple[Node, Node], float],
    edge_data: Mapping[Tuple[Node, Node], Tuple[float, float, str]],
    conflicts: Iterable[Tuple[str, str, Tuple[Node, Node], float]],
    expected: dict,
) -> TransitInstance:
    conflict_objects = [
        BoardingConflict(higher, lower, edge, time)
        for higher, lower, edge, time in conflicts
    ]
    metadata = {
        "figure": figure,
        "expected": expected,
        "finite_capacities": {
            edge: values[1]
            for edge, values in edge_data.items()
            if values[1] != INF
        },
        "boarding_relations": [
            (higher, lower, edge)
            for higher, lower, edge, _ in conflicts
        ],
    }
    return TransitInstance(
        name=name,
        edges=_edge_set_from_routes(routes, edge_data),
        demand=dict(demand),
        boarding_conflicts=conflict_objects,
        explicit_paths=list(routes.values()),
        explicit_path_labels=list(routes.keys()),
        metadata=metadata,
    )


def _with_fallbacks(
    lower_routes: Mapping[str, Route],
    lower_costs: Mapping[str, float],
) -> Dict[str, Route]:
    routes: Dict[str, Route] = {}
    for label, route in lower_routes.items():
        routes[label] = route
        origin, destination = route[0], route[-1]
        routes[f"{label}_bar"] = (
            origin,
            f"{label}_fallback",
            destination,
        )
    return routes


def _fallback_edge_data(
    lower_routes: Mapping[str, Route],
    lower_costs: Mapping[str, float],
) -> Dict[Tuple[Node, Node], Tuple[float, float, str]]:
    return {
        (route[0], f"{label}_fallback"): (
            float(lower_costs[label] + 1.0),
            INF,
            "boarding",
        )
        for label, route in lower_routes.items()
    }


def figure4_ue_neq_so() -> TransitInstance:
    """Figure 4: acyclic priority graph with UE different from SO."""
    routes = {
        "p1": (
            "a_g", "a_tilde_1", "a_1", "b_1_prime", "h_1",
            "f_1_prime", "f_g",
        ),
        "p2": (
            "a_g", "a_tilde_1", "a_1", "b_1_prime",
            "b_tilde_1_double", "b_1_double", "e_1",
            "f_1_double", "f_g",
        ),
        "p3": (
            "h_g_prime", "h_tilde_1", "h_1", "f_1_prime",
            "f_g_prime",
        ),
        "p4": (
            "h_g_prime", "h_tilde_1", "h_2", "f_2_prime",
            "f_g_prime",
        ),
    }
    edge_data = {
        ("a_1", "b_1_prime"): (5, INF, "in-vehicle"),
        ("b_1_prime", "h_1"): (5, INF, "in-vehicle"),
        ("h_1", "f_1_prime"): (5, 1, "in-vehicle"),
        ("b_1_prime", "b_tilde_1_double"): (1, INF, "transfer"),
        ("b_1_double", "e_1"): (5, INF, "in-vehicle"),
        ("e_1", "f_1_double"): (5, INF, "in-vehicle"),
        ("h_tilde_1", "h_2"): (5, INF, "boarding"),
        ("h_2", "f_2_prime"): (5, INF, "in-vehicle"),
    }
    return _build(
        name="figure4_ue_neq_so",
        figure="4",
        routes=routes,
        demand={("a_g", "f_g"): 1, ("h_g_prime", "f_g_prime"): 1},
        edge_data=edge_data,
        conflicts=[("p1", "p3", ("h_1", "f_1_prime"), 0)],
        expected={
            "path_costs": {"p1": 15, "p2": 16, "p3": 5, "p4": 10},
            "so_cost": 21,
            "so_flow": {"p1": 0, "p2": 1, "p3": 1, "p4": 0},
            "ue_exists": True,
            "ue_cost": 25,
            "ue_flows": [{"p1": 1, "p2": 0, "p3": 0, "p4": 1}],
            "que_cost": 25,
        },
    )


def _figure6(
    *,
    panel: str,
    demand_volume: float,
    capacities: Tuple[float, float, float],
) -> TransitInstance:
    lower = {
        "p1": (
            "a_g1", "a_tilde_1", "a_1", "b_1", "c_1_prime", "d_1",
            "e_1_prime", "e_tilde_1_double", "e_1_double", "i_1",
            "i_g1",
        ),
        "p2": (
            "b_g2", "b_tilde_1", "b_1", "c_1_prime",
            "c_tilde_1_double", "c_1_double", "k_1", "j_1", "j_g2",
        ),
        "p3": (
            "k_g3", "k_tilde_1", "k_1", "j_1", "h_1",
            "e_1_double", "i_1", "i_g3",
        ),
    }
    lower_costs = {"p1": 2, "p2": 2, "p3": 2}
    routes = _with_fallbacks(lower, lower_costs)
    edges = {
        ("b_1", "c_1_prime"): (1, capacities[0], "in-vehicle"),
        ("k_1", "j_1"): (1, capacities[1], "in-vehicle"),
        ("e_1_double", "i_1"): (1, capacities[2], "in-vehicle"),
        **_fallback_edge_data(lower, lower_costs),
    }
    if panel == "a":
        expected_flow = {
            "p1": 0.5, "p1_bar": 0.5,
            "p2": 0.5, "p2_bar": 0.5,
            "p3": 0.5, "p3_bar": 0.5,
        }
        ue_cost = 7.5
        que_cost = 8.0
    else:
        expected_flow = {
            "p1": 1, "p1_bar": 1,
            "p2": 0, "p2_bar": 2,
            "p3": 1, "p3_bar": 1,
        }
        ue_cost = 16.0
        que_cost = 16.0
    return _build(
        name=f"figure6{panel}_pure_boarding",
        figure=f"6({panel})",
        routes=routes,
        demand={
            ("a_g1", "i_g1"): demand_volume,
            ("b_g2", "j_g2"): demand_volume,
            ("k_g3", "i_g3"): demand_volume,
        },
        edge_data=edges,
        conflicts=[
            ("p1", "p2", ("b_1", "c_1_prime"), 0),
            ("p2", "p3", ("k_1", "j_1"), 1),
            ("p3", "p1", ("e_1_double", "i_1"), 2),
        ],
        expected={
            "path_costs": {
                "p1": 2, "p1_bar": 3,
                "p2": 2, "p2_bar": 3,
                "p3": 2, "p3_bar": 3,
            },
            "so_cost": ue_cost,
            "ue_exists": True,
            "ue_cost": ue_cost,
            "ue_flows": [expected_flow],
            "que_cost": que_cost,
        },
    )


def figure6a_pure_boarding_symmetric() -> TransitInstance:
    return _figure6(panel="a", demand_volume=1, capacities=(1, 1, 1))


def figure6b_pure_boarding_asymmetric() -> TransitInstance:
    return _figure6(panel="b", demand_volume=2, capacities=(1, 1, 3))


def figure7_pure_boarding_four_path() -> TransitInstance:
    """Figure 7: four-path pure boarding loop with multiple UE flows."""
    lower = {
        "p1": (
            "a_g1", "a_tilde_1", "a_1", "b_1", "c_1_prime", "d_1",
            "e_1_prime", "e_tilde_1_double", "e_1_double", "m_1",
            "m_g1",
        ),
        "p2": (
            "b_g2", "b_tilde_1", "b_1", "c_1_prime",
            "c_tilde_1_double", "c_1_double", "k_1", "j_1", "j_g2",
        ),
        "p3": (
            "k_g3", "k_tilde_1", "k_1", "j_1", "h_1", "i_1",
            "i_g3",
        ),
        "p4": (
            "h_g4", "h_tilde_1", "h_1", "i_1", "e_1_double", "m_1",
            "m_g4",
        ),
    }
    lower_costs = {label: 2 for label in lower}
    routes = _with_fallbacks(lower, lower_costs)
    edges = {
        ("b_1", "c_1_prime"): (1, 1, "in-vehicle"),
        ("k_1", "j_1"): (1, 1, "in-vehicle"),
        ("h_1", "i_1"): (1, 1, "in-vehicle"),
        ("e_1_double", "m_1"): (1, 1, "in-vehicle"),
        **_fallback_edge_data(lower, lower_costs),
    }
    return _build(
        name="figure7_pure_boarding_four_path",
        figure="7",
        routes=routes,
        demand={
            ("a_g1", "m_g1"): 1,
            ("b_g2", "j_g2"): 1,
            ("k_g3", "i_g3"): 1,
            ("h_g4", "m_g4"): 1,
        },
        edge_data=edges,
        conflicts=[
            ("p1", "p2", ("b_1", "c_1_prime"), 0),
            ("p2", "p3", ("k_1", "j_1"), 1),
            ("p3", "p4", ("h_1", "i_1"), 2),
            ("p4", "p1", ("e_1_double", "m_1"), 3),
        ],
        expected={
            "path_costs": {
                **{label: 2 for label in lower},
                **{f"{label}_bar": 3 for label in lower},
            },
            "so_cost": 10,
            "ue_exists": True,
            "ue_cost": 10,
            "ue_flows": [
                {
                    "p1": 0.5, "p1_bar": 0.5,
                    "p2": 0.5, "p2_bar": 0.5,
                    "p3": 0.5, "p3_bar": 0.5,
                    "p4": 0.5, "p4_bar": 0.5,
                },
                {
                    "p1": 0.25, "p1_bar": 0.75,
                    "p2": 0.75, "p2_bar": 0.25,
                    "p3": 0.25, "p3_bar": 0.75,
                    "p4": 0.75, "p4_bar": 0.25,
                },
            ],
            "que_cost": 10,
        },
    )


def figure8_pure_boarding_interacting() -> TransitInstance:
    """Figure 8: two interacting pure boarding loops sharing path p3."""
    lower = {
        "p1": (
            "a_g1", "a_tilde_1", "a_1", "b_1", "c_1_prime", "d_1",
            "e_1_prime", "e_tilde_1_double", "e_1_double", "i_1",
            "i_g1",
        ),
        "p2": (
            "b_g2", "b_tilde_1", "b_1", "c_1_prime",
            "c_tilde_1_double", "c_1_double", "f_1", "j_1", "j_g2",
        ),
        "p3": (
            "f_g3", "f_tilde_1", "f_1", "j_1", "h_1", "k_1_prime",
            "k_tilde_1_double", "k_1_double", "n_1", "o_1",
            "e_1_double", "i_1", "i_g3",
        ),
        "p4": (
            "h_g4", "h_tilde_1", "h_1", "k_1_prime", "l_1", "m_1",
            "m_g4",
        ),
        "p5": (
            "l_g5", "l_tilde_1", "l_1", "m_1", "k_1_double", "n_1",
            "n_g5",
        ),
    }
    lower_costs = {"p1": 2, "p2": 2, "p3": 4, "p4": 2, "p5": 2}
    routes = _with_fallbacks(lower, lower_costs)
    edges = {
        ("b_1", "c_1_prime"): (1, 1, "in-vehicle"),
        ("f_1", "j_1"): (1, 1, "in-vehicle"),
        ("e_1_double", "i_1"): (1, 1, "in-vehicle"),
        ("h_1", "k_1_prime"): (1, 0.5, "in-vehicle"),
        ("l_1", "m_1"): (1, 0.5, "in-vehicle"),
        ("k_1_double", "n_1"): (1, 0.5, "in-vehicle"),
        **_fallback_edge_data(lower, lower_costs),
    }
    ue_flow = {
        "p1": 0.75, "p1_bar": 0.25,
        "p2": 0.25, "p2_bar": 0.75,
        "p3": 0.25, "p3_bar": 0.75,
        "p4": 0.25, "p4_bar": 0.75,
        "p5": 0.25, "p5_bar": 0.75,
    }
    return _build(
        name="figure8_pure_boarding_interacting",
        figure="8",
        routes=routes,
        demand={
            ("a_g1", "i_g1"): 1,
            ("b_g2", "j_g2"): 1,
            ("f_g3", "i_g3"): 1,
            ("h_g4", "m_g4"): 1,
            ("l_g5", "n_g5"): 1,
        },
        edge_data=edges,
        conflicts=[
            ("p1", "p2", ("b_1", "c_1_prime"), 0),
            ("p2", "p3", ("f_1", "j_1"), 1),
            ("p3", "p1", ("e_1_double", "i_1"), 2),
            ("p3", "p4", ("h_1", "k_1_prime"), 3),
            ("p4", "p5", ("l_1", "m_1"), 4),
            ("p5", "p3", ("k_1_double", "n_1"), 5),
        ],
        expected={
            "path_costs": {
                **lower_costs,
                **{
                    f"{label}_bar": cost + 1
                    for label, cost in lower_costs.items()
                },
            },
            "so_cost": 15.25,
            "ue_exists": True,
            "ue_cost": 15.25,
            "ue_flows": [ue_flow],
            "que_cost": 15.5,
        },
    )


def _mixed_single(
    *,
    name: str,
    figure: str,
    demand_volume: float,
    middle_label: str,
    bypass_label: str,
    middle_edge_cost: float,
    bypass_edge_cost: float,
    boarding_higher: str,
    expected: dict,
) -> TransitInstance:
    routes = {
        "p1": (
            "a_g", "a_tilde_1", "a_1", "b_1_prime",
            "b_tilde_1_double", "b_1_double", "d_1_double",
            "d_tilde_1_prime", "d_1_prime", "e_1_prime", "e_g",
        ),
        middle_label: (
            "a_g", "a_tilde_1", "a_1", "b_1_prime", "c_1",
            "d_1_prime", "e_1_prime", "e_g",
        ),
        bypass_label: (
            "a_g", "a_tilde_1", "a_1", "b_1_prime",
            "b_tilde_1_triple", "b_1_triple", "e_1_triple", "e_g",
        ),
    }
    routes = {label: routes[label] for label in ("p1", "p2", "p3")}
    edge_data = {
        ("a_1", "b_1_prime"): (5, INF, "in-vehicle"),
        ("b_1_prime", "b_tilde_1_double"): (1, INF, "transfer"),
        ("b_1_double", "d_1_double"): (5, INF, "in-vehicle"),
        ("d_1_prime", "e_1_prime"): (5, 2, "in-vehicle"),
        ("b_1_prime", "c_1"): (5, INF, "in-vehicle"),
        ("c_1", "d_1_prime"): (middle_edge_cost, INF, "in-vehicle"),
        ("b_1_prime", "b_tilde_1_triple"): (1, INF, "transfer"),
        ("b_1_triple", "e_1_triple"): (
            bypass_edge_cost, INF, "in-vehicle"
        ),
    }
    return _build(
        name=name,
        figure=figure,
        routes=routes,
        demand={("a_g", "e_g"): demand_volume},
        edge_data=edge_data,
        conflicts=[
            (
                boarding_higher,
                "p1",
                ("d_1_prime", "e_1_prime"),
                0,
            )
        ],
        expected=expected,
    )


def figure9_mixed_ue_exists() -> TransitInstance:
    """Figure 9: mixed component whose boarding edge is inactive at UE."""
    return _mixed_single(
        name="figure9_mixed_ue_exists",
        figure="9",
        demand_volume=3,
        middle_label="p3",
        bypass_label="p2",
        middle_edge_cost=20,
        bypass_edge_cost=15,
        boarding_higher="p3",
        expected={
            "path_costs": {"p1": 16, "p2": 21, "p3": 35},
            "so_cost": 53,
            "ue_exists": True,
            "ue_cost": 53,
            "ue_flows": [{"p1": 2, "p2": 1, "p3": 0}],
            "que_cost": 53,
        },
    )


def figure10_mixed_no_ue(demand_volume: float = 3) -> TransitInstance:
    """Figure 10: one mixed loop, with UE at demand two but not at three."""
    at_three = abs(demand_volume - 3.0) <= 1e-9
    at_two = abs(demand_volume - 2.0) <= 1e-9
    expected = {
        "path_costs": {"p1": 16, "p2": 25, "p3": 36},
        "so_cost": 68 if at_three else 32 if at_two else None,
        "ue_exists": False if at_three else True if at_two else None,
        "ue_cost": None if at_three else 32 if at_two else None,
        "ue_flows": [] if at_three else [{"p1": 2, "p2": 0, "p3": 0}],
        "que_cost": 68 if at_three else 32 if at_two else None,
    }
    return _mixed_single(
        name=f"figure10_mixed_q{demand_volume:g}",
        figure="10",
        demand_volume=demand_volume,
        middle_label="p2",
        bypass_label="p3",
        middle_edge_cost=10,
        bypass_edge_cost=30,
        boarding_higher="p2",
        expected=expected,
    )


def figure11_mixed_interacting_no_ue() -> TransitInstance:
    """Figure 11: two interacting mixed loops and no classical UE."""
    routes = {
        "p1": (
            "a_g", "a_tilde_1", "a_1", "b_1_prime",
            "b_tilde_1_double", "b_1_double", "d_1_double",
            "d_tilde_1_prime", "d_1_prime", "e_1_prime",
            "e_tilde_1_double", "e_1_double", "h_1_double", "h_g",
        ),
        "p2": (
            "a_g", "a_tilde_1", "a_1", "b_1_prime", "c_1",
            "d_1_prime", "e_1_prime", "e_tilde_1_triple",
            "e_1_triple", "f_1", "h_1_prime", "h_g",
        ),
        "p3": (
            "a_g", "a_tilde_1", "a_1", "b_1_prime",
            "b_tilde_1_triple", "b_1_triple", "e_1_triple",
            "f_1", "h_1_prime", "h_g",
        ),
        "p4": (
            "a_g", "a_tilde_1", "a_1", "b_1_prime",
            "b_tilde_1_quadruple", "b_1_quadruple",
            "h_1_triple", "h_g",
        ),
    }
    edge_data = {
        ("a_1", "b_1_prime"): (5, INF, "in-vehicle"),
        ("b_1_double", "d_1_double"): (5, INF, "in-vehicle"),
        ("d_1_prime", "e_1_prime"): (5, 1, "in-vehicle"),
        ("e_1_double", "h_1_double"): (5, INF, "in-vehicle"),
        ("b_1_prime", "c_1"): (5, INF, "in-vehicle"),
        ("c_1", "d_1_prime"): (10, INF, "in-vehicle"),
        ("e_1_triple", "f_1"): (5, 1, "in-vehicle"),
        ("f_1", "h_1_prime"): (5, INF, "in-vehicle"),
        ("b_1_triple", "e_1_triple"): (35, INF, "in-vehicle"),
        ("b_1_quadruple", "h_1_triple"): (50, INF, "in-vehicle"),
    }
    return _build(
        name="figure11_mixed_interacting_no_ue",
        figure="11",
        routes=routes,
        demand={("a_g", "h_g"): 2},
        edge_data=edge_data,
        conflicts=[
            ("p2", "p1", ("d_1_prime", "e_1_prime"), 0),
            ("p3", "p2", ("e_1_triple", "f_1"), 1),
        ],
        expected={
            "path_costs": {"p1": 20, "p2": 35, "p3": 50, "p4": 55},
            "so_cost": 70,
            "ue_exists": False,
            "ue_cost": None,
            "ue_flows": [],
            "que_cost": 70,
        },
    )


FIGURE_BUILDERS = (
    figure4_ue_neq_so,
    figure6a_pure_boarding_symmetric,
    figure6b_pure_boarding_asymmetric,
    figure7_pure_boarding_four_path,
    figure8_pure_boarding_interacting,
    figure9_mixed_ue_exists,
    figure10_mixed_no_ue,
    figure11_mixed_interacting_no_ue,
)
