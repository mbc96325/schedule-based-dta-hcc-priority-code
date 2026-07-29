"""Controlled instances for the demand-capacity mechanism experiment."""

from __future__ import annotations

import _bootstrap  # noqa: F401
from dta import BoardingConflict, Edge, TransitInstance

from figure_instances import figure10_mixed_no_ue


INF = float("inf")


def mixed_component_instance(
    demand: float,
    capacity: float,
) -> TransitInstance:
    """Figure 10 with variable demand and contested-segment capacity."""
    base = figure10_mixed_no_ue(demand_volume=demand)
    contested_edge = ("d_1_prime", "e_1_prime")
    edges = [
        Edge(
            edge.u,
            edge.v,
            edge.cost,
            capacity if edge.as_tuple() == contested_edge else edge.capacity,
            edge.type,
        )
        for edge in base.edges
    ]
    return TransitInstance(
        name=f"mixed_component_q{demand:g}_U{capacity:g}",
        edges=edges,
        demand={("a_g", "e_g"): float(demand)},
        node_time=dict(base.node_time),
        boarding_conflicts=list(base.boarding_conflicts or []),
        explicit_paths=list(base.explicit_paths or []),
        explicit_path_labels=list(base.explicit_path_labels or []),
        metadata={
            "family": "mixed_component",
            "demand": float(demand),
            "capacity": float(capacity),
            "relation": (
                "p1 cost-priority p2 cost-priority p3; "
                "p2 boarding-priority p1"
            ),
        },
    )


def priority_bottleneck_instance(
    demand_per_group: float,
    capacity: float,
    high_priority_delay: float,
    low_priority_delay: float,
) -> TransitInstance:
    """Two groups competing for one departure with different fallback delays.

    Group A boards first. Each group can instead use a later service path whose
    generalized cost is its fallback delay.
    """
    routes = {
        "p_A_early": ("O_A", "b_A", "s", "t", "D_A"),
        "p_A_late": ("O_A", "late_A", "D_A"),
        "p_B_early": ("O_B", "b_B", "s", "t", "D_B"),
        "p_B_late": ("O_B", "late_B", "D_B"),
    }
    edges = [
        Edge("O_A", "b_A", 0, INF, "demand"),
        Edge("b_A", "s", 0, INF, "boarding"),
        Edge("O_B", "b_B", 0, INF, "demand"),
        Edge("b_B", "s", 0, INF, "boarding"),
        Edge("s", "t", 0, float(capacity), "in-vehicle"),
        Edge("t", "D_A", 0, INF, "exit"),
        Edge("t", "D_B", 0, INF, "exit"),
        Edge(
            "O_A",
            "late_A",
            float(high_priority_delay),
            INF,
            "boarding",
        ),
        Edge("late_A", "D_A", 0, INF, "exit"),
        Edge(
            "O_B",
            "late_B",
            float(low_priority_delay),
            INF,
            "boarding",
        ),
        Edge("late_B", "D_B", 0, INF, "exit"),
    ]
    return TransitInstance(
        name=(
            f"priority_bottleneck_q{demand_per_group:g}_U{capacity:g}"
            f"_hA{high_priority_delay:g}_hB{low_priority_delay:g}"
        ),
        edges=edges,
        demand={
            ("O_A", "D_A"): float(demand_per_group),
            ("O_B", "D_B"): float(demand_per_group),
        },
        boarding_conflicts=[
            BoardingConflict(
                "p_A_early",
                "p_B_early",
                ("s", "t"),
                0,
            )
        ],
        explicit_paths=list(routes.values()),
        explicit_path_labels=list(routes.keys()),
        metadata={
            "family": "priority_bottleneck",
            "demand_per_group": float(demand_per_group),
            "capacity": float(capacity),
            "high_priority_delay": float(high_priority_delay),
            "low_priority_delay": float(low_priority_delay),
        },
    )
