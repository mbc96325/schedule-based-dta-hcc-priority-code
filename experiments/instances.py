"""
Instance builders for the experiments.

* :func:`nguyen_full_instance` reproduces the complete four-OD illustrative
  example in Nguyen et al. (2001).
* :func:`nguyen_instance` retains the two-OD, horizon-extended restriction used
  by the demand-capacity sensitivity experiment.
* The ``canonical_*`` builders encode the small theory instances drawn in the
  paper figures, used to validate the structural claims in Experiment 1.

Every builder returns a :class:`dta.TransitInstance`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import _bootstrap  # noqa: F401  (sys.path side effect)
from dta import TransitInstance, Edge
from dta.network import BoardingConflict


def nguyen_full_instance() -> TransitInstance:
    """Complete illustrative example from Nguyen et al. (2001).

    The instance contains the four OD pairs, nine paths, demand vector
    ``(10, 10, 10, 10)``, and vehicle capacity 20 reported with Figure 4 and
    Tables 1--3 of the source article. Path costs are the fixed travel-cost
    terms ``l_p`` in Table 3. Boarding conflicts encode the two FIFB events at
    the shared vehicle links ``(8, 9)`` and ``(10, 13)``.
    """
    routes = {
        "p1": (1, 5, 10, 13, 3),
        "p2": (1, 8, 9, 11, 10, 13, 3),
        "p3": (1, 8, 9, 12, 14, 3),
        "p4": (1, 5, 10, 13, 15, 4),
        "p5": (1, 8, 9, 11, 10, 13, 15, 4),
        "p6": (2, 6, 8, 9, 11, 10, 13, 3),
        "p7": (2, 6, 8, 9, 12, 14, 3),
        "p8": (2, 6, 8, 9, 11, 10, 13, 15, 4),
        "p9": (2, 7, 16, 4),
    }
    segment_cost = {
        (1, 5): 15,
        (1, 8): 0,
        (2, 6): 0,
        (2, 7): 15,
        (5, 10): 30,
        (6, 8): 5,
        (7, 16): 60,
        (8, 9): 5,
        (9, 11): 5,
        (9, 12): 5,
        (10, 13): 10,
        (11, 10): 5,
        (12, 14): 20,
        (13, 3): 0,
        (13, 15): 15,
        (14, 3): 0,
        (15, 4): 0,
        (16, 4): 0,
    }
    segment_type = {
        (1, 5): "boarding",
        (1, 8): "boarding",
        (2, 6): "boarding",
        (2, 7): "boarding",
        (5, 10): "in-vehicle",
        (6, 8): "in-vehicle",
        (7, 16): "in-vehicle",
        (8, 9): "in-vehicle",
        (9, 11): "in-vehicle",
        (9, 12): "boarding",
        (10, 13): "in-vehicle",
        (11, 10): "boarding",
        (12, 14): "in-vehicle",
        (13, 3): "alighting",
        (13, 15): "in-vehicle",
        (14, 3): "alighting",
        (15, 4): "alighting",
        (16, 4): "alighting",
    }
    edges = [
        Edge(
            u,
            v,
            cost=cost,
            capacity=20 if segment_type[(u, v)] == "in-vehicle" else float("inf"),
            type=segment_type[(u, v)],
        )
        for (u, v), cost in segment_cost.items()
    ]

    conflicts = []
    for higher in ("p6", "p7", "p8"):
        for lower in ("p2", "p3", "p5"):
            conflicts.append(
                BoardingConflict(higher, lower, (8, 9), 8 * 60 + 20)
            )
    for higher in ("p1", "p4"):
        for lower in ("p2", "p5", "p6", "p8"):
            conflicts.append(
                BoardingConflict(higher, lower, (10, 13), 8 * 60 + 35)
            )

    return TransitInstance(
        name="nguyen_full",
        edges=edges,
        demand={(1, 3): 10, (1, 4): 10, (2, 3): 10, (2, 4): 10},
        boarding_conflicts=conflicts,
        explicit_paths=list(routes.values()),
        explicit_path_labels=list(routes.keys()),
        metadata={
            "published_path_costs": {
                "p1": 55,
                "p2": 25,
                "p3": 30,
                "p4": 70,
                "p5": 40,
                "p6": 30,
                "p7": 35,
                "p8": 45,
                "p9": 75,
            },
            "published_equilibrium_flow": {
                "p1": 5,
                "p2": 0,
                "p3": 5,
                "p4": 10,
                "p5": 0,
                "p6": 0,
                "p7": 10,
                "p8": 5,
                "p9": 5,
            },
            "published_penalty_equilibrium_flow": {
                "p1": 5.236,
                "p2": 0,
                "p3": 4.764,
                "p4": 10,
                "p5": 0,
                "p6": 0,
                "p7": 10,
                "p8": 6.236,
                "p9": 3.764,
            },
            "published_system_optimum_flow": {
                "p1": 0,
                "p2": 10,
                "p3": 0,
                "p4": 10,
                "p5": 0,
                "p6": 0,
                "p7": 10,
                "p8": 0,
                "p9": 10,
            },
        },
    )


def nguyen_instance() -> TransitInstance:
    """The Nguyen-style example from ``DTA_Nguyen_Comparison.py``.

    Routes, costs, segment types, the capacitated link (10 -> 13, capacity 13)
    and the demand are copied verbatim from the prototype. Node times come from
    the prototype timetable (converted to minutes), and the boarding-priority
    contest is the explicit ``(5,10)`` over ``(11,10)`` scenario. The original
    prototype has a finite horizon; we extend the OD ``(1,4)`` service by one
    additional headway so every passenger can eventually be assigned under QUE.
    """
    segment_cost = {
        (1, 5): 15, (1, 8): 0, (2, 6): 0, (2, 7): 15,
        (5, 10): 30, (6, 8): 5, (7, 16): 60, (8, 9): 5,
        (9, 11): 5, (9, 12): 5, (10, 13): 10, (11, 10): 5,
        (12, 14): 20, (13, 3): 0, (13, 15): 15, (14, 3): 0,
        (15, 4): 0, (16, 4): 0,
        # One additional departure for OD (1,4), one 30-minute headway after
        # the route-1 service in the prototype. The extra generalized cost
        # comes from the later boarding/waiting link, not from an artificial
        # penalty on the route itself.
        (1, 17): 45, (17, 18): 30, (18, 19): 10, (19, 20): 15, (20, 4): 0,
    }
    segment_type = {
        (1, 5): "boarding", (1, 8): "boarding", (2, 6): "boarding",
        (2, 7): "boarding", (5, 10): "in-vehicle", (6, 8): "in-vehicle",
        (7, 16): "in-vehicle", (8, 9): "in-vehicle", (9, 11): "in-vehicle",
        (9, 12): "boarding", (10, 13): "in-vehicle", (11, 10): "boarding",
        (12, 14): "in-vehicle", (13, 3): "alighting", (13, 15): "in-vehicle",
        (14, 3): "alighting", (15, 4): "alighting", (16, 4): "alighting",
        (1, 17): "boarding", (17, 18): "in-vehicle", (18, 19): "in-vehicle",
        (19, 20): "in-vehicle", (20, 4): "alighting",
    }
    # Capacity 13 on the contested link 10 -> 13; 20 elsewhere (prototype values).
    edges = []
    for e, cost in segment_cost.items():
        cap = 13 if e == (10, 13) else 20
        edges.append(Edge(e[0], e[1], cost=cost, capacity=cap, type=segment_type[e]))

    demand = {(1, 3): 10, (1, 4): 10}

    timetable_hm = {
        5: (8, 5), 6: (8, 15), 7: (8, 0), 8: (8, 20), 9: (8, 25), 10: (8, 35),
        11: (8, 30), 12: (8, 30), 13: (8, 45), 14: (8, 50), 15: (9, 0), 16: (9, 0),
        17: (8, 35), 18: (9, 5), 19: (9, 15), 20: (9, 30),
    }
    node_time = {n: h * 60 + m for n, (h, m) in timetable_hm.items()}

    # Explicit boarding contest on the shared in-vehicle link into node 10:
    # passengers arriving via (5,10) board before those arriving via (11,10).
    boarding_scenarios = [((5, 10), (11, 10))]

    return TransitInstance(
        name="nguyen",
        edges=edges,
        demand=demand,
        node_time=node_time,
        boarding_scenarios=boarding_scenarios,
    )


# --------------------------------------------------------------------------- #
# Canonical theory instances (Experiment 1).
#
# Each is a tiny hand-built network designed to exhibit one structural feature.
# Costs are abstract generalized-cost units; auxiliary access/egress edges have
# infinite capacity so only the labelled in-vehicle links bind.
# --------------------------------------------------------------------------- #


def canonical_ue_neq_so() -> TransitInstance:
    """Figure 4: UE != SO with the auxiliary nodes shown in the paper.

    The visible costs in Figure 4 give path costs p1=15, p2=16, p3=5, p4=10.
    Demand is one unit for each passenger group and the contested in-vehicle
    link h1 -> f1' has capacity one. Boarding priority p1 over p3 makes the
    UE/QUE choose p1 and p4, while SO chooses p2 and p3.
    """
    edges = [
        # group g, path p1 (red)
        ("a_g", "a_tilde_1", 0, float("inf"), "demand"),
        ("a_tilde_1", "a_1", 0, float("inf"), "boarding"),
        ("a_1", "b_1_prime", 5, float("inf"), "in-vehicle"),
        ("b_1_prime", "h_1", 5, float("inf"), "in-vehicle"),
        ("h_1", "f_1_prime", 5, 1, "in-vehicle"),
        ("f_1_prime", "f_g", 0, float("inf"), "alighting"),
        # group g, path p2 (blue)
        ("b_1_prime", "b_1_tilde_double", 1, float("inf"), "transfer"),
        ("b_1_tilde_double", "b_1_double", 0, float("inf"), "boarding"),
        ("b_1_double", "e_1", 5, float("inf"), "in-vehicle"),
        ("e_1", "f_1_double", 5, float("inf"), "in-vehicle"),
        ("f_1_double", "f_g", 0, float("inf"), "alighting"),
        # group g', path p3 (yellow)
        ("h_gp", "h_tilde_1", 0, float("inf"), "demand"),
        ("h_tilde_1", "h_1", 0, float("inf"), "boarding"),
        ("f_1_prime", "f_gp", 0, float("inf"), "alighting"),
        # group g', path p4 (green)
        ("h_tilde_1", "h_2", 5, float("inf"), "boarding"),
        ("h_2", "f_2_prime", 5, float("inf"), "in-vehicle"),
        ("f_2_prime", "f_gp", 0, float("inf"), "alighting"),
    ]
    node_time = {
        "b_1_prime": 0,
        "h_tilde_1": 1,
    }
    demand = {("a_g", "f_g"): 1, ("h_gp", "f_gp"): 1}
    boarding_scenarios = [(("b_1_prime", "h_1"), ("h_tilde_1", "h_1"))]
    return TransitInstance(
        "figure4_ue_neq_so", edges, demand, node_time,
        boarding_scenarios=boarding_scenarios,
    )


def figure6_mixed_loop_no_ue() -> TransitInstance:
    """Figure 6: mixed cost/boarding loop where classical UE does not exist.

    The graph uses the auxiliary nodes in Figure 6. The shared bottleneck
    d1' -> e1' has capacity two, total demand is three, and p2 has boarding
    priority over p1 on that bottleneck. This creates the mixed loop
    p1 -> p2 by cost priority and p2 -> p1 by boarding priority.
    """
    edges = [
        ("a_g", "a_tilde_1", 0, float("inf"), "demand"),
        ("a_tilde_1", "a_1", 0, float("inf"), "boarding"),
        ("a_1", "b_1_prime", 5, float("inf"), "in-vehicle"),
        # p1 (blue)
        ("b_1_prime", "b_1_tilde_double", 1, float("inf"), "transfer"),
        ("b_1_tilde_double", "b_1_double", 0, float("inf"), "boarding"),
        ("b_1_double", "d_1_double", 5, float("inf"), "in-vehicle"),
        ("d_1_double", "d_1_tilde_prime", 0, float("inf"), "transfer"),
        ("d_1_tilde_prime", "d_1_prime", 0, float("inf"), "boarding"),
        ("d_1_prime", "e_1_prime", 5, 2, "in-vehicle"),
        ("e_1_prime", "e_g", 0, float("inf"), "alighting"),
        # p2 (red)
        ("b_1_prime", "c_1", 5, float("inf"), "in-vehicle"),
        ("c_1", "d_1_prime", 10, float("inf"), "in-vehicle"),
        # p3 (yellow)
        ("b_1_prime", "b_1_tilde_triple", 1, float("inf"), "transfer"),
        ("b_1_tilde_triple", "b_1_triple", 0, float("inf"), "boarding"),
        ("b_1_triple", "e_1_triple", 30, float("inf"), "in-vehicle"),
        ("e_1_triple", "e_g", 0, float("inf"), "alighting"),
    ]
    demand = {("a_g", "e_g"): 3}
    boarding_scenarios = [(("c_1", "d_1_prime"), ("d_1_tilde_prime", "d_1_prime"))]
    return TransitInstance(
        "figure6_mixed_loop_no_ue", edges, demand,
        boarding_scenarios=boarding_scenarios,
    )


def figure7_mixed_loop_ue_exists() -> TransitInstance:
    """Figure 7: mixed cost/boarding loop where a classical UE exists.

    The topology is the same as Figure 6, but the lower and middle alternatives
    have the modified costs shown in Figure 7. The resulting loop is
    p1 -> p2 -> p3 by cost priority and p3 -> p1 by boarding priority.
    """
    edges = [
        ("a_g", "a_tilde_1", 0, float("inf"), "demand"),
        ("a_tilde_1", "a_1", 0, float("inf"), "boarding"),
        ("a_1", "b_1_prime", 5, float("inf"), "in-vehicle"),
        # p1 (blue)
        ("b_1_prime", "b_1_tilde_double", 1, float("inf"), "transfer"),
        ("b_1_tilde_double", "b_1_double", 0, float("inf"), "boarding"),
        ("b_1_double", "d_1_double", 5, float("inf"), "in-vehicle"),
        ("d_1_double", "d_1_tilde_prime", 0, float("inf"), "transfer"),
        ("d_1_tilde_prime", "d_1_prime", 0, float("inf"), "boarding"),
        ("d_1_prime", "e_1_prime", 5, 2, "in-vehicle"),
        ("e_1_prime", "e_g", 0, float("inf"), "alighting"),
        # p3 in the figure (yellow middle path)
        ("b_1_prime", "c_1", 5, float("inf"), "in-vehicle"),
        ("c_1", "d_1_prime", 20, float("inf"), "in-vehicle"),
        # p2 in the figure (red lower path)
        ("b_1_prime", "b_1_tilde_triple", 1, float("inf"), "transfer"),
        ("b_1_tilde_triple", "b_1_triple", 0, float("inf"), "boarding"),
        ("b_1_triple", "e_1_triple", 15, float("inf"), "in-vehicle"),
        ("e_1_triple", "e_g", 0, float("inf"), "alighting"),
    ]
    demand = {("a_g", "e_g"): 3}
    boarding_scenarios = [(("c_1", "d_1_prime"), ("d_1_tilde_prime", "d_1_prime"))]
    return TransitInstance(
        "figure7_mixed_loop_ue_exists", edges, demand,
        boarding_scenarios=boarding_scenarios,
    )


def canonical_boarding_cycle() -> TransitInstance:
    """Pure boarding-priority cycle, broken by the timely-last rule.

    Two passenger groups share two in-vehicle links in sequence. On the first
    link group A boards earlier (A outranks B); on the second link group B
    boards earlier (B outranks A). Their boarding-priority relations therefore
    form a 2-cycle with no cost-priority edge involved. The priority graph is
    cyclic, so a classical UE ordering need not exist, but breaking the
    timely-last claim yields a QUE (experiment-plan claims 5 and 6).
    """
    # Both groups traverse the *same* two in-vehicle edges m1->m2 (L1) and
    # n1->n2 (L2), entering each from their own boarding node so the per-edge
    # boarding-priority rule sees them as competing.
    # Capacity 12 = total demand, so all riders are served and SO is feasible;
    # the instance isolates the cycle-breaking behaviour (the Nguyen example
    # separately exercises capacity-driven denied boardings).
    edges = [
        ("m1", "m2", 1, 12, "in-vehicle"),   # shared link L1 (cap 12)
        ("n1", "n2", 1, 12, "in-vehicle"),   # shared link L2 (cap 12)
        # group A: OA -> aIn -> m1 -> m2 -> aMid -> n1 -> n2 -> DA
        ("OA", "aIn", 0, float("inf"), "boarding"),
        ("aIn", "m1", 0, float("inf"), "boarding"),
        ("m2", "aMid", 0, float("inf"), "boarding"),
        ("aMid", "n1", 0, float("inf"), "boarding"),
        ("n2", "DA", 0, float("inf"), "alighting"),
        # group B: OB -> bIn -> m1 -> m2 -> bMid -> n1 -> n2 -> DB
        ("OB", "bIn", 0, float("inf"), "boarding"),
        ("bIn", "m1", 0, float("inf"), "boarding"),
        ("m2", "bMid", 0, float("inf"), "boarding"),
        ("bMid", "n1", 0, float("inf"), "boarding"),
        ("n2", "DB", 0, float("inf"), "alighting"),
    ]
    # On L1 (boarding nodes aIn/bIn) A is earlier; on L2 (aMid/bMid) B is earlier.
    node_time = {"aIn": 0, "bIn": 1, "bMid": 2, "aMid": 3}
    demand = {("OA", "DA"): 6, ("OB", "DB"): 6}
    return TransitInstance("canonical_boarding_cycle", edges, demand, node_time)


def _pure_boarding_conflict_instance(name: str, n_groups: int, conflicts) -> TransitInstance:
    """Build paper pure-boarding-loop examples from pairwise bottlenecks.

    Each conflict tuple is ``(higher_path, lower_path, label, capacity,
    board_time)`` and represents a shared in-vehicle bottleneck used by both
    paths, with the first path having boarding priority over the second. The
    paper states that every group has a second higher-cost route but omits those
    routes from Figures 8--10, so this helper adds one uncapacitated fallback
    route per group with cost one unit above that group's shown path.
    """
    edges = []
    node_time = {}
    boarding_scenarios = []
    path_steps = {p: [] for p in range(1, n_groups + 1)}

    for high, low, label, capacity, board_time in conflicts:
        u, v = f"{label}_u", f"{label}_v"
        high_in = f"{label}_p{high}_in"
        low_in = f"{label}_p{low}_in"
        edges.extend([
            Edge(high_in, u, 0, float("inf"), "boarding"),
            Edge(low_in, u, 0, float("inf"), "boarding"),
            Edge(u, v, 1, capacity, "in-vehicle"),
        ])
        node_time[high_in] = board_time
        node_time[low_in] = board_time + 0.1
        boarding_scenarios.append(((high_in, u), (low_in, u)))
        path_steps[high].append((high_in, u, v))
        path_steps[low].append((low_in, u, v))

    for p in range(1, n_groups + 1):
        origin, dest = f"o_g{p}", f"d_g{p}"
        prev = origin
        shown_path = [origin]
        for step, (in_node, _u, v) in enumerate(path_steps[p]):
            etype = "demand" if step == 0 else "transfer"
            edges.append(Edge(prev, in_node, 0, float("inf"), etype))
            shown_path.extend([in_node, _u, v])
            prev = v
        edges.append(Edge(prev, dest, 0, float("inf"), "alighting"))
        shown_path.append(dest)

        # Omitted second route: strictly more expensive than the shown path.
        alt = f"alt_g{p}"
        fallback_cost = len(path_steps[p]) + 1
        edges.extend([
            Edge(origin, alt, fallback_cost, float("inf"), "boarding"),
            Edge(alt, dest, 0, float("inf"), "alighting"),
        ])
        path_steps[p] = [shown_path, [origin, alt, dest]]

    demand = {(f"o_g{p}", f"d_g{p}"): 1 for p in range(1, n_groups + 1)}
    explicit_paths = [
        route
        for p in range(1, n_groups + 1)
        for route in path_steps[p]
    ]
    return TransitInstance(
        name, edges, demand, node_time, boarding_scenarios,
        explicit_paths=explicit_paths,
    )


def figure8_odd_boarding_loop() -> TransitInstance:
    """Figure 8: three-path pure boarding loop with a unique UE."""
    return _pure_boarding_conflict_instance(
        "figure8_odd_boarding_loop",
        3,
        [
            (1, 2, "fig8_b1_c1", 1, 0),  # p1 > p2
            (2, 3, "fig8_k1_j1", 1, 1),  # p2 > p3
            (3, 1, "fig8_e1_i1", 1, 2),  # p3 > p1
        ],
    )


def figure9_even_boarding_loop() -> TransitInstance:
    """Figure 9: four-path pure boarding loop with multiple UE flows."""
    return _pure_boarding_conflict_instance(
        "figure9_even_boarding_loop",
        4,
        [
            (1, 2, "fig9_b1_c1", 1, 0),  # p1 > p2
            (2, 3, "fig9_k1_j1", 1, 1),  # p2 > p3
            (3, 4, "fig9_h1_i1", 1, 2),  # p3 > p4
            (4, 1, "fig9_e1_m1", 1, 3),  # p4 > p1
        ],
    )


def figure10_overlapping_boarding_loops() -> TransitInstance:
    """Figure 10: overlapping pure boarding loops sharing path p3."""
    return _pure_boarding_conflict_instance(
        "figure10_overlapping_boarding_loops",
        5,
        [
            (1, 2, "fig10_b1_c1", 1, 0),     # p1 > p2
            (2, 3, "fig10_f1_j1", 1, 1),     # p2 > p3
            (3, 1, "fig10_e1_i1", 1, 2),     # p3 > p1
            (3, 4, "fig10_h1_k1", 0.5, 3),   # p3 > p4
            (4, 5, "fig10_l1_m1", 0.5, 4),   # p4 > p5
            (5, 3, "fig10_k1_n1", 0.5, 5),   # p5 > p3
        ],
    )


def canonical_acyclic() -> TransitInstance:
    """Acyclic priority graph: UE exists and equals the greedy QUE outcome."""
    edges = [
        ("O", "A", 1, 6, "in-vehicle"),
        ("A", "D", 0, float("inf"), "alighting"),
        ("O", "B", 4, 100, "in-vehicle"),
        ("B", "D", 0, float("inf"), "alighting"),
    ]
    demand = {("O", "D"): 10}
    return TransitInstance("canonical_acyclic", edges, demand)


# --------------------------------------------------------------------------- #
# Parameterized builders for Experiment 2 (demand-capacity phase diagrams).
#
# Each takes a demand multiplier ``alpha`` and a swept capacity ``U`` on the
# single binding in-vehicle link; all other in-vehicle links are made
# effectively uncapacitated so the phase behaviour is driven by ``(alpha, U)``
# alone. The priority-graph *structure* (costs, boarding times) is independent
# of ``alpha`` and ``U``; only the flows, gap and left-behind change.
# --------------------------------------------------------------------------- #

BIG = 1e9  # effectively uncapacitated


def ue_neq_so_param(alpha: float = 1.0, U: float = 10.0) -> TransitInstance:
    """``canonical_ue_neq_so`` with demand ``10*alpha`` per group and the shared
    scarce link capacity set to ``U``."""
    d = 10.0 * alpha
    edges = [
        ("OA", "S1", 1, BIG, "boarding"),
        ("OB", "S1", 1, BIG, "boarding"),
        ("S1", "S2", 0, U, "in-vehicle"),
        ("S2", "DA", 0, BIG, "alighting"),
        ("S2", "DB", 0, BIG, "alighting"),
        ("OA", "RA", 2, BIG, "boarding"),
        ("RA", "DA", 0, BIG, "in-vehicle"),
        ("OB", "TB", 10, BIG, "boarding"),
        ("TB", "DB", 0, BIG, "in-vehicle"),
    ]
    node_time = {"OA": 0, "OB": 1}
    demand = {("OA", "DA"): d, ("OB", "DB"): d}
    return TransitInstance(f"ue_neq_so(a={alpha:g},U={U:g})", edges, demand, node_time)


def nguyen_param(alpha: float = 1.0, U: float = 13.0) -> TransitInstance:
    """Nguyen network with both demands scaled by ``alpha`` and the contested
    link ``10 -> 13`` capacity set to ``U`` (all other links uncapacitated)."""
    base = nguyen_instance()
    edges = []
    for e in base.edges:  # base.edges is a list of Edge after __post_init__
        if e.as_tuple() == (10, 13):
            cap = U
        elif e.type == "in-vehicle":
            cap = BIG
        else:
            cap = e.capacity
        edges.append(Edge(e.u, e.v, e.cost, cap, e.type))
    demand = {od: v * alpha for od, v in base.demand.items()}
    return TransitInstance(
        f"nguyen(a={alpha:g},U={U:g})", edges, demand, base.node_time,
        boarding_scenarios=base.boarding_scenarios,
    )


def boarding_cycle_param(alpha: float = 1.0, U: float = 12.0) -> TransitInstance:
    """``canonical_boarding_cycle`` with demand ``6*alpha`` per group and both
    shared links' capacity set to ``U``."""
    d = 6.0 * alpha
    edges = [
        ("m1", "m2", 1, U, "in-vehicle"),
        ("n1", "n2", 1, U, "in-vehicle"),
        ("OA", "aIn", 0, BIG, "boarding"),
        ("aIn", "m1", 0, BIG, "boarding"),
        ("m2", "aMid", 0, BIG, "boarding"),
        ("aMid", "n1", 0, BIG, "boarding"),
        ("n2", "DA", 0, BIG, "alighting"),
        ("OB", "bIn", 0, BIG, "boarding"),
        ("bIn", "m1", 0, BIG, "boarding"),
        ("m2", "bMid", 0, BIG, "boarding"),
        ("bMid", "n1", 0, BIG, "boarding"),
        ("n2", "DB", 0, BIG, "alighting"),
    ]
    node_time = {"aIn": 0, "bIn": 1, "bMid": 2, "aMid": 3}
    demand = {("OA", "DA"): d, ("OB", "DB"): d}
    return TransitInstance(f"boarding_cycle(a={alpha:g},U={U:g})", edges, demand, node_time)


# Registry used by the capacity-sweep runner: (name, builder, demand-axis label).
PHASE_NETWORKS = {
    "ue_neq_so": ue_neq_so_param,
    "nguyen": nguyen_param,
    "boarding_cycle": boarding_cycle_param,
}
