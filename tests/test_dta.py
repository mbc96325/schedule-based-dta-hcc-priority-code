"""Regression tests for the exact manuscript Figure 4 and 6--11 instances."""

from __future__ import annotations

import os
import sys

import networkx as nx

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "experiments"))

from dta import (  # noqa: E402
    AssignmentResult,
    BoardingConflict,
    Edge,
    TransitInstance,
    classical_ue_assignment,
    componentwise_quasi_ue_assignment,
    diagnostics,
    enumerate_paths,
    greedy_lexicographic_assignment,
    quasi_ue_assignment,
    que_order,
    sparse_system_optimum_assignment,
    system_optimum_assignment,
)
from dta.priority_graph import (  # noqa: E402
    BOARDING_PRIORITY,
    COST_PRIORITY,
    build_compact_priority_graph,
    build_priority_graph,
    cost_compatible_projection_order,
    relation_types,
)
from figure_instances import (  # noqa: E402
    FIGURE_BUILDERS,
    figure10_mixed_no_ue,
    figure6a_pure_boarding_symmetric,
)
from instances import nguyen_full_instance  # noqa: E402
from mechanism_instances import (  # noqa: E402
    mixed_component_instance,
    priority_bottleneck_instance,
)

TOL = 1e-7


def _labeled_flow(pathset, result):
    return {pathset.label[p]: value for p, value in result.flow.items()}


def _assert_flow(actual, expected):
    assert set(actual) == set(expected)
    for label, expected_value in expected.items():
        assert abs(actual[label] - expected_value) <= TOL, (
            label,
            actual[label],
            expected_value,
        )


def _candidate_result(instance, pathset, labeled_flow):
    flow = {
        p: float(labeled_flow.get(pathset.label[p], 0.0))
        for p in pathset.indices()
    }
    edge_flow = {
        edge: sum(flow[p] for p in paths)
        for edge, paths in pathset.edge_paths.items()
    }
    return AssignmentResult(
        name=instance.name,
        method="claimed_ue",
        feasible=True,
        total_cost=sum(flow[p] * pathset.cost[p] for p in pathset.indices()),
        flow=flow,
        edge_flow=edge_flow,
    )


def test_figure_path_costs_and_finite_capacities():
    for build in FIGURE_BUILDERS:
        instance = build()
        pathset = enumerate_paths(instance)
        expected = instance.metadata["expected"]
        actual_costs = {
            pathset.label[p]: pathset.cost[p] for p in pathset.indices()
        }
        _assert_flow(actual_costs, expected["path_costs"])

        actual_finite = {
            edge: instance.edge_capacity(*edge)
            for edge in instance.G.edges
            if instance.edge_capacity(*edge) != float("inf")
        }
        assert actual_finite == instance.metadata["finite_capacities"]


def test_figure_boarding_relations_use_the_shown_contested_edges():
    for build in FIGURE_BUILDERS:
        instance = build()
        pathset = enumerate_paths(instance)
        graph = build_priority_graph(instance, pathset)
        expected_relations = {
            (higher, lower, edge)
            for higher, lower, edge in instance.metadata["boarding_relations"]
        }
        actual_relations = set()
        for u, v in graph.edges:
            if BOARDING_PRIORITY not in relation_types(graph, u, v):
                continue
            for conflict in graph.edges[u, v].get("boarding_conflicts", ()):
                actual_relations.add(
                    (
                        pathset.label[u],
                        pathset.label[v],
                        conflict.contested_edge,
                    )
                )
        assert actual_relations == expected_relations


def test_relation_types_are_not_overwritten():
    routes = {
        "p1": ("o", "x", "d"),
        "p2": ("o", "y", "x", "d"),
    }
    instance = TransitInstance(
        name="dual_relation",
        edges=[
            Edge("o", "x", 0, float("inf"), "boarding"),
            Edge("o", "y", 1, float("inf"), "boarding"),
            Edge("y", "x", 0, float("inf"), "boarding"),
            Edge("x", "d", 1, 1, "in-vehicle"),
        ],
        demand={("o", "d"): 1},
        boarding_conflicts=[
            BoardingConflict("p1", "p2", ("x", "d"), 0)
        ],
        explicit_paths=list(routes.values()),
        explicit_path_labels=list(routes.keys()),
    )
    pathset = enumerate_paths(instance)
    graph = build_priority_graph(instance, pathset)
    both = relation_types(graph, pathset.index("p1"), pathset.index("p2"))
    assert both == {COST_PRIORITY, BOARDING_PRIORITY}


def test_so_ue_and_que_match_figure_claims():
    for build in FIGURE_BUILDERS:
        instance = build()
        pathset = enumerate_paths(instance)
        expected = instance.metadata["expected"]

        so = system_optimum_assignment(instance, pathset)
        ue = classical_ue_assignment(instance, pathset)
        que = quasi_ue_assignment(instance, pathset)

        assert so.feasible, instance.name
        assert abs(so.total_cost - expected["so_cost"]) <= TOL, instance.name
        assert ue.feasible is expected["ue_exists"], instance.name
        if expected["ue_exists"]:
            assert abs(ue.total_cost - expected["ue_cost"]) <= TOL, instance.name
            assert diagnostics.diagnose(
                instance, pathset, ue
            ).no_improving_switch
        assert que.feasible, instance.name
        assert que.left_behind == 0
        assert abs(que.total_cost - expected["que_cost"]) <= TOL, instance.name
        report = diagnostics.diagnose(instance, pathset, que)
        assert report.demand_conserved
        assert report.capacity_feasible


def test_every_claimed_ue_flow_is_feasible_and_has_no_improvement():
    for build in FIGURE_BUILDERS:
        instance = build()
        pathset = enumerate_paths(instance)
        for labeled_flow in instance.metadata["expected"]["ue_flows"]:
            result = _candidate_result(instance, pathset, labeled_flow)
            report = diagnostics.diagnose(instance, pathset, result)
            assert report.demand_conserved, instance.name
            assert report.capacity_feasible, instance.name
            assert report.no_improving_switch, (
                instance.name,
                report.improving_switches,
            )


def test_figure10_demand_two_has_ue_but_demand_three_does_not():
    low = figure10_mixed_no_ue(demand_volume=2)
    high = figure10_mixed_no_ue(demand_volume=3)
    low_ue = classical_ue_assignment(low, enumerate_paths(low))
    high_ue = classical_ue_assignment(high, enumerate_paths(high))
    assert low_ue.feasible
    assert abs(low_ue.total_cost - 32) <= TOL
    assert not high_ue.feasible
    assert high_ue.details["enumerated_cases"] > 0


def test_cycle_breaking_counts_and_removes_only_boarding_relations():
    expected_removals = {
        "4": 0,
        "6(a)": 1,
        "6(b)": 1,
        "7": 1,
        "8": 2,
        "9": 1,
        "10": 1,
        "11": 2,
    }
    for build in FIGURE_BUILDERS:
        instance = build()
        pathset = enumerate_paths(instance)
        graph = build_priority_graph(instance, pathset)
        que = quasi_ue_assignment(instance, pathset)
        assert len(que.removed_edges) == expected_removals[
            instance.metadata["figure"]
        ]
        assert all(edge.type == BOARDING_PRIORITY for edge in que.removed_edges)
        for removed in que.removed_edges:
            if COST_PRIORITY in relation_types(graph, *removed.edge):
                raise AssertionError("cycle breaker removed a cost relation")


def test_figure_graph_types():
    for build in FIGURE_BUILDERS:
        instance = build()
        pathset = enumerate_paths(instance)
        graph = build_priority_graph(instance, pathset)
        figure = instance.metadata["figure"]
        if figure == "4":
            assert nx.is_directed_acyclic_graph(graph)
        else:
            assert not nx.is_directed_acyclic_graph(graph)


def test_complete_nguyen_instance_matches_published_data():
    instance = nguyen_full_instance()
    pathset = enumerate_paths(instance)
    actual_costs = {
        pathset.label[p]: pathset.cost[p]
        for p in pathset.indices()
    }
    assert actual_costs == instance.metadata["published_path_costs"]
    assert instance.demand == {
        (1, 3): 10,
        (1, 4): 10,
        (2, 3): 10,
        (2, 4): 10,
    }
    assert all(
        instance.edge_capacity(*edge) == 20
        for edge in pathset.edge_paths
        if instance.edge_type(*edge) == "in-vehicle"
    )


def test_complete_nguyen_published_and_recomputed_assignments():
    instance = nguyen_full_instance()
    pathset = enumerate_paths(instance)

    for metadata_key, expected_cost in (
        ("published_equilibrium_flow", 2075),
        ("published_system_optimum_flow", 2050),
    ):
        labeled_flow = instance.metadata[metadata_key]
        candidate = _candidate_result(instance, pathset, labeled_flow)
        report = diagnostics.diagnose(instance, pathset, candidate)
        assert abs(candidate.total_cost - expected_cost) <= TOL
        assert report.demand_conserved
        assert report.capacity_feasible
        assert report.no_improving_switch

    so = system_optimum_assignment(instance, pathset)
    ue = classical_ue_assignment(instance, pathset)
    que = quasi_ue_assignment(instance, pathset)
    assert so.feasible and abs(so.total_cost - 2050) <= TOL
    assert ue.feasible and abs(ue.total_cost - 2050) <= TOL
    assert que.feasible and abs(que.total_cost - 2050) <= TOL
    assert len(que.removed_edges) == 7
    assert diagnostics.diagnose(
        instance, pathset, que
    ).no_improving_switch


def test_que_need_not_be_an_original_ue_when_original_ue_exists():
    instance = figure6a_pure_boarding_symmetric()
    pathset = enumerate_paths(instance)
    ue = classical_ue_assignment(instance, pathset)
    que = quasi_ue_assignment(instance, pathset)
    assert ue.feasible
    report = diagnostics.diagnose(instance, pathset, que)
    assert not report.no_improving_switch
    assert {
        (switch["from_label"], switch["to_label"])
        for switch in report.improving_switches
    } == {("p3_bar", "p3")}


def test_mixed_component_demand_capacity_boundary():
    for demand, capacity, expected_ue in (
        (1.5, 2.0, True),
        (2.0, 2.0, True),
        (2.5, 2.0, False),
    ):
        instance = mixed_component_instance(demand, capacity)
        pathset = enumerate_paths(instance)
        ue = classical_ue_assignment(instance, pathset)
        que = quasi_ue_assignment(instance, pathset)
        assert ue.feasible is expected_ue
        assert que.feasible
        assert diagnostics.diagnose(
            instance, pathset, que
        ).capacity_feasible


def test_priority_bottleneck_cost_gap_and_equilibrium():
    instance = priority_bottleneck_instance(
        demand_per_group=10,
        capacity=10,
        high_priority_delay=5,
        low_priority_delay=20,
    )
    pathset = enumerate_paths(instance)
    so = system_optimum_assignment(instance, pathset)
    ue = classical_ue_assignment(instance, pathset)
    que = quasi_ue_assignment(instance, pathset)
    assert so.feasible and ue.feasible and que.feasible
    assert abs(so.total_cost - 50) <= TOL
    assert abs(ue.total_cost - 200) <= TOL
    assert abs(que.total_cost - ue.total_cost) <= TOL
    assert diagnostics.diagnose(
        instance, pathset, ue
    ).no_improving_switch


def test_componentwise_que_matches_monolithic_que():
    instance = nguyen_full_instance()
    pathset = enumerate_paths(instance)
    monolithic = quasi_ue_assignment(instance, pathset)
    componentwise = componentwise_quasi_ue_assignment(instance, pathset)
    assert monolithic.feasible and componentwise.feasible
    assert abs(monolithic.total_cost - componentwise.total_cost) <= TOL
    for path in pathset.indices():
        assert abs(
            monolithic.flow[path] - componentwise.flow[path]
        ) <= TOL
    report = diagnostics.diagnose(instance, pathset, componentwise)
    assert report.demand_conserved
    assert report.capacity_feasible


def test_sparse_so_and_certified_greedy_loading_match_small_exact_solvers():
    instance = priority_bottleneck_instance(
        demand_per_group=10,
        capacity=10,
        high_priority_delay=5,
        low_priority_delay=20,
    )
    pathset = enumerate_paths(instance)
    dense_so = system_optimum_assignment(instance, pathset)
    sparse_so = sparse_system_optimum_assignment(instance, pathset)
    _, _, removed, order = que_order(instance, pathset)
    exact_que = quasi_ue_assignment(instance, pathset)
    greedy_que = greedy_lexicographic_assignment(
        instance, pathset, order, removed_edges=removed
    )

    assert dense_so.feasible and sparse_so.feasible
    assert abs(dense_so.total_cost - sparse_so.total_cost) <= TOL
    assert exact_que.feasible and greedy_que.feasible
    assert greedy_que.details["full_demand_certificate"]
    for path in pathset.indices():
        assert abs(exact_que.flow[path] - greedy_que.flow[path]) <= TOL


def test_cost_compatible_projection_preserves_cost_priority_and_is_acyclic():
    instance = nguyen_full_instance()
    pathset = enumerate_paths(instance)
    graph = build_compact_priority_graph(instance, pathset)
    order, removed_count, _ = cost_compatible_projection_order(
        graph, removed_sample_limit=1000
    )
    position = {path: rank for rank, path in enumerate(order)}

    retained = nx.DiGraph()
    retained.add_nodes_from(graph.nodes)
    counted_removed = 0
    for u, v in graph.edges:
        relations = relation_types(graph, u, v)
        if COST_PRIORITY in relations:
            assert position[u] < position[v]
        if position[u] < position[v]:
            retained.add_edge(u, v)
        elif BOARDING_PRIORITY in relations:
            counted_removed += 1
    assert counted_removed == removed_count
    assert nx.is_directed_acyclic_graph(retained)


def _run_all():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_")
    ]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
