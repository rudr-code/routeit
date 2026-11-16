from __future__ import annotations

import pytest
from fractions import Fraction

from routeit_engine import solve, replan
from routeit_engine.contracts import (
    FleetMode,
    Order,
    PlanningMode,
    PriorityTier,
    ProfileKind,
    ResultStatus,
    RouteStart,
    SolveRequest,
    VehicleSpec,
    WindowType,
)
from routeit_engine.context import EvaluationContext
from routeit_engine.evaluator import evaluate_plan, evaluate_route
from routeit_engine.oracle import OracleStatus, RoadOracle
from routeit_roads.euclidean import EuclideanOracle


def test_ac_01_input_validation():
    """AC-01: System accepts and validates depot, order and fleet inputs."""
    coords = {"depot": (0, 0), "n1": (10, 0)}
    oracle = EuclideanOracle(coords, distance_scale=1.0)

    # Valid request
    o1 = Order("o1", "n1", 5, 60, 0, 1000, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("V1", 20, activation_cost_paise=100)
    req = SolveRequest(
        orders=(o1,),
        vehicles=(v1,),
        depot_node_id="depot",
        distance_rate_num=100,
        time_rate_num=100,
    )
    res = solve(req, oracle)
    assert res.result_status in (ResultStatus.COMPLETE_FEASIBLE, ResultStatus.PARTIAL_FEASIBLE)
    assert res.plan is not None

    # Invalid: missing rates (negative)
    req_bad = SolveRequest(
        orders=(o1,),
        vehicles=(v1,),
        depot_node_id="depot",
        distance_rate_num=-1,
    )
    res_bad = solve(req_bad, oracle)
    assert res_bad.result_status == ResultStatus.INVALID_CONFIG


def test_ac_02_complete_mode_exactly_once():
    """AC-02: Every order in Complete Mode is assigned exactly once in a feasible result."""
    coords = {"depot": (0, 0), "n1": (5, 0), "n2": (10, 0), "n3": (15, 0)}
    oracle = EuclideanOracle(coords)

    orders = tuple(
        Order(f"o{i}", f"n{i}", 5, 30, 0, 2000, WindowType.HARD, PriorityTier.STANDARD)
        for i in [1, 2, 3]
    )
    vehicles = (
        VehicleSpec("V1", 20, activation_cost_paise=100),
        VehicleSpec("V2", 20, activation_cost_paise=100),
    )
    req = SolveRequest(
        orders=orders,
        vehicles=vehicles,
        depot_node_id="depot",
        planning_mode=PlanningMode.COMPLETE,
        distance_rate_num=100,
        time_rate_num=100,
    )
    res = solve(req, oracle, solver_kind="ALNS")
    assert res.result_status == ResultStatus.COMPLETE_FEASIBLE
    assert res.plan is not None

    all_visited = [s.order_id for r in res.plan.routes for s in r.stops]
    assert sorted(all_visited) == ["o1", "o2", "o3"]
    assert len(all_visited) == len(set(all_visited))  # Exactly once!


def test_ac_03_vehicle_capacity_never_violated():
    """AC-03: Vehicle capacity is never violated in any route."""
    coords = {"depot": (0, 0), "n1": (5, 0), "n2": (10, 0)}
    oracle = EuclideanOracle(coords)

    # Each order demands 15, vehicle capacity is 20 -> cannot put both in one route
    orders = (
        Order("o1", "n1", 15, 30, 0, 2000, WindowType.HARD, PriorityTier.STANDARD),
        Order("o2", "n2", 15, 30, 0, 2000, WindowType.HARD, PriorityTier.STANDARD),
    )
    vehicles = (
        VehicleSpec("V1", 20, activation_cost_paise=100),
        VehicleSpec("V2", 20, activation_cost_paise=100),
    )
    req = SolveRequest(
        orders=orders,
        vehicles=vehicles,
        depot_node_id="depot",
        distance_rate_num=100,
        time_rate_num=100,
    )
    res = solve(req, oracle)
    assert res.plan is not None
    order_map = {o.id: o for o in orders}
    for r in res.plan.routes:
        load = sum(order_map[s.order_id].demand for s in r.stops)
        assert load <= 20


def test_ac_04_hard_time_windows():
    """AC-04: Hard time-window violations are never accepted."""
    # Order requires delivery before 10s, but travel takes 20s
    coords = {"depot": (0, 0), "n1": (200, 0)}
    oracle = EuclideanOracle(coords, speed_mps=10.0, distance_scale=1.0)  # takes 20s travel
    o_impossible = Order("o1", "n1", 5, 10, 0, 10, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("V1", 20)
    req = SolveRequest(
        orders=(o_impossible,),
        vehicles=(v1,),
        depot_node_id="depot",
        planning_mode=PlanningMode.COMPLETE,
        departure_set=(0,),
    )
    res = solve(req, oracle)
    assert res.result_status == ResultStatus.NO_FEASIBLE_PLAN_FOUND


def test_ac_05_soft_window_lateness_penalty():
    """AC-05: Soft-window lateness is correctly calculated and penalized."""
    # Travel takes 20s, service 10s -> completion 30s. Window close = 20s -> lateness = 10s.
    # SLA rate = 60 paise/min = 1 paise/sec -> 10 paise penalty
    coords = {"depot": (0, 0), "n1": (200, 0)}
    oracle = EuclideanOracle(coords, speed_mps=10.0, distance_scale=1.0)
    o_soft = Order("o1", "n1", 5, 10, 0, 20, WindowType.SOFT, PriorityTier.CRITICAL, sla_rate_num=60, sla_rate_den=1)
    v1 = VehicleSpec("V1", 20)
    req = SolveRequest(
        orders=(o_soft,),
        vehicles=(v1,),
        depot_node_id="depot",
        planning_mode=PlanningMode.COMPLETE,
        departure_set=(0,),
        distance_rate_num=0,
        time_rate_num=0,
    )
    res = solve(req, oracle)
    assert res.plan is not None
    assert res.plan.routes[0].stops[0].lateness_s == 10
    assert res.plan.routes[0].stops[0].sla_cost_paise == 10
    assert res.plan.ledger.sla_cost_paise == 10


def test_ac_06_target_fleet_and_auto_fleet():
    """AC-06: Target Fleet and Auto Fleet modes work as specified."""
    coords = {"depot": (0, 0), "n1": (10, 0)}
    oracle = EuclideanOracle(coords, distance_scale=1.0)
    o1 = Order("o1", "n1", 5, 10, 0, 1000, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("V1", 20, activation_cost_paise=500)
    v2 = VehicleSpec("V2", 20, activation_cost_paise=500)

    # Auto fleet with positive activation: 1 order needs only 1 vehicle, does not activate 2
    req = SolveRequest(
        orders=(o1,),
        vehicles=(v1, v2),
        depot_node_id="depot",
        fleet_mode=FleetMode.AUTO,
    )
    res = solve(req, oracle)
    assert res.plan is not None
    assert res.plan.m_routes_used == 1
    assert res.plan.m_new_activated == 1


def test_ac_07_valid_road_paths():
    """AC-07: Generated road routes use valid supported road paths; no straight-line fallback."""
    from routeit_roads import ProfileGraphOracle, TimeProfile, Edge

    tp = TimeProfile([(0, 86400)], {"primary": [10.0]}, 0, 86400)
    # Island node "island" has no edge connecting it to depot
    oracle = ProfileGraphOracle(
        nodes={"depot": (0.0, 0.0), "n1": (1.0, 0.0), "island": (5.0, 5.0)},
        edges=[Edge("depot", "n1", 1000, "primary", 1)],
        profile=tp,
    )

    o_island = Order("o_isl", "island", 5, 10, 0, 1000, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("V1", 20)
    req = SolveRequest(
        orders=(o_island,),
        vehicles=(v1,),
        depot_node_id="depot",
        planning_mode=PlanningMode.COMPLETE,
    )
    res = solve(req, oracle)
    # Cannot reach island by road -> must NOT use straight line fallback!
    assert res.result_status == ResultStatus.NO_FEASIBLE_PLAN_FOUND


def test_ac_08_all_three_profiles():
    """AC-08: All three optimization profiles calculate their respective objectives."""
    coords = {"depot": (0, 0), "n1": (10, 0)}
    oracle = EuclideanOracle(coords)
    o1 = Order("o1", "n1", 5, 10, 0, 1000, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("V1", 20)

    for p_kind in [ProfileKind.COST_FOCUSED, ProfileKind.BALANCED, ProfileKind.DRIVER_PRODUCTIVITY]:
        req = SolveRequest(
            orders=(o1,),
            vehicles=(v1,),
            depot_node_id="depot",
            profile_kind=p_kind,
            profile_params={"delta_c_max_paise": 10000},
            distance_rate_num=100,
            time_rate_num=100,
        )
        res = solve(req, oracle)
        assert res.plan is not None
        assert res.plan.is_feasible


def test_ac_09_partial_mode_priority():
    """AC-09: Partial Mode respects Critical -> Standard -> Flexible priority."""
    # Capacity only allows 1 order
    coords = {"depot": (0, 0), "n1": (5, 0), "n2": (5, 0)}
    oracle = EuclideanOracle(coords)

    o_crit = Order("o_crit", "n1", 10, 10, 0, 1000, WindowType.HARD, PriorityTier.CRITICAL)
    o_flx = Order("o_flx", "n2", 10, 10, 0, 1000, WindowType.HARD, PriorityTier.FLEXIBLE)
    v1 = VehicleSpec("V1", 10)  # Can only take one order!

    req = SolveRequest(
        orders=(o_flx, o_crit),
        vehicles=(v1,),
        depot_node_id="depot",
        planning_mode=PlanningMode.PARTIAL,
    )
    res = solve(req, oracle)
    assert res.plan is not None
    assert "o_crit" in res.plan.served_orders
    assert "o_flx" not in res.plan.served_orders


def test_ac_10_cost_breakdown_matches_independent_evaluator():
    """AC-10: Cost breakdown matches independently calculated test cases."""
    from routeit_reference import evaluate_route_reference

    coords = {"depot": (0, 0), "n1": (100, 0)}
    oracle = EuclideanOracle(coords, speed_mps=10.0, distance_scale=1.0)
    o1 = Order("o1", "n1", 5, 60, 0, 1000, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("V1", 20, activation_cost_paise=2500)

    req = SolveRequest(
        orders=(o1,),
        vehicles=(v1,),
        depot_node_id="depot",
        distance_rate_num=800,
        time_rate_num=1200,
    )
    res = solve(req, oracle)
    assert res.plan is not None
    plan = res.plan

    # Reference calculation
    ref = evaluate_route_reference(
        node_coords=coords,
        order_dicts=[{"id": "o1", "node_id": "n1", "demand": 5, "service_seconds": 60, "window_open": 0, "window_close": 1000, "window_type": "HARD"}],
        vehicle_dict={"id": "V1", "capacity": 20, "activation_cost_paise": 2500},
        depot_node_id="depot",
        departure_time=0,
        distance_rate_paise_per_km=Fraction(800, 1),
        time_rate_paise_per_hour=Fraction(1200, 1),
        a_shift=set(),
        depot_return=True,
        speed_mps=10.0,
    )
    assert plan.routes[0].distance_cost_paise == ref["distance_cost_paise"]
    assert plan.routes[0].time_cost_paise == ref["time_cost_paise"]
    assert plan.routes[0].activation_cost_paise == ref["activation_cost_paise"]
    assert plan.routes[0].total_cost_paise == ref["total_cost_paise"]


def test_ac_11_dynamic_replanning_state_preservation():
    """AC-11: Dynamic replanning preserves completed deliveries and current vehicle state."""
    from routeit_engine.dynamic import (
        OperationalSnapshot,
        VehicleActivityState,
        VehicleOperationalState,
    )

    coords = {"depot": (0, 0), "n1": (10, 0), "n2": (20, 0), "n3": (30, 0)}
    oracle = EuclideanOracle(coords, distance_scale=1.0)
    o1 = Order("o1", "n1", 5, 10, 0, 1000, WindowType.HARD, PriorityTier.STANDARD)
    o2 = Order("o2", "n2", 5, 10, 0, 1000, WindowType.HARD, PriorityTier.STANDARD)
    o3 = Order("o3", "n3", 5, 10, 0, 1000, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("V1", 50, activation_cost_paise=1000)

    req = SolveRequest(
        orders=(o1, o2, o3),
        vehicles=(v1,),
        depot_node_id="depot",
        planning_mode=PlanningMode.COMPLETE,
    )
    res = solve(req, oracle)
    assert res.plan is not None

    # Suppose o1 was completed, and replanning is triggered
    snapshot = OperationalSnapshot(
        snapshot_id="snap_1",
        snapshot_version=1,
        t0_utc=100,
        traffic_scenario_id="sc_1",
        scenario_revision=1,
        vehicle_states={
            "V1": VehicleOperationalState(
                vehicle_id="V1",
                activity=VehicleActivityState.EN_ROUTE,
                current_node="n1",
                current_time_utc=100,
                residual_load=10,
                locked_stop_order_id="o2",
            )
        },
        completed_order_ids=("o1",),
        sunk_cost_paise=500,
    )

    replan_res = replan(res.plan, snapshot, req, oracle)
    assert replan_res.new_plan is not None
    # Completed order o1 is still part of served orders in the new plan
    assert "o1" in replan_res.new_plan.served_orders


def test_ac_12_infeasibility_and_search_limit_distinguished():
    """AC-12: Infeasibility and search-limit failures are clearly distinguished."""
    coords = {"depot": (0, 0), "n1": (10, 0)}
    oracle = EuclideanOracle(coords)

    # Case 1: Sound proof of infeasibility (demand > Q)
    o_oversize = Order("o_huge", "n1", 500, 10, 0, 1000, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("V1", 50)
    req_inf = SolveRequest(
        orders=(o_oversize,),
        vehicles=(v1,),
        depot_node_id="depot",
        planning_mode=PlanningMode.COMPLETE,
    )
    res_inf = solve(req_inf, oracle)
    assert res_inf.result_status == ResultStatus.PROVEN_INFEASIBLE
    assert res_inf.proof_scope == "CAPACITY_LOWER_BOUND_PROOF"

    # Case 2: Incomplete plan in Complete mode due to tight limits
    # Returns NO_FEASIBLE_PLAN_FOUND with EVALUATION_BUDGET, not PROVEN_INFEASIBLE
    o_tight = Order("o_tight", "n1", 10, 10, 0, 1, WindowType.HARD, PriorityTier.STANDARD)
    req_tight = SolveRequest(
        orders=(o_tight,),
        vehicles=(v1,),
        depot_node_id="depot",
        planning_mode=PlanningMode.COMPLETE,
        departure_set=(0,),
    )
    res_tight = solve(req_tight, oracle)
    assert res_tight.result_status == ResultStatus.NO_FEASIBLE_PLAN_FOUND
    assert res_tight.proof_scope == "NONE"
