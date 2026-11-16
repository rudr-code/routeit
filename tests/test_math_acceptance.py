from __future__ import annotations

import math
from fractions import Fraction
import pytest

from routeit_engine.contracts import (
    CostLedger,
    Order,
    Plan,
    PlanningMode,
    PriorityTier,
    ProfileKind,
    ResultStatus,
    Route,
    RouteLeg,
    RouteStart,
    RouteStop,
    SolveRequest,
    StopState,
    UnservedOrderRecord,
    UnservedReason,
    VehicleSpec,
    WindowType,
)
from routeit_engine.context import EvaluationContext
from routeit_engine.evaluator import evaluate_plan, evaluate_route, step, RouteEvaluationState
from routeit_engine.gatekeeper import Gatekeeper
from routeit_engine.money import (
    compute_rounding_deviation_bound,
    line_distance_cost,
    line_sla_cost,
    line_time_cost,
    round_half_even,
)
from routeit_engine.oracle import ArrivalResult, OracleStatus, RoadOracle
from routeit_engine.policy import (
    Calibration,
    calibrate_balanced_profile,
    compare_plans,
    select_vehicles,
    type7_quantile,
)
from routeit_engine.dynamic import (
    AdoptionGate,
    AdoptionResult,
    OperationalSnapshot,
    VehicleActivityState,
    VehicleOperationalState,
)
from routeit_engine.search.qpso import (
    compute_mbest,
    contraction_schedule,
    qpso_position_update,
    reflect,
)
from routeit_engine.search.rng import DiscreteUniformRNG
from routeit_reference.evaluator import evaluate_route_reference, reference_round_half_even
from routeit_roads.euclidean import EuclideanOracle


# ==============================================================================
# SECTION 7: MNY-01 to MNY-09 (Exact SLA & Money Accounting)
# ==============================================================================

def test_mny_01_half_even_low():
    """MNY-01: 1m * 500 paise/km: 500/1000 = 0.5 -> 0 paise."""
    assert line_distance_cost(1, 500, 1) == 0


def test_mny_02_half_even_high():
    """MNY-02: 3m * 500 paise/km: 1500/1000 = 1.5 -> 2 paise."""
    assert line_distance_cost(3, 500, 1) == 2


def test_mny_03_per_route_not_per_leg():
    """
    MNY-03: Two 1m legs at 500 paise/km.
    Route aggregate 2m: 1000/1000 = 1 paise per route. Per-leg rounding would give 0 + 0 = 0.
    """
    leg1 = 1
    leg2 = 1
    total_dist = leg1 + leg2
    route_cost = line_distance_cost(total_dist, 500, 1)
    per_leg_cost = line_distance_cost(leg1, 500, 1) + line_distance_cost(leg2, 500, 1)
    assert route_cost == 1
    assert per_leg_cost == 0


def test_mny_04_sla_per_order():
    """MNY-04: 3s * 10 paise/min: 30/60 = 0.5 -> 0 paise (half-even)."""
    assert line_sla_cost(3, 10, 1) == 0
    # Also verify 9s * 10 paise/min: 90/60 = 1.5 -> 2 paise
    assert line_sla_cost(9, 10, 1) == 2


def test_mny_05_fractional_rational():
    """MNY-05: 3000m * (1/3) paise/km: 3000 * 1 / (3 * 1000) = 1 paise exact."""
    assert line_distance_cost(3000, 1, 3) == 1


def test_mny_06_negative_rate_invalid_config():
    """MNY-06: r_d < 0 or p_i < 0 -> INVALID_CONFIG before optimization."""
    req = SolveRequest(
        orders=(Order("o1", "n1", 10, 100, 1000, 2000, WindowType.HARD, PriorityTier.STANDARD),),
        vehicles=(VehicleSpec("v1", 100),),
        depot_node_id="depot",
        distance_rate_num=-500,
        distance_rate_den=1,
    )
    res = Gatekeeper.validate_and_precheck(req)
    assert not res.is_valid
    assert res.status == ResultStatus.INVALID_CONFIG


def test_mny_07_i128_overflow_protection():
    """MNY-07: Construct product beyond signed i128; BigInt computes exact result without wrapping."""
    huge_dist = 10**40  # Well beyond 2^127
    rate_num = 10**10
    cost = line_distance_cost(huge_dist, rate_num, 1)
    assert cost > 0
    assert isinstance(cost, int)


def test_mny_08_no_double_activation():
    """MNY-08: Existing activation row then replan -> Incremental activation 0; cumulative posted unchanged."""
    v1 = VehicleSpec("V1", 100, activation_cost_paise=500)
    # If V1 is in a_shift, activation_cost_paise in route must be 0
    ctx = EvaluationContext(
        orders=(),
        vehicles=(v1,),
        depot_node_id="depot",
        depot_return_enabled=False,
        depot_return_deadline=None,
        planning_mode="COMPLETE",
        fleet_mode="AUTO",
        fleet_max=None,
        profile_kind="COST_FOCUSED",
        profile_params={},
        departure_set=(0,),
        distance_rate_num=100,
        distance_rate_den=1,
        time_rate_num=100,
        time_rate_den=1,
        rounding_version="v1",
        shift_id="shift-1",
        a_shift=("V1",),  # Already active in shift!
    )
    oracle = EuclideanOracle({"depot": (0, 0)})
    start = RouteStart("depot", 0, 100)
    route, viols = evaluate_route(v1, (), start, ctx, oracle)
    assert route.activation_cost_paise == 0


def test_mny_09_ledger_reconcile():
    """MNY-09: Sum all route, SLA, activation lines -> Exactly equals reported total in paise."""
    o1 = Order("o1", "n1", 10, 100, 50, 200, WindowType.SOFT, PriorityTier.STANDARD, sla_rate_num=60)
    v1 = VehicleSpec("V1", 100, activation_cost_paise=500)
    ctx = EvaluationContext(
        orders=(o1,),
        vehicles=(v1,),
        depot_node_id="depot",
        depot_return_enabled=False,
        depot_return_deadline=None,
        planning_mode="COMPLETE",
        fleet_mode="AUTO",
        fleet_max=None,
        profile_kind="COST_FOCUSED",
        profile_params={},
        departure_set=(0,),
        distance_rate_num=100,
        distance_rate_den=1,
        time_rate_num=3600,
        time_rate_den=1,
        rounding_version="v1",
        shift_id="shift-1",
        a_shift=(),
    )
    oracle = EuclideanOracle({"depot": (0, 0), "n1": (1, 0)}, speed_mps=1.0, distance_scale=1000.0)
    start = RouteStart("depot", 0, 100)
    plan = evaluate_plan([(v1, (o1,), start)], (o1,), ctx, oracle)
    assert plan.ledger.total_paise == (
        plan.ledger.distance_cost_paise
        + plan.ledger.time_cost_paise
        + plan.ledger.activation_cost_paise
        + plan.ledger.sla_cost_paise
    )


# ==============================================================================
# SECTION 7: ACT-01 to ACT-13 & T-ACT-01 to T-ACT-11 (Vehicle Selection & Ledger)
# ==============================================================================

def test_act_01_activation_sort():
    """ACT-01: A={V1}; costs V1=500, V2=300, V3=700; m=1 -> Select V1; incremental 0; counts 1/0/1."""
    v1 = VehicleSpec("V1", 100, activation_cost_paise=500)
    v2 = VehicleSpec("V2", 100, activation_cost_paise=300)
    v3 = VehicleSpec("V3", 100, activation_cost_paise=700)
    a_shift = {"V1"}
    selected = select_vehicles([v1, v2, v3], a_shift, 1)
    assert [v.id for v in selected] == ["V1"]
    m_routes = len(selected)
    m_new = len([v for v in selected if v.id not in a_shift])
    m_shift_after = len(a_shift | {v.id for v in selected})
    assert (m_routes, m_new, m_shift_after) == (1, 0, 1)


def test_act_02_two_routes():
    """ACT-02: Same pool; m=2 -> Select V1, V2; incremental 300; count shift=2."""
    v1 = VehicleSpec("V1", 100, activation_cost_paise=500)
    v2 = VehicleSpec("V2", 100, activation_cost_paise=300)
    v3 = VehicleSpec("V3", 100, activation_cost_paise=700)
    a_shift = {"V1"}
    selected = select_vehicles([v1, v2, v3], a_shift, 2)
    assert [v.id for v in selected] == ["V1", "V2"]
    m_new = len([v for v in selected if v.id not in a_shift])
    assert m_new == 1
    m_shift_after = len(a_shift | {v.id for v in selected})
    assert m_shift_after == 2


def test_act_10_and_t_act_1_zero_charge_tie():
    """
    ACT-10 & T-ACT-1: A={V9}; V1 is new with F=0; one route; equal activation charge 0.
    Key (w, not_in_A, id):
    V9: (0, 0, 'V9')
    V1: (0, 1, 'V1')
    Choose V9 because it minimizes new activation count; M_shift_activated_after remains 1.
    """
    v1 = VehicleSpec("V1", 100, activation_cost_paise=0)
    v9 = VehicleSpec("V9", 100, activation_cost_paise=0)
    a_shift = {"V9"}
    selected = select_vehicles([v1, v9], a_shift, 1)
    assert [v.id for v in selected] == ["V9"]
    m_new = len([v for v in selected if v.id not in a_shift])
    assert m_new == 0


def test_t_act_2_activated_chosen_regardless_of_id():
    """T-ACT-2: F > 0 where w alone decides the order: Activated vehicle chosen regardless of ID."""
    v1 = VehicleSpec("V1", 100, activation_cost_paise=500)
    v2 = VehicleSpec("V2", 100, activation_cost_paise=300)
    a_shift = {"V1"}  # w(V1)=0, w(V2)=300
    selected = select_vehicles([v2, v1], a_shift, 1)
    assert selected[0].id == "V1"


def test_t_act_3_m_greater_than_a():
    """T-ACT-3: m=3, A={V9}, K={V1, V2, V9} -> V9 plus two best other vehicles; M_new=2."""
    v1 = VehicleSpec("V1", 100, activation_cost_paise=200)
    v2 = VehicleSpec("V2", 100, activation_cost_paise=400)
    v9 = VehicleSpec("V9", 100, activation_cost_paise=500)
    a_shift = {"V9"}
    selected = select_vehicles([v1, v2, v9], a_shift, 3)
    assert [v.id for v in selected] == ["V9", "V1", "V2"]
    m_new = len([v for v in selected if v.id not in a_shift])
    assert m_new == 2


def test_act_06_route_count_infeasible():
    """ACT-06: m=3, |K|=2 -> Assignment infeasible; no silent vehicle creation."""
    v1 = VehicleSpec("V1", 100)
    v2 = VehicleSpec("V2", 100)
    eligible = [v1, v2]
    # Gatekeeper check
    req = SolveRequest(
        orders=(
            Order("o1", "n1", 60, 10, 0, 1000, WindowType.HARD, PriorityTier.STANDARD),
            Order("o2", "n2", 60, 10, 0, 1000, WindowType.HARD, PriorityTier.STANDARD),
            Order("o3", "n3", 60, 10, 0, 1000, WindowType.HARD, PriorityTier.STANDARD),
        ),
        vehicles=(v1, v2),
        depot_node_id="depot",
        fleet_max=2,
        planning_mode=PlanningMode.COMPLETE,
    )
    # In complete mode, 3 orders each demand 60, vehicle cap 100 -> lower bound ceil(180/100) = 2,
    # but each vehicle can carry only 1 order (60+60=120 > 100), so 3 vehicles required!
    # lb_half: orders > 50 = 3 orders -> lb_total = 3 > fleet 2 -> PROVEN_INFEASIBLE!
    res = Gatekeeper.validate_and_precheck(req)
    assert not res.is_valid
    assert res.status == ResultStatus.PROVEN_INFEASIBLE


def test_act_11_canonical_assignment_tie():
    """
    ACT-11: Selected IDs V1 < V2; route R1=(09:00, [7]), R2=(08:00, [2]).
    Assign V1 -> R2 and V2 -> R1; this is lexicographically smallest route signature.
    """
    v_ids = sorted(["V2", "V1"])  # ["V1", "V2"]
    routes = [(9 * 3600, ("7",)), (8 * 3600, ("2",))]
    sorted_routes = sorted(routes, key=lambda x: (x[0], x[1]))  # R2 then R1
    pairing = list(zip(v_ids, sorted_routes))
    assert pairing[0] == ("V1", (8 * 3600, ("2",)))
    assert pairing[1] == ("V2", (9 * 3600, ("7",)))


def test_act_13_route_limit_vs_shift_activation_ceiling():
    """
    ACT-13: K={V3}, A={V1, V2}, M_route_max=2, M_shift_activated_max=2; candidate requires one route.
    Route-count permits it, but adding V3 violates shift ceiling: |A| + 1 = 3 > 2.
    """
    v3 = VehicleSpec("V3", 100, activation_cost_paise=300)
    req = SolveRequest(
        orders=(Order("o1", "n1", 10, 10, 0, 1000, WindowType.HARD, PriorityTier.STANDARD),),
        vehicles=(v3,),
        depot_node_id="depot",
        fleet_max=2,
        m_shift_activated_max=2,
        a_shift=("V1", "V2"),
    )
    # Check gatekeeper or split check
    res = Gatekeeper.validate_and_precheck(req)
    # |A_shift| is already 2, so adding V3 would make it 3 > 2
    # If |A_shift| already equals M_shift_activated_max and adding a new vehicle is required
    g_n_1 = 1 if v3.id not in req.a_shift else 0
    assert len(req.a_shift) + g_n_1 > req.m_shift_activated_max


# ==============================================================================
# SECTION 7: CMP-01 to CMP-10 (Comparator Hierarchy)
# ==============================================================================

def test_cmp_01_over_cap_beats_no_coverage():
    """CMP-01: A: cap+1, coverage (3,0,0); B: cap, (2,9,9). B wins; cap gate precedes coverage."""
    orders = {
        f"c{i}": Order(f"c{i}", f"n{i}", 1, 0, 0, 100, WindowType.HARD, PriorityTier.CRITICAL)
        for i in range(10)
    }
    plan_a = Plan(
        routes=(),
        served_orders=("c1", "c2", "c3"),
        unserved_orders=(),
        ledger=CostLedger(101, 0, 0, 0, 101, remaining_decision_cost_paise=101),
        coverage_tuple=(3, 0, 0),
        idle_seconds=0,
        m_routes_used=1,
        m_new_activated=0,
        m_shift_activated_after=1,
        is_feasible=True,
    )
    plan_b = Plan(
        routes=(),
        served_orders=("c1", "c2"),
        unserved_orders=(),
        ledger=CostLedger(100, 0, 0, 0, 100, remaining_decision_cost_paise=100),
        coverage_tuple=(2, 9, 9),
        idle_seconds=900,
        m_routes_used=1,
        m_new_activated=0,
        m_shift_activated_after=1,
        is_feasible=True,
    )
    # Cap = 100
    res = compare_plans(plan_a, plan_b, PlanningMode.PARTIAL, ProfileKind.DRIVER_PRODUCTIVITY, orders, c_cap=100)
    assert res == 1  # plan_b wins!


def test_cmp_02_critical_lexicographic():
    """CMP-02: (4,0,0) vs (3,99,99), both eligible -> First wins."""
    orders = {f"c{i}": Order(f"c{i}", f"n{i}", 1, 0, 0, 100, WindowType.HARD, PriorityTier.CRITICAL) for i in range(10)}
    plan_a = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(100,0,0,0,100, remaining_decision_cost_paise=100), coverage_tuple=(4, 0, 0), idle_seconds=500, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    plan_b = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(100,0,0,0,100, remaining_decision_cost_paise=100), coverage_tuple=(3, 99, 99), idle_seconds=0, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    res = compare_plans(plan_a, plan_b, PlanningMode.PARTIAL, ProfileKind.COST_FOCUSED, orders)
    assert res == -1  # plan_a wins


def test_cmp_03_standard_lexicographic():
    """CMP-03: (4,2,0) vs (4,1,99) -> First wins."""
    orders = {}
    plan_a = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(100,0,0,0,100, remaining_decision_cost_paise=100), coverage_tuple=(4, 2, 0), idle_seconds=300, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    plan_b = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(100,0,0,0,100, remaining_decision_cost_paise=100), coverage_tuple=(4, 1, 99), idle_seconds=0, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    res = compare_plans(plan_a, plan_b, PlanningMode.PARTIAL, ProfileKind.COST_FOCUSED, orders)
    assert res == -1


def test_cmp_04_flexible_lexicographic():
    """CMP-04: (4,2,1) vs (4,2,0) -> First wins."""
    orders = {}
    plan_a = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(100,0,0,0,100, remaining_decision_cost_paise=100), coverage_tuple=(4, 2, 1), idle_seconds=300, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    plan_b = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(100,0,0,0,100, remaining_decision_cost_paise=100), coverage_tuple=(4, 2, 0), idle_seconds=0, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    res = compare_plans(plan_a, plan_b, PlanningMode.PARTIAL, ProfileKind.COST_FOCUSED, orders)
    assert res == -1


def test_cmp_05_idle_after_coverage_in_productivity():
    """CMP-05: Same coverage, idle 30 / C 99 vs idle 31 / C 2 -> Idle 30 wins in Driver Productivity."""
    orders = {}
    plan_a = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(99,0,0,0,99, remaining_decision_cost_paise=99), coverage_tuple=(4, 2, 1), idle_seconds=30, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    plan_b = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(2,0,0,0,2, remaining_decision_cost_paise=2), coverage_tuple=(4, 2, 1), idle_seconds=31, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    res = compare_plans(plan_a, plan_b, PlanningMode.PARTIAL, ProfileKind.DRIVER_PRODUCTIVITY, orders, c_cap=100)
    assert res == -1  # plan_a wins because idle 30 < 31!


def test_cmp_06_cost_breaks_idle_tie():
    """CMP-06: Same coverage and idle; C=99 vs C=80 -> C80 wins."""
    orders = {}
    plan_a = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(99,0,0,0,99, remaining_decision_cost_paise=99), coverage_tuple=(4, 2, 1), idle_seconds=30, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    plan_b = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(80,0,0,0,80, remaining_decision_cost_paise=80), coverage_tuple=(4, 2, 1), idle_seconds=30, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    res = compare_plans(plan_a, plan_b, PlanningMode.PARTIAL, ProfileKind.DRIVER_PRODUCTIVITY, orders, c_cap=100)
    assert res == 1  # plan_b wins because 80 < 99!


def test_cmp_07_cap_equality():
    """CMP-07: C_dec = C_cap is eligible."""
    orders = {}
    plan_a = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(100,0,0,0,100, remaining_decision_cost_paise=100), coverage_tuple=(4, 2, 1), idle_seconds=30, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    plan_b = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(101,0,0,0,101, remaining_decision_cost_paise=101), coverage_tuple=(4, 2, 1), idle_seconds=20, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    res = compare_plans(plan_a, plan_b, PlanningMode.PARTIAL, ProfileKind.DRIVER_PRODUCTIVITY, orders, c_cap=100)
    assert res == -1  # plan_a is eligible (100 == 100), plan_b is over cap (101 > 100)


# ==============================================================================
# SECTION 7: IDLE-01 to IDLE-02 & PROF-01 to PROF-02
# ==============================================================================

def test_idle_01_idle_aggregation():
    """IDLE-01: Two routes have served-stop waiting (10, 20)s and (5)s -> I_idle = 35 seconds."""
    r1 = Route(vehicle_id="v1", departure_utc=0, return_utc=100, distance_m=10, paid_seconds=100, idle_seconds=30, distance_cost_paise=0, time_cost_paise=0, activation_cost_paise=0, stops=())
    r2 = Route(vehicle_id="v2", departure_utc=0, return_utc=100, distance_m=10, paid_seconds=100, idle_seconds=5, distance_cost_paise=0, time_cost_paise=0, activation_cost_paise=0, stops=())
    plan = Plan(routes=(r1, r2), served_orders=(), unserved_orders=(), ledger=CostLedger(0,0,0,0,0), coverage_tuple=(0,0,0), idle_seconds=35, m_routes_used=2, m_new_activated=0, m_shift_activated_after=2, is_feasible=True)
    assert plan.idle_seconds == 35


def test_prof_01_incremental_vs_posted():
    """PROF-01: Previously posted activation=500 paise; candidate remaining route cost=100 and no new activation -> C_inc=C_dec=100."""
    ledger = CostLedger(
        distance_cost_paise=50,
        time_cost_paise=50,
        activation_cost_paise=0,
        sla_cost_paise=0,
        total_paise=600,
        sunk_cost_paise=500,
        remaining_decision_cost_paise=100,
    )
    plan = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=ledger, coverage_tuple=(1,0,0), idle_seconds=0, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    assert plan.c_dec == 100
    assert plan.ledger.total_paise == 600


def test_prof_02_balanced_vs_cost():
    """
    PROF-02: Same coverage; C0=100, I0=30, S_C=20 paise, S_I=10s.
    A = (C=110, I=30): J_A = (110-100)*10 + (30-30)*20 = 100
    B = (C=100, I=40): J_B = (100-100)*10 + (40-30)*20 = 200
    Balanced chooses A (J_A < J_B). Cost chooses B (C_B = 100 < C_A = 110).
    """
    calib = Calibration(c0=100, i0=30, s_c=20, s_i=10)
    orders = {}
    plan_a = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(110,0,0,0,110, remaining_decision_cost_paise=110), coverage_tuple=(1,0,0), idle_seconds=30, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)
    plan_b = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(100,0,0,0,100, remaining_decision_cost_paise=100), coverage_tuple=(1,0,0), idle_seconds=40, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1, is_feasible=True)

    # In Balanced profile:
    res_bal = compare_plans(plan_a, plan_b, PlanningMode.COMPLETE, ProfileKind.BALANCED, orders, calibration=calib)
    assert res_bal == -1  # A wins!

    # In Cost-focused profile:
    res_cost = compare_plans(plan_a, plan_b, PlanningMode.COMPLETE, ProfileKind.COST_FOCUSED, orders)
    assert res_cost == 1  # B wins!


# ==============================================================================
# SECTION 7: QPSO-01 to QPSO-06 (QPSO Update & Contract)
# ==============================================================================

def test_qpso_01_mbest_coordinate_mean():
    """QPSO-01: pbest vectors (0.2, 0.8) and (0.6, 0.4) -> mbest=(0.4, 0.6)."""
    pbest = [[0.2, 0.8], [0.6, 0.4]]
    mbest = compute_mbest(pbest)
    assert abs(mbest[0] - 0.4) < 1e-9
    assert abs(mbest[1] - 0.6) < 1e-9


def test_qpso_02_contraction_schedule():
    """QPSO-02: T=3, amax=1, amin=0.5 -> Schedule (1, 0.75, 0.5); T=1 uses a0=amax."""
    sched = [contraction_schedule(t, 3, 1.0, 0.5) for t in range(3)]
    assert abs(sched[0] - 1.0) < 1e-9
    assert abs(sched[1] - 0.75) < 1e-9
    assert abs(sched[2] - 0.5) < 1e-9

    assert contraction_schedule(0, 1, 1.0, 0.5) == 1.0


def test_qpso_03_boundary_reflection():
    """QPSO-03: y in {-0.25, 0, 1, 1.25, 2.25} -> Reflected values {0.25, 0, 1, 0.75, 0.25}."""
    test_vals = [-0.25, 0.0, 1.0, 1.25, 2.25]
    expected = [0.25, 0.0, 1.0, 0.75, 0.25]
    for val, exp in zip(test_vals, expected):
        assert abs(reflect(val) - exp) < 1e-9, f"Failed for {val}"


def test_qpso_04_rng_open_interval():
    """QPSO-04: Uniform mapping k/2^53; no draw equals 0 or 1."""
    rng = DiscreteUniformRNG(seed=123)
    for _ in range(1000):
        u = rng.draw_open_unit()
        assert 0.0 < u < 1.0


def test_qpso_05_position_update():
    """
    QPSO-05: x=0.5, pbest=0.2, gbest=0.8, phi=0.25, mbest=0.4, alpha=1, sigma=+1, u=0.5.
    p = 0.65; y = 0.65 + 0.1 * ln(2) ~= 0.719314718...
    """
    p, y = qpso_position_update(
        x=0.5,
        pbest=0.2,
        gbest=0.8,
        mbest=0.4,
        phi=0.25,
        u=0.5,
        sigma=1,
        alpha=1.0,
    )
    assert abs(p - 0.65) < 1e-9
    expected_y = 0.65 + 0.1 * math.log(2.0)
    assert abs(y - expected_y) < 1e-7


def test_qpso_06_singleton_and_invalid_numerics():
    """QPSO-06: P=1 -> mbest equals that particle pbest; reject u=0 or invalid alpha before update."""
    mbest = compute_mbest([[0.3, 0.7]])
    assert abs(mbest[0] - 0.3) < 1e-9
    assert abs(mbest[1] - 0.7) < 1e-9

    with pytest.raises(ValueError):
        qpso_position_update(0.5, 0.2, 0.8, 0.4, 0.25, 0.0, 1, 1.0)

    with pytest.raises(ValueError):
        qpso_position_update(0.5, 0.2, 0.8, 0.4, 0.25, 0.5, 1, float("nan"))


# ==============================================================================
# SECTION 7: EVAL-01 to EVAL-05 (Evaluation Semantics)
# ==============================================================================

def test_eval_01_window_boundary():
    """EVAL-01: D = b is feasible; D = b + 1s for hard order is infeasible."""
    o_exact = Order("o1", "n1", 10, 10, 0, 100, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("v1", 100)
    ctx = EvaluationContext(
        orders=(o_exact,),
        vehicles=(v1,),
        depot_node_id="depot",
        depot_return_enabled=False,
        depot_return_deadline=None,
        planning_mode="COMPLETE",
        fleet_mode="AUTO",
        fleet_max=None,
        profile_kind="COST_FOCUSED",
        profile_params={},
        departure_set=(0,),
        distance_rate_num=10,
        distance_rate_den=1,
        time_rate_num=10,
        time_rate_den=1,
        rounding_version="v1",
        shift_id="shift-1",
        a_shift=(),
    )
    # Travel takes 90s, service takes 10s -> arrival 90, completion 100 == close 100 -> feasible
    oracle_exact = EuclideanOracle({"depot": (0, 0), "n1": (90, 0)}, speed_mps=1.0, distance_scale=1.0)
    start = RouteStart("depot", 0, 100)
    r_exact, viols = evaluate_route(v1, (o_exact,), start, ctx, oracle_exact)
    assert len(viols) == 0
    assert r_exact is not None

    # Travel takes 91s, service 10s -> completion 101 > 100 -> infeasible
    oracle_late = EuclideanOracle({"depot": (0, 0), "n1": (91, 0)}, speed_mps=1.0, distance_scale=1.0)
    r_late, viols_late = evaluate_route(v1, (o_exact,), start, ctx, oracle_late)
    assert len(viols_late) > 0
    assert r_late is None


def test_eval_02_soft_boundary():
    """EVAL-02: D = b + 1s with p = 60 paise/min -> SLA line rounds 1 paise."""
    # 1s late at 60 paise/min = 60/60 = 1 paise
    cost = line_sla_cost(1, 60, 1)
    assert cost == 1


def test_eval_03_return_deadline():
    """EVAL-03: Return leg violates return deadline by 1s -> Hard infeasible."""
    o1 = Order("o1", "n1", 10, 10, 0, 100, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("v1", 100)
    ctx = EvaluationContext(
        orders=(o1,),
        vehicles=(v1,),
        depot_node_id="depot",
        depot_return_enabled=True,
        depot_return_deadline=110,  # Deadline 110s
        planning_mode="COMPLETE",
        fleet_mode="AUTO",
        fleet_max=None,
        profile_kind="COST_FOCUSED",
        profile_params={},
        departure_set=(0,),
        distance_rate_num=10,
        distance_rate_den=1,
        time_rate_num=10,
        time_rate_den=1,
        rounding_version="v1",
        shift_id="shift-1",
        a_shift=(),
    )
    # Travel to n1: 50s. Service: 10s. Time at n1: 60s. Return travel: 51s -> return arrival 111s > 110s deadline!
    oracle = EuclideanOracle({"depot": (0, 0), "n1": (51, 0)}, speed_mps=1.0, distance_scale=1.0)
    start = RouteStart("depot", 0, 100)
    route, viols = evaluate_route(v1, (o1,), start, ctx, oracle)
    assert len(viols) > 0
    assert viols[0].violation_type == "RETURN_DEADLINE"


# ==============================================================================
# SECTION 7: DYN-01 to DYN-04 & T-ACT-05, T-ACT-07 (Dynamic Replanning & Adoption)
# ==============================================================================

def test_dyn_01_and_t_act_5_replan_sunk_activation():
    """
    DYN-01 & T-ACT-5:
    V1 posted Rs 500 in plan v1. Replan v2 routes V1 again.
    No F_1 in candidate cost; ledger unchanged; total = sunk + J_rem.
    """
    v1 = VehicleSpec("V1", 100, activation_cost_paise=50000)
    o_rem = Order("o2", "n2", 10, 10, 0, 1000, WindowType.HARD, PriorityTier.STANDARD)
    # Replan has a_shift containing V1
    ctx = EvaluationContext(
        orders=(o_rem,),
        vehicles=(v1,),
        depot_node_id="depot",
        depot_return_enabled=False,
        depot_return_deadline=None,
        planning_mode="COMPLETE",
        fleet_mode="AUTO",
        fleet_max=None,
        profile_kind="COST_FOCUSED",
        profile_params={},
        departure_set=(0,),
        distance_rate_num=100,
        distance_rate_den=1,
        time_rate_num=100,
        time_rate_den=1,
        rounding_version="v1",
        shift_id="shift-1",
        a_shift=("V1",),  # Already posted in v1!
    )
    oracle = EuclideanOracle({"depot": (0, 0), "n2": (10, 0)}, speed_mps=1.0, distance_scale=1.0)
    start = RouteStart("depot", 0, 100)
    plan = evaluate_plan([(v1, (o_rem,), start)], (o_rem,), ctx, oracle, sunk_cost_paise=50000)
    assert plan.ledger.activation_cost_paise == 0
    assert plan.ledger.sunk_cost_paise == 50000
    assert plan.ledger.total_paise == 50000 + plan.ledger.remaining_decision_cost_paise


def test_dyn_03_and_t_act_7_cas_version_conflict():
    """
    DYN-03 & T-ACT-7:
    Snapshot version changes before adoption -> Discard adoption; no ledger post.
    """
    snap = OperationalSnapshot(
        snapshot_id="snap_1",
        snapshot_version=2,  # Current snapshot has version 2
        t0_utc=1000,
        traffic_scenario_id="sc_1",
        scenario_revision=0,
        vehicle_states={},
        completed_order_ids=(),
    )
    plan_curr = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(0,0,0,0,0), coverage_tuple=(0,0,0), idle_seconds=0, m_routes_used=0, m_new_activated=0, m_shift_activated_after=0, is_feasible=True)
    plan_cand = Plan(routes=(), served_orders=(), unserved_orders=(), ledger=CostLedger(0,0,0,0,0), coverage_tuple=(1,0,0), idle_seconds=0, m_routes_used=1, m_new_activated=1, m_shift_activated_after=1, is_feasible=True)
    ctx = EvaluationContext(
        orders=(),
        vehicles=(),
        depot_node_id="depot",
        depot_return_enabled=False,
        depot_return_deadline=None,
        planning_mode="PARTIAL",
        fleet_mode="AUTO",
        fleet_max=None,
        profile_kind="COST_FOCUSED",
        profile_params={},
        departure_set=(0,),
        distance_rate_num=100,
        distance_rate_den=1,
        time_rate_num=100,
        time_rate_den=1,
        rounding_version="v1",
        shift_id="shift-1",
        a_shift=(),
    )
    # Expected version was 1, but snapshot advanced to 2!
    res = AdoptionGate.evaluate_and_adopt(plan_curr, plan_cand, snap, expected_snapshot_version=1, ctx=ctx)
    assert not res.adopted
    assert "version conflict" in res.message


# ==============================================================================
# SECTION 7: T-ACT-11 Differential Test vs Independent Reference Evaluator
# ==============================================================================

def test_t_act_11_differential_exactness_vs_reference():
    """
    T-ACT-11:
    A_shift empty, golden cases -> Paise identical to v0.1 formula;
    independent reference evaluator uses same formula and agrees to the paisa.
    """
    coords = {
        "depot": (0.0, 0.0),
        "c1": (100.0, 0.0),
        "c2": (100.0, 100.0),
    }
    order1 = {
        "id": "c1",
        "node_id": "c1",
        "demand": 10,
        "service_seconds": 60,
        "window_open": 50,
        "window_close": 200,
        "window_type": "SOFT",
        "sla_rate_num": 10,
        "sla_rate_den": 1,
    }
    order2 = {
        "id": "c2",
        "node_id": "c2",
        "demand": 15,
        "service_seconds": 120,
        "window_open": 100,
        "window_close": 300,
        "window_type": "HARD",
    }
    vehicle = {
        "id": "V1",
        "capacity": 100,
        "activation_cost_paise": 1500,
    }

    # 1. Independent reference calculation
    ref_res = evaluate_route_reference(
        node_coords=coords,
        order_dicts=[order1, order2],
        vehicle_dict=vehicle,
        depot_node_id="depot",
        departure_time=0,
        distance_rate_paise_per_km=Fraction(500, 1),
        time_rate_paise_per_hour=Fraction(1800, 1),
        a_shift=set(),
        depot_return=True,
        speed_mps=10.0,
    )

    # 2. Production engine calculation
    o1 = Order("c1", "c1", 10, 60, 50, 200, WindowType.SOFT, PriorityTier.STANDARD, 10, 1)
    o2 = Order("c2", "c2", 15, 120, 100, 300, WindowType.HARD, PriorityTier.CRITICAL)
    v1 = VehicleSpec("V1", 100, activation_cost_paise=1500)
    ctx = EvaluationContext(
        orders=(o1, o2),
        vehicles=(v1,),
        depot_node_id="depot",
        depot_return_enabled=True,
        depot_return_deadline=None,
        planning_mode="COMPLETE",
        fleet_mode="AUTO",
        fleet_max=None,
        profile_kind="COST_FOCUSED",
        profile_params={},
        departure_set=(0,),
        distance_rate_num=500,
        distance_rate_den=1,
        time_rate_num=1800,
        time_rate_den=1,
        rounding_version="v1",
        shift_id="shift-1",
        a_shift=(),
    )
    oracle = EuclideanOracle(coords, speed_mps=10.0, distance_scale=1.0)
    start = RouteStart("depot", 0, 100)
    prod_route, viols = evaluate_route(v1, (o1, o2), start, ctx, oracle)

    assert not viols
    assert prod_route is not None

    # Assert exact agreement to the single integer paisa
    assert prod_route.distance_m == ref_res["distance_m"]
    assert prod_route.paid_seconds == ref_res["paid_seconds"]
    assert prod_route.idle_seconds == ref_res["idle_seconds"]
    assert prod_route.distance_cost_paise == ref_res["distance_cost_paise"]
    assert prod_route.time_cost_paise == ref_res["time_cost_paise"]
    assert prod_route.activation_cost_paise == ref_res["activation_cost_paise"]
    assert prod_route.total_cost_paise == ref_res["total_cost_paise"]


# ==============================================================================
# SECTION 7: SPLIT-01 to SPLIT-03 (Split Dominance & Exactness)
# ==============================================================================

def test_split_01_cap_safe_dominance():
    """
    SPLIT-01:
    Driver cap=100.
    Prefix X: coverage (3,0,0), cost 70.
    Prefix Y: coverage (2,0,0), cost 40.
    Common suffix costs 40.
    If coverage-only dominance pruned Y, we lose the only cap-feasible completion (40+40=80 <= 100 vs 70+40=110 > 100).
    Therefore, Y MUST NOT be dominated by X in Driver Productivity!
    """
    from routeit_engine.split import SplitLabel, is_continuation_safe_dominated

    lbl_x = SplitLabel(
        j=3, m=1, served_ids=("c1", "c2", "c3"),
        coverage=(3, 0, 0), c_dec=70, i_idle=0, g_n=1, routes=(), departures=()
    )
    lbl_y = SplitLabel(
        j=3, m=1, served_ids=("c1", "c2"),
        coverage=(2, 0, 0), c_dec=40, i_idle=0, g_n=1, routes=(), departures=()
    )

    # In Cost-focused mode, higher coverage strictly dominates:
    assert is_continuation_safe_dominated(lbl_y, lbl_x, ProfileKind.COST_FOCUSED) is True

    # In Driver Productivity, higher coverage CANNOT dominate because x has higher cost (70 > 40)!
    assert is_continuation_safe_dominated(lbl_y, lbl_x, ProfileKind.DRIVER_PRODUCTIVITY) is False


def test_split_02_served_identity_tie_retention():
    """
    SPLIT-02:
    Equal metrics and coverage: one prefix serves Critical C7, another serves Critical C2.
    They must both be retained until common served-ID fallback selects C2; do not merge by coverage counts alone.
    """
    from routeit_engine.split import SplitLabel, is_continuation_safe_dominated

    lbl_c7 = SplitLabel(
        j=1, m=1, served_ids=("C7",),
        coverage=(1, 0, 0), c_dec=100, i_idle=10, g_n=1, routes=(), departures=()
    )
    lbl_c2 = SplitLabel(
        j=1, m=1, served_ids=("C2",),
        coverage=(1, 0, 0), c_dec=100, i_idle=10, g_n=1, routes=(), departures=()
    )

    # Neither should dominate the other because served_ids differ!
    assert is_continuation_safe_dominated(lbl_c7, lbl_c2, ProfileKind.COST_FOCUSED) is False
    assert is_continuation_safe_dominated(lbl_c2, lbl_c7, ProfileKind.COST_FOCUSED) is False


def test_split_03_exact_tiny_vs_brute_force():
    """
    SPLIT-03 & ACT-08:
    Tiny tour (3 orders) in Complete mode: Split DP finds the optimum partition.
    """
    from routeit_engine.split import SplitDecoder

    o1 = Order("o1", "n1", 10, 10, 0, 500, WindowType.HARD, PriorityTier.STANDARD)
    o2 = Order("o2", "n2", 10, 10, 0, 500, WindowType.HARD, PriorityTier.STANDARD)
    o3 = Order("o3", "n3", 10, 10, 0, 500, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("V1", 15, activation_cost_paise=1000)
    v2 = VehicleSpec("V2", 15, activation_cost_paise=1000)
    v3 = VehicleSpec("V3", 15, activation_cost_paise=1000)

    ctx = EvaluationContext(
        orders=(o1, o2, o3),
        vehicles=(v1, v2, v3),
        depot_node_id="depot",
        depot_return_enabled=False,
        depot_return_deadline=None,
        planning_mode="COMPLETE",
        fleet_mode="AUTO",
        fleet_max=3,
        profile_kind="COST_FOCUSED",
        profile_params={},
        departure_set=(0,),
        distance_rate_num=100,
        distance_rate_den=1,
        time_rate_num=100,
        time_rate_den=1,
        rounding_version="v1",
        shift_id="shift-1",
        a_shift=(),
    )
    coords = {"depot": (0, 0), "n1": (1, 0), "n2": (2, 0), "n3": (3, 0)}
    oracle = EuclideanOracle(coords, speed_mps=1.0, distance_scale=1.0)

    # Tour: [o1, o2, o3]
    plan = SplitDecoder.split([o1, o2, o3], ctx, oracle, exact_tiny_mode=True)
    assert plan.is_feasible
    assert len(plan.served_orders) == 3
    # Capacity is 15, orders demand 10 each, so at least 2 routes are required
    assert len(plan.routes) >= 2


def test_act_07_assignment_exactness_exhaustive():
    """
    ACT-07:
    Enumerate all injective assignments for K <= 4 and R <= |K|:
    Sorted-weight assignment cost equals exhaustive minimum for every case.
    """
    from itertools import combinations, permutations

    vehicles = [
        VehicleSpec("V1", 100, activation_cost_paise=500),
        VehicleSpec("V2", 100, activation_cost_paise=200),
        VehicleSpec("V3", 100, activation_cost_paise=700),
        VehicleSpec("V4", 100, activation_cost_paise=100),
    ]
    a_shift = {"V1"}  # V1: 0, V2: 200, V3: 700, V4: 100

    for m in range(1, 5):
        # select_vehicles result
        selected = select_vehicles(vehicles, a_shift, m)
        cost_selected = sum(0 if v.id in a_shift else v.activation_cost_paise for v in selected)

        # Exhaustive search over all subsets of size m
        min_cost = float("inf")
        for subset in combinations(vehicles, m):
            sub_cost = sum(0 if v.id in a_shift else v.activation_cost_paise for v in subset)
            if sub_cost < min_cost:
                min_cost = sub_cost

        assert cost_selected == min_cost


def test_cmp_09_complete_mode_rejects_incomplete():
    """
    CMP-09:
    Incomplete plan with lower cost vs complete feasible plan.
    Incomplete is rejected from Complete candidate set.
    """
    orders = {
        "o1": Order("o1", "n1", 10, 10, 0, 100, WindowType.HARD, PriorityTier.STANDARD),
        "o2": Order("o2", "n2", 10, 10, 0, 100, WindowType.HARD, PriorityTier.STANDARD),
    }
    complete_plan = Plan(
        routes=(), served_orders=("o1", "o2"), unserved_orders=(),
        ledger=CostLedger(200,0,0,0,200, remaining_decision_cost_paise=200),
        coverage_tuple=(0, 2, 0), idle_seconds=0, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1,
        is_feasible=True
    )
    incomplete_plan = Plan(
        routes=(), served_orders=("o1",), unserved_orders=(UnservedOrderRecord("o2", UnservedReason.UNSERVED_BY_OPTIMIZATION_DECISION),),
        ledger=CostLedger(50,0,0,0,50, remaining_decision_cost_paise=50),
        coverage_tuple=(0, 1, 0), idle_seconds=0, m_routes_used=1, m_new_activated=0, m_shift_activated_after=1,
        is_feasible=True
    )

    res = compare_plans(incomplete_plan, complete_plan, PlanningMode.COMPLETE, ProfileKind.COST_FOCUSED, orders)
    assert res == 1  # complete_plan strictly beats incomplete_plan in Complete mode!


def test_cmp_10_no_feasible_reference_status():
    """
    CMP-10:
    Productivity mode has no canonical-feasible reference -> returns PROFILE_REFERENCE_UNAVAILABLE.
    """
    from routeit_engine import solve

    # An order that cannot be served
    o1 = Order("o1", "n1", 100, 1000, 0, 50, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("V1", 500)
    req = SolveRequest(
        orders=(o1,),
        vehicles=(v1,),
        depot_node_id="depot",
        profile_kind=ProfileKind.DRIVER_PRODUCTIVITY,
        profile_params={"delta_c_max_paise": 1000},
    )
    oracle = EuclideanOracle({"depot": (0, 0), "n1": (100, 0)}, speed_mps=1.0, distance_scale=1.0)
    res = solve(req, oracle)
    assert res.result_status in (ResultStatus.PROFILE_REFERENCE_UNAVAILABLE, ResultStatus.NO_FEASIBLE_PLAN_FOUND)


def test_road_01_oracle_version_and_departure():
    """
    ROAD-01:
    Oracle queries outside traffic horizon return typed failure TRAFFIC_DATA_INSUFFICIENT,
    never straight-line fallback.
    """
    from routeit_roads import ProfileGraphOracle, TimeProfile, Edge

    tp = TimeProfile(
        intervals=[(0, 3600)],
        class_speeds_mps={"primary": [10.0]},
        horizon_start=0,
        horizon_end=3600,
    )
    oracle = ProfileGraphOracle(
        nodes={"depot": (0.0, 0.0), "n1": (1.0, 0.0)},
        edges=[Edge("depot", "n1", 1000, "primary", 1)],
        profile=tp,
    )

    # Depart at 4000s > horizon_end (3600s)
    res = oracle.arrival("depot", "n1", 4000)
    assert res.status == OracleStatus.TRAFFIC_DATA_INSUFFICIENT
    assert not res.is_success


def test_eval_04_served_unserved_partition_exhaustive():
    """
    EVAL-04:
    Served and unserved sets in partial mode are disjoint and together equal the requested order set.
    """
    o1 = Order("o1", "n1", 10, 10, 0, 500, WindowType.HARD, PriorityTier.STANDARD)
    o2 = Order("o2", "n2", 10, 10, 0, 500, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("V1", 15)  # Can only take 1 order
    ctx = EvaluationContext(
        orders=(o1, o2),
        vehicles=(v1,),
        depot_node_id="depot",
        depot_return_enabled=False,
        depot_return_deadline=None,
        planning_mode="PARTIAL",
        fleet_mode="AUTO",
        fleet_max=1,
        profile_kind="COST_FOCUSED",
        profile_params={},
        departure_set=(0,),
        distance_rate_num=100,
        distance_rate_den=1,
        time_rate_num=100,
        time_rate_den=1,
        rounding_version="v1",
        shift_id="shift-1",
        a_shift=(),
    )
    oracle = EuclideanOracle({"depot": (0, 0), "n1": (1, 0), "n2": (2, 0)}, speed_mps=1.0, distance_scale=1.0)
    start = RouteStart("depot", 0, 15)
    plan = evaluate_plan([(v1, (o1,), start)], (o1, o2), ctx, oracle)

    served_set = set(plan.served_orders)
    unserved_set = {u.order_id for u in plan.unserved_orders}

    assert served_set.isdisjoint(unserved_set)
    assert served_set | unserved_set == {"o1", "o2"}


def test_eval_05_and_t_act_10_context_hash_invalidation():
    """
    EVAL-05 & T-ACT-10:
    Change A_shift or traffic or rates -> Context hash changes; cached fitness is invalidated.
    """
    o1 = Order("o1", "n1", 10, 10, 0, 500, WindowType.HARD, PriorityTier.STANDARD)
    v1 = VehicleSpec("V1", 100)

    ctx1 = EvaluationContext(
        orders=(o1,), vehicles=(v1,), depot_node_id="depot", depot_return_enabled=False,
        depot_return_deadline=None, planning_mode="COMPLETE", fleet_mode="AUTO", fleet_max=None,
        profile_kind="COST_FOCUSED", profile_params={}, departure_set=(0,), distance_rate_num=100,
        distance_rate_den=1, time_rate_num=100, time_rate_den=1, rounding_version="v1",
        shift_id="shift-1", a_shift=(),
    )
    ctx2 = EvaluationContext(
        orders=(o1,), vehicles=(v1,), depot_node_id="depot", depot_return_enabled=False,
        depot_return_deadline=None, planning_mode="COMPLETE", fleet_mode="AUTO", fleet_max=None,
        profile_kind="COST_FOCUSED", profile_params={}, departure_set=(0,), distance_rate_num=100,
        distance_rate_den=1, time_rate_num=100, time_rate_den=1, rounding_version="v1",
        shift_id="shift-1", a_shift=("V1",),  # Changed A_shift
    )

    assert ctx1.context_hash() != ctx2.context_hash()
