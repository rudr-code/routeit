from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from routeit_engine.contracts import (
    Order,
    Plan,
    PlanningMode,
    PriorityTier,
    ProfileKind,
    VehicleSpec,
)
from routeit_engine.money import round_half_even


def select_vehicles(
    eligible: Sequence[VehicleSpec],
    a_shift: Set[str],
    m: int,
) -> List[VehicleSpec]:
    """
    Normative vehicle selection rule (Addendum A.3 & Q-HGALS 5.0 §2.3):
    Sort eligible vehicles by:
      (w_k, 1 if k not in a_shift else 0, vehicle_id)
    where w_k = 0 if k in a_shift else vehicle.activation_cost_paise.
    Already-activated vehicles win equal-weight ties.
    Returns the first m vehicles.
    """
    def key_fn(v: VehicleSpec):
        w_k = 0 if v.id in a_shift else v.activation_cost_paise
        not_in_a = 0 if v.id in a_shift else 1
        return (w_k, not_in_a, v.id)

    sorted_eligible = sorted(eligible, key=key_fn)
    return sorted_eligible[:m]


def type7_quantile(data: Sequence[int], q: float) -> Fraction:
    """
    Type-7 quantile calculation using 1-based order statistics and exact rational arithmetic:
    h = 1 + (N - 1) * q
    j = floor(h)
    gamma = h - j
    Q_q = (1 - gamma) * X[j] + gamma * X[min(j+1, N)]
    """
    N = len(data)
    if N == 0:
        raise ValueError("Cannot calculate quantile of empty data")
    if N == 1:
        return Fraction(data[0], 1)

    s_data = sorted(data)
    # Using exact rational for h
    # q is given as float (e.g. 0.10, 0.90) -> convert to Fraction
    q_frac = Fraction(str(q))
    h_frac = 1 + (N - 1) * q_frac
    j = int(math.floor(float(h_frac)))
    gamma = h_frac - j

    # X[j] is 1-based, so in 0-based index it is j - 1
    idx1 = max(0, min(j - 1, N - 1))
    idx2 = max(0, min(j, N - 1))

    val1 = Fraction(s_data[idx1], 1)
    val2 = Fraction(s_data[idx2], 1)

    q_val = (1 - gamma) * val1 + gamma * val2
    return q_val


@dataclass(frozen=True)
class Calibration:
    c0: int
    i0: int
    s_c: int
    s_i: int
    low_quality: bool = False


def calibrate_balanced_profile(sample_plans: Sequence[Plan]) -> Optional[Calibration]:
    """
    Derives C0, I0 anchors and S_C, S_I scales from a sample of feasible plans.
    """
    feasible_plans = [p for p in sample_plans if p.is_feasible]
    if not feasible_plans:
        return None

    c_vals = [p.c_dec for p in feasible_plans]
    i_vals = [p.idle_seconds for p in feasible_plans]

    c0 = min(c_vals)
    i0 = min(i_vals)

    q90_c = type7_quantile(c_vals, 0.90)
    q10_c = type7_quantile(c_vals, 0.10)
    spread_c = q90_c - q10_c

    q90_i = type7_quantile(i_vals, 0.90)
    q10_i = type7_quantile(i_vals, 0.10)
    spread_i = q90_i - q10_i

    rounded_spread_c = round_half_even(spread_c.numerator, spread_c.denominator)
    rounded_spread_i = round_half_even(spread_i.numerator, spread_i.denominator)

    s_c = max(1, rounded_spread_c)
    s_i = max(1, rounded_spread_i)

    low_q = (rounded_spread_c <= 1 or rounded_spread_i <= 1)
    return Calibration(c0=c0, i0=i0, s_c=s_c, s_i=s_i, low_quality=low_q)


def canonical_route_signature(plan: Plan) -> Tuple[Tuple[str, int, Tuple[str, ...]], ...]:
    """
    Returns deterministic route signature:
    Ordered by (vehicle_id, departure_second, stop_id_sequence).
    """
    routes_sig = []
    for r in plan.routes:
        stop_ids = tuple(s.order_id for s in r.stops)
        routes_sig.append((r.vehicle_id, r.departure_utc, stop_ids))
    return tuple(sorted(routes_sig))


def served_order_signature(plan: Plan, orders_by_id: Dict[str, Order]) -> Tuple[str, ...]:
    """
    Served-order signature in Critical/Standard/Flexible tier order, IDs ascending.
    """
    crit_ids = sorted([oid for oid in plan.served_orders if orders_by_id[oid].priority == PriorityTier.CRITICAL])
    std_ids = sorted([oid for oid in plan.served_orders if orders_by_id[oid].priority == PriorityTier.STANDARD])
    flx_ids = sorted([oid for oid in plan.served_orders if orders_by_id[oid].priority == PriorityTier.FLEXIBLE])
    return tuple(crit_ids + std_ids + flx_ids)


def plan_canonical_key(
    plan: Plan,
    mode: PlanningMode,
    profile: ProfileKind,
    orders_by_id: Dict[str, Order],
    calibration: Optional[Calibration] = None,
    c_cap: Optional[int] = None,
) -> Tuple[Any, ...]:
    """
    Extracts the full lexicographic sorting key for a plan such that:
    plan_x is strictly better than plan_y iff plan_canonical_key(plan_x) < plan_canonical_key(plan_y).
    """
    # Gate 1: Hard feasibility
    # 0 = feasible, 1 = infeasible
    feas_gate = 0 if plan.is_feasible else 1

    # Gate 2: Driver productivity cost cap eligibility
    cap_gate = 0
    if profile == ProfileKind.DRIVER_PRODUCTIVITY and c_cap is not None:
        cap_gate = 0 if plan.c_dec <= c_cap else 1

    # Gate 3: Complete mode zero unserved gate
    complete_gate = 0
    if mode == PlanningMode.COMPLETE:
        complete_gate = 0 if len(plan.unserved_orders) == 0 else 1

    # Coverage tuple: (N_C, N_S, N_F) - we want to MAXIMIZE coverage, so invert for ascending comparison
    neg_coverage = (-plan.coverage_tuple[0], -plan.coverage_tuple[1], -plan.coverage_tuple[2])

    # Profile specific objective tuple
    if profile == ProfileKind.COST_FOCUSED:
        profile_tuple = (plan.c_dec, plan.idle_seconds, plan.m_shift_activated_after)
    elif profile == ProfileKind.BALANCED:
        if calibration is not None:
            # J_B integer numerator: (C_dec - C0)*S_I + (I_idle - I0)*S_C
            j_b_num = (plan.c_dec - calibration.c0) * calibration.s_i + (plan.idle_seconds - calibration.i0) * calibration.s_c
        else:
            j_b_num = plan.c_dec
        profile_tuple = (j_b_num, plan.c_dec, plan.idle_seconds, plan.m_shift_activated_after)
    elif profile == ProfileKind.DRIVER_PRODUCTIVITY:
        # First require C_dec <= C_cap (in cap_gate); then maximize coverage; then minimize (I_idle, C_dec, M_shift_activated_after)
        profile_tuple = (plan.idle_seconds, plan.c_dec, plan.m_shift_activated_after)
    else:
        profile_tuple = (plan.c_dec, plan.idle_seconds, plan.m_shift_activated_after)

    # Technical tie-breaks
    served_sig = served_order_signature(plan, orders_by_id)
    route_sig = canonical_route_signature(plan)
    plan_hash = plan.canonical_hash

    if profile == ProfileKind.DRIVER_PRODUCTIVITY:
        # In Driver Productivity:
        # Feasibility gate -> Cap gate -> Lexicographic coverage -> Profile tuple (idle, cost, fleet) -> Tie-breaks
        return (feas_gate, cap_gate, complete_gate, neg_coverage, profile_tuple, served_sig, route_sig, plan_hash)
    else:
        # Cost or Balanced:
        # Feasibility gate -> Complete gate -> Lexicographic coverage (for Partial mode) -> Profile tuple -> Tie-breaks
        return (feas_gate, complete_gate, neg_coverage, profile_tuple, served_sig, route_sig, plan_hash)


def compare_plans(
    x: Plan,
    y: Plan,
    mode: PlanningMode,
    profile: ProfileKind,
    orders_by_id: Dict[str, Order],
    calibration: Optional[Calibration] = None,
    c_cap: Optional[int] = None,
) -> int:
    """
    Returns:
      -1 if x is strictly better than y
      +1 if y is strictly better than x
       0 if x and y are canonically equal
    """
    key_x = plan_canonical_key(x, mode, profile, orders_by_id, calibration, c_cap)
    key_y = plan_canonical_key(y, mode, profile, orders_by_id, calibration, c_cap)

    if key_x < key_y:
        return -1
    elif key_x > key_y:
        return 1
    else:
        return 0
