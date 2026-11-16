from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from routeit_engine.contracts import (
    CostLedger,
    Order,
    Plan,
    PlanningMode,
    PriorityTier,
    ProfileKind,
    Route,
    RouteLeg,
    RouteStart,
    RouteStop,
    UnservedOrderRecord,
    UnservedReason,
    VehicleSpec,
    WindowType,
)
from routeit_engine.context import EvaluationContext
from routeit_engine.evaluator import evaluate_plan, evaluate_route
from routeit_engine.money import compute_rounding_deviation_bound
from routeit_engine.oracle import RoadOracle
from routeit_engine.policy import (
    Calibration,
    canonical_route_signature,
    select_vehicles,
    served_order_signature,
)


@dataclass(frozen=True)
class SplitLabel:
    j: int  # Tour position consumed (0..n)
    m: int  # Number of routes opened
    served_ids: Tuple[str, ...]  # Exact served order IDs in order
    coverage: Tuple[int, int, int]  # (N_C, N_S, N_F)
    c_dec: int  # Incremental decision cost in paise
    i_idle: int  # Idle seconds
    g_n: int  # New activations count
    routes: Tuple[Tuple[Order, ...], ...]  # Route segments
    departures: Tuple[int, ...]  # Departure time per route
    traces: Tuple[Route, ...] = ()  # Evaluated route objects


def is_continuation_safe_dominated(
    cand: SplitLabel,
    other: SplitLabel,
    profile: ProfileKind,
    calibration: Optional[Calibration] = None,
    c_cap: Optional[int] = None,
) -> bool:
    """
    Checks if 'cand' is dominated by 'other' under continuation-safe dominance rule (Q-HGALS 5.0 §2.7).
    Must have same structural state (j, m).
    """
    if cand.j != other.j or cand.m != other.m:
        return False

    cov_cand = cand.coverage
    cov_other = other.coverage

    # If other strictly beats cand in coverage:
    # (other_C > cand_C) or (other_C == cand_C and other_S > cand_S) ...
    other_strictly_better_cov = (
        cov_other[0] > cov_cand[0]
        or (cov_other[0] == cov_cand[0] and cov_other[1] > cov_cand[1])
        or (cov_other[0] == cov_cand[0] and cov_other[1] == cov_cand[1] and cov_other[2] > cov_cand[2])
    )

    other_equal_cov = (cov_other == cov_cand)

    if profile in (ProfileKind.COST_FOCUSED, ProfileKind.BALANCED):
        if other_strictly_better_cov:
            return True
        if other_equal_cov:
            if profile == ProfileKind.COST_FOCUSED:
                if other.c_dec < cand.c_dec:
                    return True
                if other.c_dec == cand.c_dec and other.i_idle <= cand.i_idle:
                    return other.served_ids == cand.served_ids
            elif profile == ProfileKind.BALANCED:
                # Compare J_B
                if calibration is not None:
                    j_b_other = (other.c_dec - calibration.c0) * calibration.s_i + (other.i_idle - calibration.i0) * calibration.s_c
                    j_b_cand = (cand.c_dec - calibration.c0) * calibration.s_i + (cand.idle_seconds - calibration.i0) * calibration.s_c
                else:
                    j_b_other = other.c_dec
                    j_b_cand = cand.c_dec

                if j_b_other < j_b_cand:
                    return True
                if j_b_other == j_b_cand and other.c_dec <= cand.c_dec and other.i_idle <= cand.i_idle:
                    return other.served_ids == cand.served_ids
    elif profile == ProfileKind.DRIVER_PRODUCTIVITY:
        # In Driver Productivity:
        # Strictly better coverage dominates ONLY IF other.c_dec <= cand.c_dec (SPLIT-01 counterexample)
        if other_strictly_better_cov:
            if other.c_dec <= cand.c_dec:
                return True
            return False
        if other_equal_cov:
            # Idle <= and also c_dec <= as cap resource
            if other.i_idle < cand.i_idle and other.c_dec <= cand.c_dec:
                return True
            if other.i_idle == cand.i_idle and other.c_dec <= cand.c_dec:
                return other.served_ids == cand.served_ids

    return False


class SplitDecoder:
    @staticmethod
    def split(
        giant_tour: Sequence[Order],
        ctx: EvaluationContext,
        oracle: RoadOracle,
        calibration: Optional[Calibration] = None,
        c_cap: Optional[int] = None,
        exact_tiny_mode: bool = False,
        frontier_limit: int = 500,
    ) -> Plan:
        n = len(giant_tour)
        eligible = [v for v in ctx.vehicles if v.available]
        q_limit = min(len(eligible), ctx.fleet_max) if ctx.fleet_max is not None else len(eligible)

        # Build G_C and G_N prefix tables
        sorted_k = select_vehicles(eligible, set(ctx.a_shift), len(eligible))
        g_c = [0] * (len(sorted_k) + 1)
        g_n = [0] * (len(sorted_k) + 1)
        for idx, veh in enumerate(sorted_k):
            w_k = 0 if veh.id in ctx.a_shift else veh.activation_cost_paise
            not_in_a = 0 if veh.id in ctx.a_shift else 1
            g_c[idx + 1] = g_c[idx] + w_k
            g_n[idx + 1] = g_n[idx] + not_in_a

        # Label storage per state (j, m): list of nondominated labels
        dp: Dict[Tuple[int, int], List[SplitLabel]] = {}
        dp[(0, 0)] = [
            SplitLabel(
                j=0,
                m=0,
                served_ids=(),
                coverage=(0, 0, 0),
                c_dec=0,
                i_idle=0,
                g_n=0,
                routes=(),
                departures=(),
                traces=(),
            )
        ]

        dummy_veh = sorted_k[0] if sorted_k else VehicleSpec(id="dummy", capacity=1000)

        for j_from in range(n):
            for m in range(q_limit):
                current_labels = dp.get((j_from, m), [])
                if not current_labels:
                    continue

                m_next = m + 1
                if m_next > q_limit:
                    continue

                # Check shift activation ceiling
                if ctx.m_shift_activated_max is not None:
                    if len(ctx.a_shift) + g_n[m_next] > ctx.m_shift_activated_max:
                        continue

                delta_g_c = g_c[m_next] - g_c[m_next - 1]
                delta_g_n = g_n[m_next] - g_n[m_next - 1]

                # Standalone skip transition for Partial Mode: tour position j_from + 1 is skipped
                if ctx.planning_mode == PlanningMode.PARTIAL:
                    skip_j = j_from + 1
                    target_key = (skip_j, m)
                    if target_key not in dp:
                        dp[target_key] = []
                    for lbl in current_labels:
                        new_lbl = SplitLabel(
                            j=skip_j,
                            m=m,
                            served_ids=lbl.served_ids,
                            coverage=lbl.coverage,
                            c_dec=lbl.c_dec,
                            i_idle=lbl.i_idle,
                            g_n=lbl.g_n,
                            routes=lbl.routes,
                            departures=lbl.departures,
                            traces=lbl.traces,
                        )
                        dp[target_key].append(new_lbl)

                # Segment transitions: evaluate blocks [j_from + 1, j_to]
                for j_to in range(j_from + 1, n + 1):
                    block_orders = giant_tour[j_from:j_to]
                    # Check if total capacity is satisfied
                    if sum(o.demand for o in block_orders) > dummy_veh.capacity:
                        break

                    # Internal subsequences:
                    # In Complete Mode: exactly block_orders.
                    # In Partial Mode:
                    # If exact_tiny_mode (e.g. n <= 6): enumerate non-empty subsequences with endpoints served.
                    # Otherwise: the whole block.
                    if ctx.planning_mode == PlanningMode.PARTIAL and exact_tiny_mode and len(block_orders) > 2:
                        internal = block_orders[1:-1]
                        subseqs = []
                        for r_len in range(len(internal) + 1):
                            for sub in combinations(internal, r_len):
                                subseqs.append((block_orders[0],) + sub + (block_orders[-1],))
                    else:
                        subseqs = [tuple(block_orders)]

                    for subseq in subseqs:
                        for depart_t in ctx.departure_set:
                            start = RouteStart(
                                node_id=ctx.depot_node_id,
                                departure_time=depart_t,
                                initial_load=dummy_veh.capacity,
                            )
                            route_eval, viols = evaluate_route(dummy_veh, subseq, start, ctx, oracle)
                            if viols or route_eval is None:
                                continue

                            # Valid route segment! Extend labels
                            seg_c = route_eval.distance_cost_paise + route_eval.time_cost_paise + sum(s.sla_cost_paise for s in route_eval.stops)
                            seg_idle = route_eval.idle_seconds

                            # Incremental coverage
                            n_c = sum(1 for o in subseq if o.priority == PriorityTier.CRITICAL)
                            n_s = sum(1 for o in subseq if o.priority == PriorityTier.STANDARD)
                            n_f = sum(1 for o in subseq if o.priority == PriorityTier.FLEXIBLE)
                            seg_sub_ids = tuple(o.id for o in subseq)

                            target_key = (j_to, m_next)
                            if target_key not in dp:
                                dp[target_key] = []

                            for parent_lbl in current_labels:
                                new_c = parent_lbl.c_dec + seg_c + delta_g_c
                                new_idle = parent_lbl.i_idle + seg_idle
                                new_gn = parent_lbl.g_n + delta_g_n
                                new_cov = (
                                    parent_lbl.coverage[0] + n_c,
                                    parent_lbl.coverage[1] + n_s,
                                    parent_lbl.coverage[2] + n_f,
                                )
                                new_served = parent_lbl.served_ids + seg_sub_ids

                                cand_label = SplitLabel(
                                    j=j_to,
                                    m=m_next,
                                    served_ids=new_served,
                                    coverage=new_cov,
                                    c_dec=new_c,
                                    i_idle=new_idle,
                                    g_n=new_gn,
                                    routes=parent_lbl.routes + (subseq,),
                                    departures=parent_lbl.departures + (depart_t,),
                                    traces=parent_lbl.traces + (route_eval,),
                                )

                                # Check dominance against existing labels
                                is_dominated = False
                                for existing in dp[target_key]:
                                    if is_continuation_safe_dominated(
                                        cand_label,
                                        existing,
                                        ProfileKind(ctx.profile_kind),
                                        calibration,
                                        c_cap,
                                    ):
                                        is_dominated = True
                                        break

                                if not is_dominated:
                                    # Prune labels dominated by cand_label
                                    dp[target_key] = [
                                        ex
                                        for ex in dp[target_key]
                                        if not is_continuation_safe_dominated(
                                            ex,
                                            cand_label,
                                            ProfileKind(ctx.profile_kind),
                                            calibration,
                                            c_cap,
                                        )
                                    ]
                                    dp[target_key].append(cand_label)

                            # Frontier limit pruning if bounded mode
                            if len(dp[target_key]) > frontier_limit:
                                dp[target_key] = dp[target_key][:frontier_limit]

        # Gather all labels reaching j = n
        terminal_labels: List[SplitLabel] = []
        for m in range(q_limit + 1):
            terminal_labels.extend(dp.get((n, m), []))

        if not terminal_labels:
            # No feasible partition found
            return evaluate_plan([], ctx.orders, ctx, oracle)

        # Convert terminal labels to candidate Plans and select best using common comparator
        order_map = {o.id: o for o in ctx.orders}
        best_plan: Optional[Plan] = None

        for lbl in terminal_labels:
            m = lbl.m
            if m == 0 and ctx.planning_mode == PlanningMode.COMPLETE:
                continue

            selected_vehs = select_vehicles(eligible, set(ctx.a_shift), m)

            # Route descriptors pairing:
            # Sort selected IDs ascending; sort route descriptors by (departure second, stop-ID sequence)
            route_descriptors = []
            for r_idx, r_orders in enumerate(lbl.routes):
                dep = lbl.departures[r_idx]
                stop_ids = tuple(o.id for o in r_orders)
                route_descriptors.append((dep, stop_ids, r_orders))

            sorted_routes = sorted(route_descriptors, key=lambda x: (x[0], x[1]))
            sorted_veh_ids = sorted(selected_vehs, key=lambda v: v.id)

            routes_with_specs = []
            for idx, (_, _, r_orders) in enumerate(sorted_routes):
                veh = sorted_veh_ids[idx]
                start = RouteStart(
                    node_id=ctx.depot_node_id,
                    departure_time=sorted_routes[idx][0],
                    initial_load=veh.capacity,
                )
                routes_with_specs.append((veh, r_orders, start))

            cand_plan = evaluate_plan(routes_with_specs, ctx.orders, ctx, oracle)
            if best_plan is None:
                best_plan = cand_plan
            else:
                from routeit_engine.policy import compare_plans
                cmp_res = compare_plans(
                    cand_plan,
                    best_plan,
                    PlanningMode(ctx.planning_mode),
                    ProfileKind(ctx.profile_kind),
                    order_map,
                    calibration,
                    c_cap,
                )
                if cmp_res < 0:
                    best_plan = cand_plan

        return best_plan if best_plan is not None else evaluate_plan([], ctx.orders, ctx, oracle)
