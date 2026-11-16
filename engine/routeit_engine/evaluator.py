from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from routeit_engine.contracts import (
    CostLedger,
    Order,
    Plan,
    Route,
    RouteLeg,
    RouteStart,
    RouteStop,
    StopState,
    UnservedOrderRecord,
    UnservedReason,
    VehicleSpec,
    Violation,
    ViolationType,
    WindowType,
)
from routeit_engine.context import EvaluationContext
from routeit_engine.money import (
    compute_rounding_deviation_bound,
    line_distance_cost,
    line_sla_cost,
    line_time_cost,
)
from routeit_engine.oracle import OracleStatus, RoadOracle


class RouteEvaluationState:
    def __init__(self, node: str, time: int, load: int):
        self.node = node
        self.time = time
        self.load = load


def step(
    state: RouteEvaluationState,
    stop: Order,
    ctx: EvaluationContext,
    oracle: RoadOracle,
) -> Tuple[Optional[RouteEvaluationState], Optional[RouteStop], Optional[RouteLeg], Optional[Violation]]:
    """
    Core schedule step function:
    r = oracle.arrival(state.node, stop.node_id, state.time, ctx)
    A = r.arrive_t
    B = max(A, stop.window_open)
    W = B - A  # customer-window idle waiting, >= 0
    D = B + stop.service_seconds
    L = max(0, D - stop.window_close)
    if stop.window_type == HARD and L > 0:
        return Violation(HARD_WINDOW, ...)
    sla = line_sla_cost(L, stop.sla_rate_num, stop.sla_rate_den) if stop.window_type == SOFT else 0
    """
    arrival_res = oracle.arrival(state.node, stop.node_id, state.time, ctx)
    if not arrival_res.is_success:
        return (
            None,
            None,
            None,
            Violation(
                violation_type=ViolationType.ROAD_PATH,
                entity_id=stop.id,
                details=f"Road failure {arrival_res.status}: {arrival_res.message}",
            ),
        )

    arrive_t = arrival_res.arrive_t
    service_start = max(arrive_t, stop.window_open)
    wait_s = service_start - arrive_t
    service_end = service_start + stop.service_seconds
    lateness_s = max(0, service_end - stop.window_close)

    if stop.window_type == WindowType.HARD and lateness_s > 0:
        return (
            None,
            None,
            None,
            Violation(
                violation_type=ViolationType.HARD_WINDOW,
                entity_id=stop.id,
                details=f"Hard window violation at {stop.id}: completion {service_end} > close {stop.window_close}",
                magnitude=lateness_s,
            ),
        )

    if state.load < stop.demand:
        return (
            None,
            None,
            None,
            Violation(
                violation_type=ViolationType.CAPACITY,
                entity_id=stop.id,
                details=f"Capacity exceeded at {stop.id}: demand {stop.demand} > remaining load {state.load}",
                magnitude=stop.demand - state.load,
            ),
        )

    sla_paise = line_sla_cost(lateness_s, stop.sla_rate_num, stop.sla_rate_den) if stop.window_type == WindowType.SOFT else 0

    route_stop = RouteStop(
        seq=0,
        order_id=stop.id,
        arrival_utc=arrive_t,
        wait_s=wait_s,
        service_start_utc=service_start,
        service_end_utc=service_end,
        lateness_s=lateness_s,
        sla_cost_paise=sla_paise,
        stop_state=StopState.PENDING,
    )

    leg = RouteLeg(
        seq=0,
        from_node=state.node,
        to_node=stop.node_id,
        distance_m=arrival_res.distance_m,
        depart_utc=state.time,
        arrive_utc=arrive_t,
        edge_ids=arrival_res.edge_ids,
    )

    next_state = RouteEvaluationState(
        node=stop.node_id,
        time=service_end,
        load=state.load - stop.demand,
    )

    return next_state, route_stop, leg, None


def evaluate_route(
    vehicle: VehicleSpec,
    orders: Sequence[Order],
    start: RouteStart,
    ctx: EvaluationContext,
    oracle: RoadOracle,
) -> Tuple[Optional[Route], List[Violation]]:
    """
    Evaluates a single sequence of orders for a given vehicle starting from RouteStart.
    """
    violations: List[Violation] = []
    total_demand = sum(o.demand for o in orders)
    if total_demand > vehicle.capacity:
        violations.append(
            Violation(
                violation_type=ViolationType.CAPACITY,
                entity_id=vehicle.id,
                details=f"Vehicle {vehicle.id} capacity {vehicle.capacity} exceeded by route demand {total_demand}",
                magnitude=total_demand - vehicle.capacity,
            )
        )
        return None, violations

    state = RouteEvaluationState(
        node=start.node_id,
        time=start.departure_time,
        load=vehicle.capacity,
    )

    evaluated_stops: List[RouteStop] = []
    evaluated_legs: List[RouteLeg] = []
    total_distance_m = 0
    total_wait_s = 0

    for idx, order in enumerate(orders):
        next_state, stop_res, leg_res, viol = step(state, order, ctx, oracle)
        if viol is not None:
            violations.append(viol)
            return None, violations
        assert next_state is not None and stop_res is not None and leg_res is not None

        # Fix sequence numbers
        stop_obj = RouteStop(
            seq=idx + 1,
            order_id=stop_res.order_id,
            arrival_utc=stop_res.arrival_utc,
            wait_s=stop_res.wait_s,
            service_start_utc=stop_res.service_start_utc,
            service_end_utc=stop_res.service_end_utc,
            lateness_s=stop_res.lateness_s,
            sla_cost_paise=stop_res.sla_cost_paise,
            stop_state=StopState.LOCKED if (start.locked_stop == order.id) else StopState.PENDING,
        )
        leg_obj = RouteLeg(
            seq=idx + 1,
            from_node=leg_res.from_node,
            to_node=leg_res.to_node,
            distance_m=leg_res.distance_m,
            depart_utc=leg_res.depart_utc,
            arrive_utc=leg_res.arrive_utc,
            edge_ids=leg_res.edge_ids,
        )

        evaluated_stops.append(stop_obj)
        evaluated_legs.append(leg_obj)
        total_distance_m += leg_obj.distance_m
        total_wait_s += stop_obj.wait_s
        state = next_state

    # Depot return leg if enabled
    return_utc = state.time
    if ctx.depot_return_enabled and orders:
        ret_arrival = oracle.arrival(state.node, ctx.depot_node_id, state.time, ctx)
        if not ret_arrival.is_success:
            violations.append(
                Violation(
                    violation_type=ViolationType.ROAD_PATH,
                    entity_id=vehicle.id,
                    details=f"Depot return path failure: {ret_arrival.message}",
                )
            )
            return None, violations

        return_utc = ret_arrival.arrive_t
        ret_leg = RouteLeg(
            seq=len(orders) + 1,
            from_node=state.node,
            to_node=ctx.depot_node_id,
            distance_m=ret_arrival.distance_m,
            depart_utc=state.time,
            arrive_utc=return_utc,
            edge_ids=ret_arrival.edge_ids,
        )
        evaluated_legs.append(ret_leg)
        total_distance_m += ret_arrival.distance_m

        if ctx.depot_return_deadline is not None and return_utc > ctx.depot_return_deadline:
            violations.append(
                Violation(
                    violation_type=ViolationType.RETURN_DEADLINE,
                    entity_id=vehicle.id,
                    details=f"Depot return deadline exceeded: {return_utc} > {ctx.depot_return_deadline}",
                    magnitude=return_utc - ctx.depot_return_deadline,
                )
            )
            return None, violations

    # Paid duration H_k = vehicle departure through route completion
    paid_duration_s = max(0, return_utc - start.departure_time)

    # Cost calculations (integer paise, round_half_even per route)
    c_dist = line_distance_cost(total_distance_m, ctx.distance_rate_num, ctx.distance_rate_den)
    c_time = line_time_cost(paid_duration_s, ctx.time_rate_num, ctx.time_rate_den)

    # Activation cost: according to Addendum A, charged only if newly activated (not in A_shift)
    is_newly_activated = vehicle.id not in ctx.a_shift
    c_act = vehicle.activation_cost_paise if is_newly_activated else 0

    route = Route(
        vehicle_id=vehicle.id,
        departure_utc=start.departure_time,
        return_utc=return_utc,
        distance_m=total_distance_m,
        paid_seconds=paid_duration_s,
        idle_seconds=total_wait_s,
        distance_cost_paise=c_dist,
        time_cost_paise=c_time,
        activation_cost_paise=c_act,
        stops=tuple(evaluated_stops),
        legs=tuple(evaluated_legs),
    )
    return route, []


def evaluate_plan(
    routes_with_specs: Sequence[Tuple[VehicleSpec, Sequence[Order], RouteStart]],
    all_requested_orders: Sequence[Order],
    ctx: EvaluationContext,
    oracle: RoadOracle,
    sunk_cost_paise: int = 0,
    completed_order_ids: Sequence[str] = (),
) -> Plan:
    """
    Evaluates a candidate multi-vehicle plan against the EvaluationContext.
    """
    all_violations: List[Violation] = []
    built_routes: List[Route] = []
    served_order_ids: Set[str] = set(completed_order_ids)

    for vehicle, order_seq, start in routes_with_specs:
        if not order_seq:
            continue
        route, viols = evaluate_route(vehicle, order_seq, start, ctx, oracle)
        if viols:
            all_violations.extend(viols)
        if route is not None:
            built_routes.append(route)
            for o in order_seq:
                served_order_ids.add(o.id)

    order_map = {o.id: o for o in all_requested_orders}
    unserved_records: List[UnservedOrderRecord] = []
    for o_id, order in order_map.items():
        if o_id not in served_order_ids:
            unserved_records.append(
                UnservedOrderRecord(
                    order_id=o_id,
                    reason=UnservedReason.UNSERVED_BY_OPTIMIZATION_DECISION,
                    detail="Excluded during search/split",
                )
            )

    # Coverage tuple by priority tier: (Critical, Standard, Flexible)
    n_crit = sum(1 for oid in served_order_ids if order_map[oid].priority.value == "CRITICAL")
    n_std = sum(1 for oid in served_order_ids if order_map[oid].priority.value == "STANDARD")
    n_flx = sum(1 for oid in served_order_ids if order_map[oid].priority.value == "FLEXIBLE")
    coverage_tuple = (n_crit, n_std, n_flx)

    # Aggregate ledger components
    total_dist_cost = sum(r.distance_cost_paise for r in built_routes)
    total_time_cost = sum(r.time_cost_paise for r in built_routes)
    total_act_cost = sum(r.activation_cost_paise for r in built_routes)
    total_sla_cost = sum(s.sla_cost_paise for r in built_routes for s in r.stops)
    total_idle_seconds = sum(r.idle_seconds for r in built_routes)

    total_decision_cost = total_dist_cost + total_time_cost + total_act_cost + total_sla_cost
    grand_total_paise = sunk_cost_paise + total_decision_cost

    num_soft_orders = sum(1 for r in built_routes for s in r.stops if s.sla_cost_paise > 0 or order_map[s.order_id].window_type == WindowType.SOFT)
    dev_bound = compute_rounding_deviation_bound(len(built_routes), num_soft_orders)

    ledger = CostLedger(
        distance_cost_paise=total_dist_cost,
        time_cost_paise=total_time_cost,
        activation_cost_paise=total_act_cost,
        sla_cost_paise=total_sla_cost,
        total_paise=grand_total_paise,
        rounding_deviation_bound=dev_bound,
        sunk_cost_paise=sunk_cost_paise,
        remaining_decision_cost_paise=total_decision_cost,
    )

    # Fleet metrics
    used_vids = {r.vehicle_id for r in built_routes}
    m_routes_used = len(built_routes)
    m_new_activated = len(used_vids - set(ctx.a_shift))
    m_shift_activated_after = len(set(ctx.a_shift) | used_vids)

    is_feasible = (len(all_violations) == 0)

    # Sorted served orders for deterministic signature
    sorted_served = tuple(sorted(served_order_ids))

    plan = Plan(
        routes=tuple(built_routes),
        served_orders=sorted_served,
        unserved_orders=tuple(unserved_records),
        ledger=ledger,
        coverage_tuple=coverage_tuple,
        idle_seconds=total_idle_seconds,
        m_routes_used=m_routes_used,
        m_new_activated=m_new_activated,
        m_shift_activated_after=m_shift_activated_after,
        is_feasible=is_feasible,
        violations=tuple(all_violations),
        canonical_hash="",
    )
    return plan
