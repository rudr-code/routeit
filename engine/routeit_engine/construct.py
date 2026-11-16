from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from routeit_engine.contracts import (
    Order,
    Plan,
    PlanningMode,
    RouteStart,
    VehicleSpec,
    WindowType,
)
from routeit_engine.context import EvaluationContext
from routeit_engine.evaluator import evaluate_plan, evaluate_route
from routeit_engine.oracle import RoadOracle
from routeit_engine.policy import select_vehicles


def _build_initial_plan(
    routes_orders: List[List[Order]],
    all_orders: Sequence[Order],
    ctx: EvaluationContext,
    oracle: RoadOracle,
) -> Plan:
    """Helper to convert a list of order lists into a canonical Plan using select_vehicles."""
    non_empty_routes = [r for r in routes_orders if r]
    m = len(non_empty_routes)
    eligible = [v for v in ctx.vehicles if v.available]
    selected_vehicles = select_vehicles(eligible, set(ctx.a_shift), m)

    # Sort route descriptors by (departure second, stop-ID sequence) for canonical tie-break
    routes_with_specs = []
    dep_time = ctx.departure_set[0] if ctx.departure_set else 0

    for idx, r_orders in enumerate(non_empty_routes):
        veh = selected_vehicles[idx]
        start = RouteStart(
            node_id=ctx.depot_node_id,
            departure_time=dep_time,
            initial_load=veh.capacity,
        )
        routes_with_specs.append((veh, tuple(r_orders), start))

    return evaluate_plan(routes_with_specs, all_orders, ctx, oracle)


class CostGreedyConstructor:
    @staticmethod
    def construct(ctx: EvaluationContext, oracle: RoadOracle) -> Plan:
        eligible = [v for v in ctx.vehicles if v.available]
        q_limit = min(len(eligible), ctx.fleet_max) if ctx.fleet_max is not None else len(eligible)
        orders_remaining = list(ctx.orders)
        # Sort orders by window_open, then priority
        orders_remaining.sort(key=lambda o: (o.window_open, 0 if o.priority.value == "CRITICAL" else 1 if o.priority.value == "STANDARD" else 2, o.id))

        routes: List[List[Order]] = []
        dep_time = ctx.departure_set[0] if ctx.departure_set else 0

        while orders_remaining and len(routes) < q_limit:
            current_route: List[Order] = []
            veh = eligible[len(routes)]
            start = RouteStart(node_id=ctx.depot_node_id, departure_time=dep_time, initial_load=veh.capacity)

            improved = True
            while improved and orders_remaining:
                improved = False
                best_order = None
                best_cost = None

                for order in orders_remaining:
                    cand_route = current_route + [order]
                    r_eval, viols = evaluate_route(veh, cand_route, start, ctx, oracle)
                    if not viols and r_eval is not None:
                        if best_cost is None or r_eval.total_cost_paise < best_cost:
                            best_cost = r_eval.total_cost_paise
                            best_order = order

                if best_order is not None:
                    current_route.append(best_order)
                    orders_remaining.remove(best_order)
                    improved = True

            if current_route:
                routes.append(current_route)
            else:
                break

        return _build_initial_plan(routes, ctx.orders, ctx, oracle)


class IdleAwareConstructor:
    @staticmethod
    def construct(ctx: EvaluationContext, oracle: RoadOracle) -> Plan:
        eligible = [v for v in ctx.vehicles if v.available]
        q_limit = min(len(eligible), ctx.fleet_max) if ctx.fleet_max is not None else len(eligible)
        orders_remaining = list(ctx.orders)
        orders_remaining.sort(key=lambda o: (o.window_open, o.id))

        routes: List[List[Order]] = []
        dep_time = ctx.departure_set[0] if ctx.departure_set else 0

        while orders_remaining and len(routes) < q_limit:
            current_route: List[Order] = []
            veh = eligible[len(routes)]
            start = RouteStart(node_id=ctx.depot_node_id, departure_time=dep_time, initial_load=veh.capacity)

            improved = True
            while improved and orders_remaining:
                improved = False
                best_order = None
                best_idle = None

                for order in orders_remaining:
                    cand_route = current_route + [order]
                    r_eval, viols = evaluate_route(veh, cand_route, start, ctx, oracle)
                    if not viols and r_eval is not None:
                        # Prioritize minimal idle waiting W_i
                        if best_idle is None or r_eval.idle_seconds < best_idle:
                            best_idle = r_eval.idle_seconds
                            best_order = order

                if best_order is not None:
                    current_route.append(best_order)
                    orders_remaining.remove(best_order)
                    improved = True

            if current_route:
                routes.append(current_route)
            else:
                break

        return _build_initial_plan(routes, ctx.orders, ctx, oracle)


class Regret2Constructor:
    @staticmethod
    def construct(ctx: EvaluationContext, oracle: RoadOracle) -> Plan:
        eligible = [v for v in ctx.vehicles if v.available]
        q_limit = min(len(eligible), ctx.fleet_max) if ctx.fleet_max is not None else len(eligible)
        orders_remaining = list(ctx.orders)
        dep_time = ctx.departure_set[0] if ctx.departure_set else 0

        # Initialize empty routes up to q_limit (or at least 1)
        routes: List[List[Order]] = [[] for _ in range(max(1, min(len(eligible), 2)))]

        while orders_remaining:
            best_order = None
            best_regret = -1
            best_insert_pos = (0, 0)
            best_cand_cost = None

            for order in orders_remaining:
                insertion_costs: List[Tuple[int, int, int]] = []  # (added_cost, r_idx, pos)
                for r_idx, r in enumerate(routes):
                    veh = eligible[min(r_idx, len(eligible) - 1)]
                    start = RouteStart(node_id=ctx.depot_node_id, departure_time=dep_time, initial_load=veh.capacity)
                    base_eval, _ = evaluate_route(veh, r, start, ctx, oracle)
                    base_cost = base_eval.total_cost_paise if base_eval else 0

                    for pos in range(len(r) + 1):
                        cand_seq = r[:pos] + [order] + r[pos:]
                        cand_eval, viols = evaluate_route(veh, cand_seq, start, ctx, oracle)
                        if not viols and cand_eval is not None:
                            added_cost = cand_eval.total_cost_paise - base_cost
                            insertion_costs.append((added_cost, r_idx, pos))

                if not insertion_costs:
                    continue

                insertion_costs.sort(key=lambda x: x[0])
                if len(insertion_costs) == 1:
                    regret = insertion_costs[0][0]
                else:
                    regret = insertion_costs[1][0] - insertion_costs[0][0]

                if regret > best_regret:
                    best_regret = regret
                    best_order = order
                    best_insert_pos = (insertion_costs[0][1], insertion_costs[0][2])
                    best_cand_cost = insertion_costs[0][0]

            if best_order is not None:
                r_idx, pos = best_insert_pos
                routes[r_idx].insert(pos, best_order)
                orders_remaining.remove(best_order)
            else:
                # Could not insert any remaining order into existing routes
                # Try opening a new route if within q_limit
                if len(routes) < q_limit:
                    routes.append([])
                else:
                    break

        return _build_initial_plan(routes, ctx.orders, ctx, oracle)


class RandomizedFeasibleConstructor:
    @staticmethod
    def construct(ctx: EvaluationContext, oracle: RoadOracle, seed: int = 42) -> Plan:
        rng = random.Random(seed)
        eligible = [v for v in ctx.vehicles if v.available]
        q_limit = min(len(eligible), ctx.fleet_max) if ctx.fleet_max is not None else len(eligible)
        orders_remaining = list(ctx.orders)
        rng.shuffle(orders_remaining)

        routes: List[List[Order]] = []
        dep_time = ctx.departure_set[0] if ctx.departure_set else 0

        while orders_remaining and len(routes) < q_limit:
            current_route: List[Order] = []
            veh = eligible[len(routes)]
            start = RouteStart(node_id=ctx.depot_node_id, departure_time=dep_time, initial_load=veh.capacity)

            improved = True
            while improved and orders_remaining:
                improved = False
                feasible_candidates = []

                for order in orders_remaining:
                    cand_route = current_route + [order]
                    r_eval, viols = evaluate_route(veh, cand_route, start, ctx, oracle)
                    if not viols and r_eval is not None:
                        feasible_candidates.append((order, r_eval.total_cost_paise))

                if feasible_candidates:
                    # Choose randomly among top 3
                    feasible_candidates.sort(key=lambda x: x[1])
                    top_k = feasible_candidates[: min(3, len(feasible_candidates))]
                    chosen_order, _ = rng.choice(top_k)
                    current_route.append(chosen_order)
                    orders_remaining.remove(chosen_order)
                    improved = True

            if current_route:
                routes.append(current_route)
            else:
                break

        return _build_initial_plan(routes, ctx.orders, ctx, oracle)
