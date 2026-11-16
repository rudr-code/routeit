from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from routeit_engine.contracts import (
    Order,
    Plan,
    PlanningMode,
    ProfileKind,
    RouteStart,
    VehicleSpec,
)
from routeit_engine.context import EvaluationContext
from routeit_engine.evaluator import evaluate_plan
from routeit_engine.oracle import RoadOracle
from routeit_engine.policy import Calibration, compare_plans, select_vehicles


def _reconstruct_plan(
    routes_orders: List[List[Order]],
    all_orders: Sequence[Order],
    ctx: EvaluationContext,
    oracle: RoadOracle,
) -> Plan:
    non_empty = [r for r in routes_orders if r]
    m = len(non_empty)
    eligible = [v for v in ctx.vehicles if v.available]
    selected_vehs = select_vehicles(eligible, set(ctx.a_shift), m)
    dep_time = ctx.departure_set[0] if ctx.departure_set else 0

    routes_with_specs = []
    for idx, r in enumerate(non_empty):
        veh = selected_vehs[idx]
        start = RouteStart(node_id=ctx.depot_node_id, departure_time=dep_time, initial_load=veh.capacity)
        routes_with_specs.append((veh, tuple(r), start))

    return evaluate_plan(routes_with_specs, all_orders, ctx, oracle)


class LocalSearch:
    @staticmethod
    def improve(
        plan: Plan,
        ctx: EvaluationContext,
        oracle: RoadOracle,
        calibration: Optional[Calibration] = None,
        c_cap: Optional[int] = None,
        max_iterations: int = 50,
    ) -> Plan:
        order_map = {o.id: o for o in ctx.orders}
        best_plan = plan
        routes_orders: List[List[Order]] = [
            [order_map[s.order_id] for s in r.stops] for r in plan.routes
        ]

        improved = True
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            # 1. Intra-route 2-opt
            for r_idx in range(len(routes_orders)):
                r = routes_orders[r_idx]
                if len(r) < 4:
                    continue
                for i in range(len(r) - 1):
                    for j in range(i + 2, len(r) + 1):
                        new_r = r[:i] + list(reversed(r[i:j])) + r[j:]
                        new_routes = [list(ro) for ro in routes_orders]
                        new_routes[r_idx] = new_r
                        cand = _reconstruct_plan(new_routes, ctx.orders, ctx, oracle)
                        if cand.is_feasible:
                            if compare_plans(cand, best_plan, PlanningMode(ctx.planning_mode), ProfileKind(ctx.profile_kind), order_map, calibration, c_cap) < 0:
                                best_plan = cand
                                routes_orders = new_routes
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

            if improved:
                continue

            # 2. Inter-route Relocate
            if len(routes_orders) >= 2:
                for r1 in range(len(routes_orders)):
                    for r2 in range(len(routes_orders)):
                        if r1 == r2 or not routes_orders[r1]:
                            continue
                        for i in range(len(routes_orders[r1])):
                            target_order = routes_orders[r1][i]
                            for j in range(len(routes_orders[r2]) + 1):
                                new_routes = [list(ro) for ro in routes_orders]
                                new_routes[r1].pop(i)
                                new_routes[r2].insert(j, target_order)
                                cand = _reconstruct_plan(new_routes, ctx.orders, ctx, oracle)
                                if cand.is_feasible:
                                    if compare_plans(cand, best_plan, PlanningMode(ctx.planning_mode), ProfileKind(ctx.profile_kind), order_map, calibration, c_cap) < 0:
                                        best_plan = cand
                                        routes_orders = new_routes
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

            if improved:
                continue

            # 3. Inter-route Swap
            if len(routes_orders) >= 2:
                for r1 in range(len(routes_orders)):
                    for r2 in range(r1 + 1, len(routes_orders)):
                        for i in range(len(routes_orders[r1])):
                            for j in range(len(routes_orders[r2])):
                                new_routes = [list(ro) for ro in routes_orders]
                                new_routes[r1][i], new_routes[r2][j] = new_routes[r2][j], new_routes[r1][i]
                                cand = _reconstruct_plan(new_routes, ctx.orders, ctx, oracle)
                                if cand.is_feasible:
                                    if compare_plans(cand, best_plan, PlanningMode(ctx.planning_mode), ProfileKind(ctx.profile_kind), order_map, calibration, c_cap) < 0:
                                        best_plan = cand
                                        routes_orders = new_routes
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

        return best_plan
