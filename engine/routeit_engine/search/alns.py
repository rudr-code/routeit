from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from routeit_engine.contracts import (
    Order,
    Plan,
    PlanningMode,
    PriorityTier,
    ProfileKind,
    RouteStart,
    VehicleSpec,
)
from routeit_engine.context import EvaluationContext
from routeit_engine.evaluator import evaluate_plan, evaluate_route
from routeit_engine.oracle import RoadOracle
from routeit_engine.policy import Calibration, compare_plans, select_vehicles
from routeit_engine.search.ls import _reconstruct_plan


class ALNSStrategy:
    @staticmethod
    def solve(
        initial_plan: Plan,
        ctx: EvaluationContext,
        oracle: RoadOracle,
        calibration: Optional[Calibration] = None,
        c_cap: Optional[int] = None,
        iterations: int = 100,
        seed: int = 42,
    ) -> Plan:
        rng = random.Random(seed)
        order_map = {o.id: o for o in ctx.orders}
        best_plan = initial_plan
        current_plan = initial_plan

        routes_orders: List[List[Order]] = [
            [order_map[s.order_id] for s in r.stops] for r in current_plan.routes
        ]

        # Weights for destroy operators
        destroy_weights = [1.0, 1.0, 1.0]

        for it in range(iterations):
            # Select destroy operator
            total_w = sum(destroy_weights)
            r_pick = rng.uniform(0, total_w)
            op_idx = 0
            acc = 0.0
            for idx, w in enumerate(destroy_weights):
                acc += w
                if r_pick <= acc:
                    op_idx = idx
                    break

            # Clone routes
            working_routes = [list(r) for r in routes_orders]
            all_served = [o for r in working_routes for o in r]
            if not all_served:
                continue

            num_to_destroy = min(len(all_served), max(1, int(len(all_served) * 0.3)))
            removed: List[Order] = []

            if op_idx == 0:
                # Random destroy
                rng.shuffle(all_served)
                removed = all_served[:num_to_destroy]
            elif op_idx == 1:
                # Worst cost destroy: order by demand or service time as heuristic proxy
                sorted_by_impact = sorted(all_served, key=lambda o: (o.demand, o.service_seconds), reverse=True)
                removed = sorted_by_impact[:num_to_destroy]
            else:
                # Priority targeted destroy: remove flexible/standard first
                sorted_by_prio = sorted(
                    all_served,
                    key=lambda o: (
                        0 if o.priority == PriorityTier.FLEXIBLE else 1 if o.priority == PriorityTier.STANDARD else 2,
                        o.id,
                    ),
                )
                removed = sorted_by_prio[:num_to_destroy]

            rem_ids = {o.id for o in removed}
            working_routes = [[o for o in r if o.id not in rem_ids] for r in working_routes]

            # Repair: Greedy best insertion
            eligible = [v for v in ctx.vehicles if v.available]
            dep_time = ctx.departure_set[0] if ctx.departure_set else 0

            for o in removed:
                best_ins = None
                best_add_cost = None

                for r_idx in range(len(working_routes)):
                    r = working_routes[r_idx]
                    veh = eligible[min(r_idx, len(eligible) - 1)]
                    start = RouteStart(node_id=ctx.depot_node_id, departure_time=dep_time, initial_load=veh.capacity)
                    base_eval, _ = evaluate_route(veh, r, start, ctx, oracle)
                    base_cost = base_eval.total_cost_paise if base_eval else 0

                    for pos in range(len(r) + 1):
                        cand_seq = r[:pos] + [o] + r[pos:]
                        cand_eval, viols = evaluate_route(veh, cand_seq, start, ctx, oracle)
                        if not viols and cand_eval is not None:
                            added = cand_eval.total_cost_paise - base_cost
                            if best_add_cost is None or added < best_add_cost:
                                best_add_cost = added
                                best_ins = (r_idx, pos)

                if best_ins is not None:
                    working_routes[best_ins[0]].insert(best_ins[1], o)

            cand_plan = _reconstruct_plan(working_routes, ctx.orders, ctx, oracle)
            if cand_plan.is_feasible:
                # Update current if improving
                if compare_plans(cand_plan, current_plan, PlanningMode(ctx.planning_mode), ProfileKind(ctx.profile_kind), order_map, calibration, c_cap) < 0:
                    current_plan = cand_plan
                    routes_orders = working_routes
                    destroy_weights[op_idx] += 1.0

                    if compare_plans(cand_plan, best_plan, PlanningMode(ctx.planning_mode), ProfileKind(ctx.profile_kind), order_map, calibration, c_cap) < 0:
                        best_plan = cand_plan
                        destroy_weights[op_idx] += 2.0

        return best_plan
