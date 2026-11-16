from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from routeit_engine.contracts import (
    Order,
    PlanningMode,
    ResultStatus,
    SolveRequest,
    UnservedOrderRecord,
    UnservedReason,
    VehicleSpec,
)


@dataclass(frozen=True)
class PrecheckResult:
    is_valid: bool
    status: ResultStatus = ResultStatus.COMPLETE_FEASIBLE
    message: str = ""
    unserviceable_orders: Tuple[UnservedOrderRecord, ...] = ()
    fleet_capacity_lower_bound: int = 0


class Gatekeeper:
    @staticmethod
    def validate_and_precheck(req: SolveRequest) -> PrecheckResult:
        # Check rates
        if (
            req.distance_rate_num < 0
            or req.distance_rate_den <= 0
            or req.time_rate_num < 0
            or req.time_rate_den <= 0
        ):
            return PrecheckResult(
                is_valid=False,
                status=ResultStatus.INVALID_CONFIG,
                message="Distance or time rate is negative or denominator <= 0",
            )

        # Check vehicle list
        if not req.vehicles:
            return PrecheckResult(
                is_valid=False,
                status=ResultStatus.INVALID_INPUT,
                message="Vehicle pool is empty",
            )

        v_ids = set()
        for v in req.vehicles:
            if v.id in v_ids:
                return PrecheckResult(
                    is_valid=False,
                    status=ResultStatus.INVALID_INPUT,
                    message=f"Duplicate vehicle ID: {v.id}",
                )
            v_ids.add(v.id)
            if v.capacity <= 0:
                return PrecheckResult(
                    is_valid=False,
                    status=ResultStatus.INVALID_INPUT,
                    message=f"Vehicle {v.id} has non-positive capacity {v.capacity}",
                )
            if v.activation_cost_paise < 0:
                return PrecheckResult(
                    is_valid=False,
                    status=ResultStatus.INVALID_CONFIG,
                    message=f"Vehicle {v.id} has negative activation cost",
                )

        # Check order list
        order_ids = set()
        max_v_capacity = max(v.capacity for v in req.vehicles)
        unserviceable: List[UnservedOrderRecord] = []

        for o in req.orders:
            if o.id in order_ids:
                return PrecheckResult(
                    is_valid=False,
                    status=ResultStatus.INVALID_INPUT,
                    message=f"Duplicate order ID: {o.id}",
                )
            order_ids.add(o.id)

            if o.window_open > o.window_close:
                return PrecheckResult(
                    is_valid=False,
                    status=ResultStatus.INVALID_INPUT,
                    message=f"Order {o.id} has inconsistent window: open {o.window_open} > close {o.window_close}",
                )
            if o.demand < 0 or o.service_seconds < 0:
                return PrecheckResult(
                    is_valid=False,
                    status=ResultStatus.INVALID_INPUT,
                    message=f"Order {o.id} has negative demand or service seconds",
                )
            if o.sla_rate_num < 0 or o.sla_rate_den <= 0:
                return PrecheckResult(
                    is_valid=False,
                    status=ResultStatus.INVALID_CONFIG,
                    message=f"Order {o.id} has invalid SLA rate: {o.sla_rate_num}/{o.sla_rate_den}",
                )

            # Sound proof: individual capacity infeasibility
            if o.demand > max_v_capacity:
                unserviceable.append(
                    UnservedOrderRecord(
                        order_id=o.id,
                        reason=UnservedReason.PROVEN_UNSERVICEABLE_CAPACITY,
                        detail=f"Order demand {o.demand} exceeds maximum vehicle capacity {max_v_capacity}",
                    )
                )

        # In Complete Mode, if any order exceeds vehicle capacity, the entire problem is proven infeasible
        if req.planning_mode == PlanningMode.COMPLETE and unserviceable:
            return PrecheckResult(
                is_valid=False,
                status=ResultStatus.PROVEN_INFEASIBLE,
                message=f"Order(s) {[u.order_id for u in unserviceable]} demand exceeds fleet capacity",
                unserviceable_orders=tuple(unserviceable),
            )

        # Check shift activation ceiling conflict (ACT-13)
        if req.m_shift_activated_max is not None:
            if len(req.a_shift) > req.m_shift_activated_max:
                return PrecheckResult(
                    is_valid=False,
                    status=ResultStatus.SHIFT_FLEET_POLICY_CONFLICT,
                    message=f"Shift activation count {len(req.a_shift)} exceeds ceiling {req.m_shift_activated_max}",
                )

        # Fleet lower bound proof
        eligible_vehicles = [v for v in req.vehicles if v.available]
        q = min(len(eligible_vehicles), req.fleet_max) if req.fleet_max is not None else len(eligible_vehicles)

        total_demand = sum(o.demand for o in req.orders if o.id not in {u.order_id for u in unserviceable})
        lb_fleet = math.ceil(total_demand / max_v_capacity) if max_v_capacity > 0 else 0
        lb_half = sum(1 for o in req.orders if o.demand > max_v_capacity / 2)
        lb_total = max(lb_fleet, lb_half)

        if req.planning_mode == PlanningMode.COMPLETE and lb_total > q:
            return PrecheckResult(
                is_valid=False,
                status=ResultStatus.PROVEN_INFEASIBLE,
                message=f"Fleet lower bound {lb_total} exceeds available fleet limit {q}",
                fleet_capacity_lower_bound=lb_total,
            )

        return PrecheckResult(
            is_valid=True,
            status=ResultStatus.COMPLETE_FEASIBLE,
            unserviceable_orders=tuple(unserviceable),
            fleet_capacity_lower_bound=lb_total,
        )
