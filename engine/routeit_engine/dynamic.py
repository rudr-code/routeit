from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from routeit_engine.contracts import (
    CostLedger,
    Order,
    Plan,
    PlanningMode,
    ProfileKind,
    ResultStatus,
    Route,
    RouteStart,
    StopState,
    UnservedOrderRecord,
    VehicleSpec,
)
from routeit_engine.context import EvaluationContext
from routeit_engine.evaluator import evaluate_plan, evaluate_route
from routeit_engine.oracle import RoadOracle
from routeit_engine.policy import Calibration, compare_plans, select_vehicles


class VehicleActivityState(str, Enum):
    DEPOT_IDLE = "DEPOT_IDLE"
    EN_ROUTE = "EN_ROUTE"
    AT_STOP = "AT_STOP"
    RETURNING = "RETURNING"
    COMPLETED = "COMPLETED"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class VehicleOperationalState:
    vehicle_id: str
    activity: VehicleActivityState
    current_node: str
    current_time_utc: int
    residual_load: int
    locked_stop_order_id: Optional[str] = None
    locked_stop_departure_utc: Optional[int] = None


@dataclass(frozen=True)
class OperationalSnapshot:
    snapshot_id: str
    snapshot_version: int
    t0_utc: int
    traffic_scenario_id: str
    scenario_revision: int
    vehicle_states: Dict[str, VehicleOperationalState]
    completed_order_ids: Tuple[str, ...]
    sunk_distance_m: int = 0
    sunk_paid_seconds: int = 0
    sunk_sla_paise: int = 0
    sunk_cost_paise: int = 0


@dataclass(frozen=True)
class AdoptionResult:
    adopted: bool
    status: ResultStatus
    message: str = ""
    new_plan: Optional[Plan] = None
    new_snapshot_version: int = 0
    newly_posted_vehicle_ids: Tuple[str, ...] = ()


class AdoptionGate:
    @staticmethod
    def evaluate_and_adopt(
        current_plan: Plan,
        candidate_plan: Plan,
        snapshot: OperationalSnapshot,
        expected_snapshot_version: int,
        ctx: EvaluationContext,
        calibration: Optional[Calibration] = None,
        c_cap: Optional[int] = None,
        theta_money_paise: int = 0,
    ) -> AdoptionResult:
        # Step 1: Check snapshot version concurrency
        if snapshot.snapshot_version != expected_snapshot_version:
            return AdoptionResult(
                adopted=False,
                status=ResultStatus.NO_FEASIBLE_PLAN_FOUND,
                message=f"Snapshot version conflict: expected {expected_snapshot_version}, got {snapshot.snapshot_version}",
            )

        # Step 2: Validate that completed work is strictly preserved
        completed_set = set(snapshot.completed_order_ids)
        cand_served_set = set(candidate_plan.served_orders)
        if not completed_set.issubset(cand_served_set):
            return AdoptionResult(
                adopted=False,
                status=ResultStatus.NO_FEASIBLE_PLAN_FOUND,
                message="Candidate plan violates invariant: completed orders were dropped",
            )

        # Step 3: Validate candidate passes all hard constraints
        if not candidate_plan.is_feasible:
            return AdoptionResult(
                adopted=False,
                status=ResultStatus.NO_FEASIBLE_PLAN_FOUND,
                message=f"Candidate plan failed hard constraints: {[v.details for v in candidate_plan.violations]}",
            )

        # Step 4: Validate cap in Driver Productivity
        if ctx.profile_kind == ProfileKind.DRIVER_PRODUCTIVITY and c_cap is not None:
            if candidate_plan.c_dec > c_cap:
                return AdoptionResult(
                    adopted=False,
                    status=ResultStatus.PROFILE_CAP_UNSATISFIED,
                    message=f"Candidate plan cost {candidate_plan.c_dec} exceeds cap {c_cap}",
                )

        # Step 5: Check stability / hysteresis improvement
        order_map = {o.id: o for o in ctx.orders}
        cmp_res = compare_plans(
            candidate_plan,
            current_plan,
            PlanningMode(ctx.planning_mode),
            ProfileKind(ctx.profile_kind),
            order_map,
            calibration,
            c_cap,
        )

        if cmp_res >= 0:
            return AdoptionResult(
                adopted=False,
                status=ResultStatus.COMPLETE_FEASIBLE,
                message="Candidate does not strictly improve upon current plan under policy comparator",
                new_plan=current_plan,
                new_snapshot_version=snapshot.snapshot_version,
            )

        # Step 6: Identify newly activated vehicle IDs to post to shift ledger
        cand_active_vids = {r.vehicle_id for r in candidate_plan.routes}
        newly_activated = tuple(sorted(list(cand_active_vids - set(ctx.a_shift))))

        return AdoptionResult(
            adopted=True,
            status=ResultStatus.COMPLETE_FEASIBLE if len(candidate_plan.unserved_orders) == 0 else ResultStatus.PARTIAL_FEASIBLE,
            message="Plan successfully adopted",
            new_plan=candidate_plan,
            new_snapshot_version=snapshot.snapshot_version + 1,
            newly_posted_vehicle_ids=newly_activated,
        )
