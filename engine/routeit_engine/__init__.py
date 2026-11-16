from __future__ import annotations

import time
import uuid
from typing import Optional

from routeit_engine.contracts import (
    Order,
    Plan,
    PlanningMode,
    ProfileKind,
    ResultStatus,
    RouteStart,
    RunResult,
    SolveRequest,
    StopReason,
)
from routeit_engine.context import EvaluationContext
from routeit_engine.construct import (
    CostGreedyConstructor,
    IdleAwareConstructor,
    RandomizedFeasibleConstructor,
    Regret2Constructor,
)
from routeit_engine.dynamic import (
    AdoptionGate,
    AdoptionResult,
    OperationalSnapshot,
)
from routeit_engine.evaluator import evaluate_plan
from routeit_engine.gatekeeper import Gatekeeper
from routeit_engine.oracle import RoadOracle
from routeit_engine.policy import (
    Calibration,
    calibrate_balanced_profile,
    compare_plans,
)
from routeit_engine.search.alns import ALNSStrategy
from routeit_engine.search.ga import GAStrategy
from routeit_engine.search.ls import LocalSearch
from routeit_engine.search.qpso import QPSOStrategy
from routeit_engine.split import SplitDecoder
from routeit_engine.telemetry import WorkCounters


__version__ = "5.0.0"


def solve(
    req: SolveRequest,
    oracle: RoadOracle,
    solver_kind: str = "ALNS",
) -> RunResult:
    start_time = time.perf_counter()
    run_id = str(uuid.uuid4())
    counters = WorkCounters()

    # Step 1: Gatekeeper validation
    precheck = Gatekeeper.validate_and_precheck(req)
    if not precheck.is_valid:
        return RunResult(
            run_id=run_id,
            context_hash="",
            result_status=precheck.status,
            stop_reason=StopReason.CONVERGED_PROVEN_EXACT if precheck.status == ResultStatus.PROVEN_INFEASIBLE else None,
            proof_scope="CAPACITY_LOWER_BOUND_PROOF" if precheck.status == ResultStatus.PROVEN_INFEASIBLE else "NONE",
            error_message=precheck.message,
        )

    # Step 2: Build EvaluationContext
    ctx = EvaluationContext(
        orders=req.orders,
        vehicles=req.vehicles,
        depot_node_id=req.depot_node_id,
        depot_return_enabled=req.depot_return_enabled,
        depot_return_deadline=req.depot_return_deadline,
        planning_mode=req.planning_mode.value,
        fleet_mode=req.fleet_mode.value,
        fleet_max=req.fleet_max,
        profile_kind=req.profile_kind.value,
        profile_params=req.profile_params,
        departure_set=req.departure_set,
        distance_rate_num=req.distance_rate_num,
        distance_rate_den=req.distance_rate_den,
        time_rate_num=req.time_rate_num,
        time_rate_den=req.time_rate_den,
        rounding_version=req.rounding_version,
        shift_id=req.shift_id,
        a_shift=req.a_shift,
        m_shift_activated_max=req.m_shift_activated_max,
    )
    c_hash = ctx.context_hash()

    # Step 3: Run constructors to build initial seed pool
    seed_plans = [
        CostGreedyConstructor.construct(ctx, oracle),
        IdleAwareConstructor.construct(ctx, oracle),
        Regret2Constructor.construct(ctx, oracle),
        RandomizedFeasibleConstructor.construct(ctx, oracle, seed=req.seed),
    ]
    counters.candidates_generated += len(seed_plans)
    counters.canonical_evaluations += len(seed_plans)

    feasible_seeds = [p for p in seed_plans if p.is_feasible]

    # Calibration for Balanced profile
    calibration: Optional[Calibration] = None
    if req.profile_kind == ProfileKind.BALANCED:
        calibration = calibrate_balanced_profile(seed_plans)

    # Driver Productivity handling (two-pass)
    c_cap: Optional[int] = None
    if req.profile_kind == ProfileKind.DRIVER_PRODUCTIVITY:
        if not feasible_seeds:
            return RunResult(
                run_id=run_id,
                context_hash=c_hash,
                result_status=ResultStatus.PROFILE_REFERENCE_UNAVAILABLE,
                error_message="No feasible reference seed found for Driver Productivity profile",
            )
        # Pass 1: find lowest cost among feasible seeds
        c_ref = min(p.c_dec for p in feasible_seeds)
        delta_c_max = int(req.profile_params.get("delta_c_max_paise", 0))
        if delta_c_max < 0:
            return RunResult(
                run_id=run_id,
                context_hash=c_hash,
                result_status=ResultStatus.INVALID_CONFIG,
                error_message="delta_c_max_paise cannot be negative",
            )
        c_cap = c_ref + delta_c_max

    # Step 4: Search method execution
    order_map = {o.id: o for o in req.orders}
    best_candidate: Optional[Plan] = None

    for sp in seed_plans:
        if best_candidate is None:
            best_candidate = sp
        else:
            if compare_plans(sp, best_candidate, req.planning_mode, req.profile_kind, order_map, calibration, c_cap) < 0:
                best_candidate = sp

    solver_upper = solver_kind.upper()
    if "QPSO" in solver_upper and feasible_seeds:
        best_candidate = QPSOStrategy.solve(
            req.orders,
            ctx,
            oracle,
            seed_plans=seed_plans,
            calibration=calibration,
            c_cap=c_cap,
            pop_size=15,
            generations=15,
            seed=req.seed,
        )
    elif "GA" in solver_upper:
        best_candidate = GAStrategy.solve(
            req.orders,
            ctx,
            oracle,
            seed_plans=seed_plans,
            calibration=calibration,
            c_cap=c_cap,
            pop_size=15,
            generations=15,
            seed=req.seed,
        )
    elif "ALNS" in solver_upper:
        best_candidate = ALNSStrategy.solve(
            best_candidate,
            ctx,
            oracle,
            calibration=calibration,
            c_cap=c_cap,
            iterations=50,
            seed=req.seed,
        )
    else:
        # Local search on best constructor seed
        best_candidate = LocalSearch.improve(
            best_candidate,
            ctx,
            oracle,
            calibration=calibration,
            c_cap=c_cap,
            max_iterations=30,
        )

    # Step 5: Final certification
    is_complete_mode = (req.planning_mode == PlanningMode.COMPLETE)
    if is_complete_mode and len(best_candidate.unserved_orders) > 0:
        return RunResult(
            run_id=run_id,
            context_hash=c_hash,
            result_status=ResultStatus.NO_FEASIBLE_PLAN_FOUND,
            stop_reason=StopReason.EVALUATION_BUDGET,
            plan=best_candidate,
            work_counters=counters.to_dict(),
            error_message="Complete mode: could not serve all required orders feasibly",
        )

    if not best_candidate.is_feasible:
        return RunResult(
            run_id=run_id,
            context_hash=c_hash,
            result_status=ResultStatus.NO_FEASIBLE_PLAN_FOUND,
            stop_reason=StopReason.EVALUATION_BUDGET,
            plan=best_candidate,
            work_counters=counters.to_dict(),
            error_message="No feasible plan found satisfying all constraints",
        )

    if req.profile_kind == ProfileKind.DRIVER_PRODUCTIVITY and c_cap is not None:
        if best_candidate.c_dec > c_cap:
            return RunResult(
                run_id=run_id,
                context_hash=c_hash,
                result_status=ResultStatus.PROFILE_CAP_UNSATISFIED,
                stop_reason=StopReason.EVALUATION_BUDGET,
                plan=best_candidate,
                work_counters=counters.to_dict(),
                error_message=f"No plan found under cost cap {c_cap}",
            )

    result_status = (
        ResultStatus.COMPLETE_FEASIBLE
        if len(best_candidate.unserved_orders) == 0
        else ResultStatus.PARTIAL_FEASIBLE
    )

    return RunResult(
        run_id=run_id,
        context_hash=c_hash,
        result_status=result_status,
        stop_reason=StopReason.EVALUATION_BUDGET,
        plan=best_candidate,
        proof_scope="NONE",
        flags={
            "c_ref_is_upper_bound": True if req.profile_kind == ProfileKind.DRIVER_PRODUCTIVITY else False,
            "low_calibration_quality": calibration.low_quality if calibration else False,
        },
        work_counters=counters.to_dict(),
    )


def replan(
    current_plan: Plan,
    snapshot: OperationalSnapshot,
    req: SolveRequest,
    oracle: RoadOracle,
) -> AdoptionResult:
    """
    Dynamic replanning pipeline:
    Solves remaining work under OperationalSnapshot and applies AdoptionGate.
    """
    # Exclude completed orders from the solve request
    pending_orders = tuple(o for o in req.orders if o.id not in snapshot.completed_order_ids)
    replan_req = SolveRequest(
        orders=pending_orders,
        vehicles=req.vehicles,
        depot_node_id=req.depot_node_id,
        depot_return_enabled=req.depot_return_enabled,
        depot_return_deadline=req.depot_return_deadline,
        planning_mode=req.planning_mode,
        fleet_mode=req.fleet_mode,
        fleet_max=req.fleet_max,
        profile_kind=req.profile_kind,
        profile_params=req.profile_params,
        departure_set=req.departure_set,
        distance_rate_num=req.distance_rate_num,
        distance_rate_den=req.distance_rate_den,
        time_rate_num=req.time_rate_num,
        time_rate_den=req.time_rate_den,
        rounding_version=req.rounding_version,
        shift_id=req.shift_id,
        a_shift=req.a_shift,
        m_shift_activated_max=req.m_shift_activated_max,
    )

    candidate_res = solve(replan_req, oracle, solver_kind="ALNS")
    if candidate_res.plan is None:
        return AdoptionResult(
            adopted=False,
            status=ResultStatus.NO_FEASIBLE_PLAN_FOUND,
            message=candidate_res.error_message or "Replanning failed to generate candidate plan",
        )

    # Re-evaluate candidate with sunk costs attached
    order_map = {o.id: o for o in req.orders}
    routes_with_specs = []
    dep_time = req.departure_set[0] if req.departure_set else 0
    for r in candidate_res.plan.routes:
        veh = next(v for v in req.vehicles if v.id == r.vehicle_id)
        start = RouteStart(node_id=req.depot_node_id, departure_time=dep_time, initial_load=veh.capacity)
        r_orders = tuple(order_map[s.order_id] for s in r.stops)
        routes_with_specs.append((veh, r_orders, start))

    ctx = EvaluationContext(
        orders=req.orders,
        vehicles=req.vehicles,
        depot_node_id=req.depot_node_id,
        depot_return_enabled=req.depot_return_enabled,
        depot_return_deadline=req.depot_return_deadline,
        planning_mode=req.planning_mode.value,
        fleet_mode=req.fleet_mode.value,
        fleet_max=req.fleet_max,
        profile_kind=req.profile_kind.value,
        profile_params=req.profile_params,
        departure_set=req.departure_set,
        distance_rate_num=req.distance_rate_num,
        distance_rate_den=req.distance_rate_den,
        time_rate_num=req.time_rate_num,
        time_rate_den=req.time_rate_den,
        rounding_version=req.rounding_version,
        shift_id=req.shift_id,
        a_shift=req.a_shift,
        m_shift_activated_max=req.m_shift_activated_max,
    )

    full_candidate_plan = evaluate_plan(
        routes_with_specs,
        req.orders,
        ctx,
        oracle,
        sunk_cost_paise=snapshot.sunk_cost_paise,
        completed_order_ids=snapshot.completed_order_ids,
    )

    return AdoptionGate.evaluate_and_adopt(
        current_plan=current_plan,
        candidate_plan=full_candidate_plan,
        snapshot=snapshot,
        expected_snapshot_version=snapshot.snapshot_version,
        ctx=ctx,
    )
