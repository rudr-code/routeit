from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException

from backend.app.db import Database, get_db
from backend.app.queue import process_single_run
from backend.app.schemas import (
    OptimizationRunCreate,
    OptimizationRunResponse,
    PlanningRequestCreate,
    PlanningRequestResponse,
    RunResultResponse,
)

router = APIRouter(prefix="/v1", tags=["Planning & Optimization"])


@router.post("/planning-requests", response_model=PlanningRequestResponse, status_code=201)
def create_planning_request(req: PlanningRequestCreate, db: Database = Depends(get_db)):
    pr_id = str(uuid.uuid4())
    now_ts = time.time()
    # Fast context hash calculation
    context_data = {
        "depot_id": req.depot_id,
        "orders": sorted(req.order_ids),
        "vehicles": sorted(req.vehicle_ids),
        "planning_mode": req.planning_mode,
        "fleet_mode": req.fleet_mode,
        "profile_kind": req.profile_kind,
        "shift_id": req.shift_id,
        "cost_profile_id": req.cost_profile_id,
        "traffic_scenario_id": req.traffic_scenario_id,
    }
    import hashlib
    c_hash = hashlib.sha256(json.dumps(context_data, sort_keys=True).encode("utf-8")).hexdigest()

    db.execute(
        """
        INSERT INTO planning_request (
            id, shift_id, depot_id, planning_mode, fleet_mode, fleet_max, profile_kind,
            profile_params_json, departure_set_json, cost_profile_id, traffic_scenario_id,
            context_hash, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'VALIDATED', ?)
        """,
        (
            pr_id,
            req.shift_id,
            req.depot_id,
            req.planning_mode,
            req.fleet_mode,
            req.fleet_max,
            req.profile_kind,
            json.dumps(req.profile_params),
            json.dumps(req.departure_set),
            req.cost_profile_id,
            req.traffic_scenario_id,
            c_hash,
            now_ts,
        ),
    )
    db.commit()

    return PlanningRequestResponse(
        planning_request_id=pr_id,
        shift_id=req.shift_id,
        status="VALIDATED",
        context_hash=c_hash,
        warnings=[],
        proofs=[],
    )


@router.post("/optimization-runs", response_model=OptimizationRunResponse, status_code=202)
def start_optimization_run(req: OptimizationRunCreate, db: Database = Depends(get_db)):
    pr = db.fetchone("SELECT * FROM planning_request WHERE id = ?", (req.planning_request_id,))
    if not pr:
        raise HTTPException(status_code=404, detail="Planning request not found")

    run_id = str(uuid.uuid4())
    now_ts = time.time()
    db.execute(
        """
        INSERT INTO optimization_run (
            id, planning_request_id, solver_kind, seed, job_status, created_at
        ) VALUES (?, ?, ?, ?, 'QUEUED', ?)
        """,
        (run_id, req.planning_request_id, req.solver_kind, req.seed, now_ts),
    )
    db.commit()

    # Process synchronously or let queue worker pick it up
    process_single_run(run_id, db)

    return OptimizationRunResponse(run_id=run_id, job_status="QUEUED")


@router.get("/optimization-runs/{id}", response_model=Dict[str, Any])
def get_run_status(id: str, db: Database = Depends(get_db)):
    r = db.fetchone("SELECT * FROM optimization_run WHERE id = ?", (id,))
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")

    plan_row = db.fetchone("SELECT id FROM plan WHERE run_id = ?", (id,))
    plan_id = plan_row["id"] if plan_row else None

    return {
        "run_id": r["id"],
        "job_status": r["job_status"],
        "result_status": r["result_status"],
        "stop_reason": r["stop_reason"],
        "plan_id": plan_id,
        "counters": json.loads(r["counters_json"]) if r["counters_json"] else {},
        "started_at": r["started_at"],
        "finished_at": r["finished_at"],
    }


@router.get("/optimization-runs/{id}/result", response_model=RunResultResponse)
def get_run_result(id: str, db: Database = Depends(get_db)):
    r = db.fetchone("SELECT * FROM optimization_run WHERE id = ?", (id,))
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")

    plan_row = db.fetchone("SELECT * FROM plan WHERE run_id = ?", (id,))
    unserved_rows = db.fetchall("SELECT * FROM unserved_order WHERE plan_id = ?", (plan_row["id"],)) if plan_row else []

    coverage = {}
    if plan_row:
        coverage = {
            "critical": plan_row["coverage_critical"],
            "standard": plan_row["coverage_standard"],
            "flexible": plan_row["coverage_flexible"],
        }

    return RunResultResponse(
        run_id=r["id"],
        job_status=r["job_status"],
        result_status=r["result_status"] or "NO_FEASIBLE_PLAN_FOUND",
        plan_id=plan_row["id"] if plan_row else None,
        stop_reason=r["stop_reason"],
        proof_scope="NONE",
        coverage=coverage,
        unserved=[{"order_id": u["order_id"], "reason": u["reason"], "detail": u["detail"]} for u in unserved_rows],
        flags={},
        context_hash=plan_row["context_hash"] if plan_row else "",
    )


@router.post("/optimization-runs/{id}/cancel", status_code=200)
def cancel_run(id: str, db: Database = Depends(get_db)):
    r = db.fetchone("SELECT * FROM optimization_run WHERE id = ?", (id,))
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")
    db.execute("UPDATE optimization_run SET job_status = 'CANCELLED', stop_reason = 'CANCELLED' WHERE id = ?", (id,))
    db.commit()
    return {"message": "Run cancelled"}
