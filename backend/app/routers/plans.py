from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException

from backend.app.db import Database, get_db
from backend.app.schemas import (
    CostBreakdownResponse,
    PlanResponse,
    ReplanEventCreate,
    RouteResponse,
    RouteStopResponse,
)

router = APIRouter(prefix="/v1/plans", tags=["Plans"])


@router.get("/{id}", response_model=PlanResponse)
def get_plan(id: str, db: Database = Depends(get_db)):
    plan_row = db.fetchone("SELECT * FROM plan WHERE id = ?", (id,))
    if not plan_row:
        raise HTTPException(status_code=404, detail="Plan not found")

    cost_row = db.fetchone("SELECT * FROM cost_breakdown WHERE plan_id = ?", (id,))
    routes_rows = db.fetchall("SELECT * FROM route WHERE plan_id = ?", (id,))
    unserved_rows = db.fetchall("SELECT * FROM unserved_order WHERE plan_id = ?", (id,))

    routes_out: List[RouteResponse] = []
    for r in routes_rows:
        stops_rows = db.fetchall(
            "SELECT * FROM route_stop WHERE route_id = ? ORDER BY seq ASC",
            (r["id"],),
        )
        stops_out = [
            RouteStopResponse(
                seq=s["seq"],
                order_id=s["order_id"],
                arrival_utc=s["arrival_utc"],
                wait_s=s["wait_s"],
                service_start_utc=s["service_start_utc"],
                service_end_utc=s["service_end_utc"],
                lateness_s=s["lateness_s"],
                sla_cost_paise=s["sla_cost_paise"],
                stop_state=s["stop_state"],
            )
            for s in stops_rows
        ]
        routes_out.append(
            RouteResponse(
                route_id=r["id"],
                vehicle_id=r["vehicle_id"],
                departure_utc=r["departure_utc"],
                return_utc=r["return_utc"],
                distance_m=r["distance_m"],
                paid_seconds=r["paid_seconds"],
                idle_seconds=r["idle_seconds"],
                distance_cost_paise=r["distance_cost_paise"],
                time_cost_paise=r["time_cost_paise"],
                activation_cost_paise=r["activation_cost_paise"],
                total_cost_paise=(
                    r["distance_cost_paise"]
                    + r["time_cost_paise"]
                    + r["activation_cost_paise"]
                    + sum(st.sla_cost_paise for st in stops_out)
                ),
                stops=stops_out,
            )
        )

    cost_bd = CostBreakdownResponse(
        distance_cost_paise=cost_row["distance_cost_paise"] if cost_row else 0,
        time_cost_paise=cost_row["time_cost_paise"] if cost_row else 0,
        activation_cost_paise=cost_row["activation_cost_paise"] if cost_row else 0,
        sla_cost_paise=cost_row["sla_cost_paise"] if cost_row else 0,
        total_paise=cost_row["total_paise"] if cost_row else 0,
        sunk_cost_paise=cost_row["sunk_cost_paise"] if cost_row else 0,
        remaining_decision_cost_paise=cost_row["remaining_decision_cost_paise"] if cost_row else 0,
    )

    return PlanResponse(
        plan_id=plan_row["id"],
        run_id=plan_row["run_id"],
        version=plan_row["version"],
        shift_id=plan_row["shift_id"],
        status=plan_row["status"],
        result_status=plan_row["result_status"],
        context_hash=plan_row["context_hash"],
        coverage={
            "critical": plan_row["coverage_critical"],
            "standard": plan_row["coverage_standard"],
            "flexible": plan_row["coverage_flexible"],
        },
        total_cost_paise=plan_row["total_cost_paise"],
        idle_seconds=plan_row["idle_seconds"],
        active_vehicles=plan_row["active_vehicles"],
        routes=routes_out,
        cost_breakdown=cost_bd,
        unserved=[{"order_id": u["order_id"], "reason": u["reason"], "detail": u["detail"]} for u in unserved_rows],
    )


@router.get("/{id}/cost-breakdown", response_model=CostBreakdownResponse)
def get_plan_cost_breakdown(id: str, db: Database = Depends(get_db)):
    r = db.fetchone("SELECT * FROM cost_breakdown WHERE plan_id = ?", (id,))
    if not r:
        raise HTTPException(status_code=404, detail="Cost breakdown not found")
    return CostBreakdownResponse(
        distance_cost_paise=r["distance_cost_paise"],
        time_cost_paise=r["time_cost_paise"],
        activation_cost_paise=r["activation_cost_paise"],
        sla_cost_paise=r["sla_cost_paise"],
        total_paise=r["total_paise"],
        sunk_cost_paise=r["sunk_cost_paise"],
        remaining_decision_cost_paise=r["remaining_decision_cost_paise"],
    )


@router.get("/{id}/unserved")
def get_plan_unserved(id: str, db: Database = Depends(get_db)):
    rows = db.fetchall("SELECT * FROM unserved_order WHERE plan_id = ?", (id,))
    return [{"order_id": r["order_id"], "reason": r["reason"], "detail": r["detail"]} for r in rows]


@router.post("/{id}/events", status_code=202)
def inject_event(id: str, event: ReplanEventCreate, db: Database = Depends(get_db)):
    plan_row = db.fetchone("SELECT * FROM plan WHERE id = ?", (id,))
    if not plan_row:
        raise HTTPException(status_code=404, detail="Plan not found")

    event_id = str(uuid.uuid4())
    return {
        "replan_event_id": event_id,
        "plan_id": id,
        "state": "RECEIVED",
        "kind": event.kind,
        "occurred_at": event.occurred_at,
    }


@router.post("/{id}/replan", status_code=200)
def trigger_replan(id: str, db: Database = Depends(get_db)):
    plan_row = db.fetchone("SELECT * FROM plan WHERE id = ?", (id,))
    if not plan_row:
        raise HTTPException(status_code=404, detail="Plan not found")

    # Ingest replan trigger and return current status
    return {
        "plan_id": id,
        "status": "REPLANNING_INITIATED",
        "current_version": plan_row["version"],
        "shift_id": plan_row["shift_id"],
    }
