from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Set

from backend.app.db import Database, get_db
from routeit_engine import solve
from routeit_engine.contracts import (
    FleetMode,
    Order,
    PlanningMode,
    PriorityTier,
    ProfileKind,
    ResultStatus,
    SolveRequest,
    VehicleSpec,
    WindowType,
)
from routeit_roads.euclidean import EuclideanOracle


logger = logging.getLogger(__name__)


def process_single_run(run_id: str, db: Database):
    with db._lock:
        run_row = db.fetchone("SELECT * FROM optimization_run WHERE id = ?", (run_id,))
        if not run_row or run_row["job_status"] != "QUEUED":
            return

        db.execute(
            "UPDATE optimization_run SET job_status = 'RUNNING', started_at = ? WHERE id = ?",
            (time.time(), run_id),
        )
        db.commit()

        pr_row = db.fetchone("SELECT * FROM planning_request WHERE id = ?", (run_row["planning_request_id"],))
        if not pr_row:
            db.execute("UPDATE optimization_run SET job_status = 'FAILED' WHERE id = ?", (run_id,))
            db.commit()
            return

    shift_id = pr_row["shift_id"]

    # 1. Read existing shift activation ledger (Addendum A.2)
    with db._lock:
        ledger_rows = db.fetchall(
            "SELECT vehicle_id FROM shift_activation_ledger WHERE shift_id = ?",
            (shift_id,),
        )
        a_shift = tuple(r["vehicle_id"] for r in ledger_rows)

        # 2. Fetch depot
        depot_row = db.fetchone("SELECT * FROM depot WHERE id = ?", (pr_row["depot_id"],))
        # Fetch cost profile
        cost_prof = db.fetchone("SELECT * FROM cost_profile WHERE id = ?", (pr_row["cost_profile_id"],))

        # Fetch orders
        order_rows = db.fetchall("SELECT * FROM delivery_order WHERE status != 'CANCELLED'")
        cust_rows = {c["id"]: c for c in db.fetchall("SELECT * FROM customer")}

        # Fetch vehicles
        veh_rows = db.fetchall("SELECT * FROM vehicle WHERE available = 1")

    if not depot_row or not cost_prof or not veh_rows:
        with db._lock:
            db.execute("UPDATE optimization_run SET job_status = 'FAILED' WHERE id = ?", (run_id,))
            db.commit()
        return

    # Build node coordinates map
    coords: Dict[str, tuple[float, float]] = {
        depot_row["snapped_node_id"]: (depot_row["lat"], depot_row["lon"])
    }

    orders_list: List[Order] = []
    for o in order_rows:
        cid = o["customer_id"]
        c = cust_rows.get(cid)
        if c:
            node_id = c["snapped_node_id"]
            coords[node_id] = (c["lat"], c["lon"])
            orders_list.append(
                Order(
                    id=o["id"],
                    node_id=node_id,
                    demand=o["demand_units"],
                    service_seconds=o["service_seconds"],
                    window_open=o["window_open"],
                    window_close=o["window_close"],
                    window_type=WindowType(o["window_type"]),
                    priority=PriorityTier(o["priority"]),
                    sla_rate_num=o["sla_rate_num"],
                    sla_rate_den=o["sla_rate_den"],
                )
            )

    vehicles_list: List[VehicleSpec] = [
        VehicleSpec(
            id=v["id"],
            capacity=v["capacity"],
            available=bool(v["available"]),
            activation_cost_paise=v["activation_cost_paise"],
            profile=v["profile"],
        )
        for v in veh_rows
    ]

    departure_set = tuple(json.loads(pr_row["departure_set_json"]))
    profile_params = json.loads(pr_row["profile_params_json"])

    solve_req = SolveRequest(
        orders=tuple(orders_list),
        vehicles=tuple(vehicles_list),
        depot_node_id=depot_row["snapped_node_id"],
        depot_return_enabled=True,
        planning_mode=PlanningMode(pr_row["planning_mode"]),
        fleet_mode=FleetMode(pr_row["fleet_mode"]),
        fleet_max=pr_row["fleet_max"],
        profile_kind=ProfileKind(pr_row["profile_kind"]),
        profile_params=profile_params,
        departure_set=departure_set,
        distance_rate_num=cost_prof["distance_rate_num"],
        distance_rate_den=cost_prof["distance_rate_den"],
        time_rate_num=cost_prof["time_rate_num"],
        time_rate_den=cost_prof["time_rate_den"],
        rounding_version=cost_prof["rounding_policy_version"],
        shift_id=shift_id,
        a_shift=a_shift,
        seed=run_row["seed"],
    )

    # Use Euclidean road oracle
    oracle = EuclideanOracle(coords, speed_mps=10.0, distance_scale=1000.0)

    try:
        run_res = solve(solve_req, oracle, solver_kind=run_row["solver_kind"])
    except Exception as exc:
        logger.exception("Error executing solve()")
        with db._lock:
            db.execute(
                "UPDATE optimization_run SET job_status = 'FAILED', error_json = ? WHERE id = ?",
                (json.dumps({"error": str(exc)}), run_id),
            )
            db.commit()
        return

    # Atomic persistence of plan, routes, ledger, and shift_activation_ledger
    with db._lock:
        now_ts = time.time()
        plan_id = str(uuid.uuid4())

        if run_res.plan is not None:
            plan = run_res.plan
            # Insert plan row
            db.execute(
                """
                INSERT INTO plan (
                    id, run_id, version, shift_id, status, result_status, context_hash,
                    coverage_critical, coverage_standard, coverage_flexible,
                    total_cost_paise, idle_seconds, active_vehicles, created_at
                ) VALUES (?, ?, 1, ?, 'PUBLISHED', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan_id,
                    run_id,
                    shift_id,
                    run_res.result_status.value,
                    run_res.context_hash,
                    plan.coverage_tuple[0],
                    plan.coverage_tuple[1],
                    plan.coverage_tuple[2],
                    plan.ledger.total_paise,
                    plan.idle_seconds,
                    plan.m_routes_used,
                    now_ts,
                ),
            )

            # Insert cost breakdown
            db.execute(
                """
                INSERT INTO cost_breakdown (
                    plan_id, distance_cost_paise, time_cost_paise, activation_cost_paise,
                    sla_cost_paise, total_paise, sunk_cost_paise, remaining_decision_cost_paise
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan_id,
                    plan.ledger.distance_cost_paise,
                    plan.ledger.time_cost_paise,
                    plan.ledger.activation_cost_paise,
                    plan.ledger.sla_cost_paise,
                    plan.ledger.total_paise,
                    plan.ledger.sunk_cost_paise,
                    plan.ledger.remaining_decision_cost_paise,
                ),
            )

            # Insert routes and stops
            for route in plan.routes:
                r_id = str(uuid.uuid4())
                db.execute(
                    """
                    INSERT INTO route (
                        id, plan_id, vehicle_id, departure_utc, return_utc, distance_m,
                        paid_seconds, idle_seconds, distance_cost_paise, time_cost_paise, activation_cost_paise
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r_id,
                        plan_id,
                        route.vehicle_id,
                        route.departure_utc,
                        route.return_utc,
                        route.distance_m,
                        route.paid_seconds,
                        route.idle_seconds,
                        route.distance_cost_paise,
                        route.time_cost_paise,
                        route.activation_cost_paise,
                    ),
                )
                for s in route.stops:
                    s_id = str(uuid.uuid4())
                    db.execute(
                        """
                        INSERT INTO route_stop (
                            id, route_id, plan_id, seq, order_id, arrival_utc, wait_s,
                            service_start_utc, service_end_utc, lateness_s, sla_cost_paise, stop_state
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            s_id,
                            r_id,
                            plan_id,
                            s.seq,
                            s.order_id,
                            s.arrival_utc,
                            s.wait_s,
                            s.service_start_utc,
                            s.service_end_utc,
                            s.lateness_s,
                            s.sla_cost_paise,
                            s.stop_state.value,
                        ),
                    )

            # Insert unserved orders
            for u in plan.unserved_orders:
                db.execute(
                    "INSERT INTO unserved_order (plan_id, order_id, reason, detail) VALUES (?, ?, ?, ?)",
                    (plan_id, u.order_id, u.reason.value, u.detail),
                )

            # Atomic post to shift_activation_ledger for newly activated vehicles (Addendum A.2)
            for r in plan.routes:
                if r.vehicle_id not in a_shift:
                    # Look up vehicle activation cost
                    veh_match = next((v for v in vehicles_list if v.id == r.vehicle_id), None)
                    fee = veh_match.activation_cost_paise if veh_match else 0
                    db.execute(
                        """
                        INSERT OR IGNORE INTO shift_activation_ledger (
                            shift_id, vehicle_id, posted_paise, posted_by_plan_id, idempotency_key, posted_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (shift_id, r.vehicle_id, fee, plan_id, f"run-{run_id}-{r.vehicle_id}", now_ts),
                    )

        # Update run status
        db.execute(
            """
            UPDATE optimization_run
            SET job_status = 'COMPLETED',
                result_status = ?,
                stop_reason = ?,
                counters_json = ?,
                finished_at = ?
            WHERE id = ?
            """,
            (
                run_res.result_status.value,
                run_res.stop_reason.value if run_res.stop_reason else None,
                json.dumps(run_res.work_counters),
                now_ts,
                run_id,
            ),
        )
        db.commit()


class JobQueueWorker:
    def __init__(self, db: Database):
        self.db = db
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                row = self.db.fetchone(
                    "SELECT id FROM optimization_run WHERE job_status = 'QUEUED' ORDER BY created_at ASC LIMIT 1"
                )
                if row:
                    process_single_run(row["id"], self.db)
                else:
                    time.sleep(0.5)
            except Exception as e:
                logger.exception("JobQueueWorker loop exception")
                time.sleep(1.0)
