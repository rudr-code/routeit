from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DB_PATH = os.environ.get(
    "ROUTEIT_DB_PATH",
    str(Path(__file__).resolve().parent.parent / "routeit_v5.db"),
)


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._init_schema()

    def _init_schema(self):
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS depot (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    snapped_node_id TEXT NOT NULL,
                    opening_time INTEGER NOT NULL DEFAULT 0,
                    closing_time INTEGER NOT NULL DEFAULT 86400
                );

                CREATE TABLE IF NOT EXISTS vehicle (
                    id TEXT PRIMARY KEY,
                    code TEXT NOT NULL UNIQUE,
                    capacity INTEGER NOT NULL CHECK(capacity > 0),
                    activation_cost_paise INTEGER NOT NULL DEFAULT 0,
                    available INTEGER NOT NULL DEFAULT 1,
                    profile TEXT NOT NULL DEFAULT 'default'
                );

                CREATE TABLE IF NOT EXISTS customer (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    snapped_node_id TEXT NOT NULL,
                    address_text TEXT NOT NULL,
                    address_source TEXT NOT NULL DEFAULT 'DISPATCHER_TYPED',
                    delivery_instructions TEXT DEFAULT '',
                    confirmation_status TEXT NOT NULL DEFAULT 'UNCONFIRMED',
                    confirmed_at REAL
                );

                CREATE TABLE IF NOT EXISTS delivery_order (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL REFERENCES customer(id) ON DELETE CASCADE,
                    external_ref TEXT UNIQUE NOT NULL,
                    demand_units INTEGER NOT NULL CHECK(demand_units > 0),
                    service_seconds INTEGER NOT NULL DEFAULT 300,
                    window_open INTEGER NOT NULL DEFAULT 0,
                    window_close INTEGER NOT NULL DEFAULT 86400,
                    window_type TEXT NOT NULL DEFAULT 'HARD',
                    priority TEXT NOT NULL DEFAULT 'STANDARD',
                    sla_rate_num INTEGER NOT NULL DEFAULT 0,
                    sla_rate_den INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'NEW'
                );

                CREATE TABLE IF NOT EXISTS cost_profile (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL DEFAULT 'v1',
                    distance_rate_num INTEGER NOT NULL DEFAULT 800,
                    distance_rate_den INTEGER NOT NULL DEFAULT 1,
                    time_rate_num INTEGER NOT NULL DEFAULT 10000,
                    time_rate_den INTEGER NOT NULL DEFAULT 1,
                    rounding_policy_version TEXT NOT NULL DEFAULT 'v1_round_even'
                );

                CREATE TABLE IF NOT EXISTS traffic_scenario (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'TIME_PROFILE',
                    revision INTEGER NOT NULL DEFAULT 0,
                    params_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS planning_request (
                    id TEXT PRIMARY KEY,
                    shift_id TEXT NOT NULL,
                    depot_id TEXT NOT NULL REFERENCES depot(id),
                    planning_mode TEXT NOT NULL DEFAULT 'COMPLETE',
                    fleet_mode TEXT NOT NULL DEFAULT 'AUTO',
                    fleet_max INTEGER,
                    profile_kind TEXT NOT NULL DEFAULT 'COST_FOCUSED',
                    profile_params_json TEXT NOT NULL DEFAULT '{}',
                    departure_set_json TEXT NOT NULL DEFAULT '[0]',
                    cost_profile_id TEXT NOT NULL REFERENCES cost_profile(id),
                    traffic_scenario_id TEXT NOT NULL REFERENCES traffic_scenario(id),
                    context_hash TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'VALIDATED',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS optimization_run (
                    id TEXT PRIMARY KEY,
                    planning_request_id TEXT NOT NULL REFERENCES planning_request(id),
                    solver_kind TEXT NOT NULL DEFAULT 'ALNS',
                    seed INTEGER NOT NULL DEFAULT 42,
                    job_status TEXT NOT NULL DEFAULT 'QUEUED',
                    result_status TEXT,
                    stop_reason TEXT,
                    counters_json TEXT NOT NULL DEFAULT '{}',
                    error_json TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL
                );

                CREATE TABLE IF NOT EXISTS plan (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES optimization_run(id),
                    version INTEGER NOT NULL DEFAULT 1,
                    shift_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PUBLISHED',
                    result_status TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    coverage_critical INTEGER NOT NULL DEFAULT 0,
                    coverage_standard INTEGER NOT NULL DEFAULT 0,
                    coverage_flexible INTEGER NOT NULL DEFAULT 0,
                    total_cost_paise INTEGER NOT NULL DEFAULT 0,
                    idle_seconds INTEGER NOT NULL DEFAULT 0,
                    active_vehicles INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS shift_activation_ledger (
                    shift_id TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL REFERENCES vehicle(id),
                    posted_paise INTEGER NOT NULL CHECK(posted_paise >= 0),
                    posted_by_plan_id TEXT NOT NULL REFERENCES plan(id),
                    idempotency_key TEXT NOT NULL,
                    posted_at REAL NOT NULL,
                    PRIMARY KEY (shift_id, vehicle_id),
                    UNIQUE (idempotency_key, vehicle_id)
                );

                CREATE TABLE IF NOT EXISTS route (
                    id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL REFERENCES plan(id) ON DELETE CASCADE,
                    vehicle_id TEXT NOT NULL REFERENCES vehicle(id),
                    departure_utc INTEGER NOT NULL,
                    return_utc INTEGER NOT NULL,
                    distance_m INTEGER NOT NULL,
                    paid_seconds INTEGER NOT NULL,
                    idle_seconds INTEGER NOT NULL,
                    distance_cost_paise INTEGER NOT NULL,
                    time_cost_paise INTEGER NOT NULL,
                    activation_cost_paise INTEGER NOT NULL,
                    UNIQUE (plan_id, vehicle_id)
                );

                CREATE TABLE IF NOT EXISTS route_stop (
                    id TEXT PRIMARY KEY,
                    route_id TEXT NOT NULL REFERENCES route(id) ON DELETE CASCADE,
                    plan_id TEXT NOT NULL REFERENCES plan(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL,
                    order_id TEXT NOT NULL REFERENCES delivery_order(id),
                    arrival_utc INTEGER NOT NULL,
                    wait_s INTEGER NOT NULL,
                    service_start_utc INTEGER NOT NULL,
                    service_end_utc INTEGER NOT NULL,
                    lateness_s INTEGER NOT NULL,
                    sla_cost_paise INTEGER NOT NULL,
                    stop_state TEXT NOT NULL DEFAULT 'PENDING',
                    UNIQUE (route_id, seq),
                    UNIQUE (plan_id, order_id)
                );

                CREATE TABLE IF NOT EXISTS unserved_order (
                    plan_id TEXT NOT NULL REFERENCES plan(id) ON DELETE CASCADE,
                    order_id TEXT NOT NULL REFERENCES delivery_order(id),
                    reason TEXT NOT NULL,
                    detail TEXT DEFAULT '',
                    PRIMARY KEY (plan_id, order_id)
                );

                CREATE TABLE IF NOT EXISTS cost_breakdown (
                    plan_id TEXT PRIMARY KEY REFERENCES plan(id) ON DELETE CASCADE,
                    distance_cost_paise INTEGER NOT NULL,
                    time_cost_paise INTEGER NOT NULL,
                    activation_cost_paise INTEGER NOT NULL,
                    sla_cost_paise INTEGER NOT NULL,
                    total_paise INTEGER NOT NULL,
                    sunk_cost_paise INTEGER NOT NULL DEFAULT 0,
                    remaining_decision_cost_paise INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            # Seed default demo records
            self._seed_defaults()

    def _seed_defaults(self):
        # Insert default depot Sayajigunj if not present
        row = self._conn.execute("SELECT id FROM depot WHERE id = 'depot-sayajigunj'").fetchone()
        if not row:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO depot (id, name, lat, lon, snapped_node_id, opening_time, closing_time)
                VALUES ('depot-sayajigunj', 'Vadodara Central Distribution Hub (Sayajigunj)', 22.3072, 73.1812, 'depot_node', 25200, 64800);
                """
            )
            # Insert demo vehicles
            vehs = [
                ("v-101", "VAN-01", 100, 30000),
                ("v-102", "VAN-02", 100, 30000),
                ("v-103", "VAN-03", 100, 45000),
                ("v-104", "VAN-04", 100, 45000),
            ]
            for vid, code, cap, act in vehs:
                self._conn.execute(
                    "INSERT OR IGNORE INTO vehicle (id, code, capacity, activation_cost_paise) VALUES (?, ?, ?, ?)",
                    (vid, code, cap, act),
                )
            # Insert default cost profile
            self._conn.execute(
                """
                INSERT OR IGNORE INTO cost_profile (id, name, distance_rate_num, distance_rate_den, time_rate_num, time_rate_den)
                VALUES ('cost-prof-std', 'Standard Urban Logistics INR', 800, 1, 10000, 1);
                """
            )
            # Insert default traffic scenario
            self._conn.execute(
                """
                INSERT OR IGNORE INTO traffic_scenario (id, name, revision)
                VALUES ('traffic-vadodara', 'Vadodara Peak Hours Traffic Model', 0);
                """
            )

    def execute(self, query: str, params: Tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(query, params)

    def fetchall(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(query, params).fetchall()

    def fetchone(self, query: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(query, params).fetchone()

    def commit(self):
        with self._lock:
            self._conn.commit()


_db_instance: Optional[Database] = None


def get_db() -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
