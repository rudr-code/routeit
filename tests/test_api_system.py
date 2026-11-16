from __future__ import annotations

import pytest
from backend.app.db import Database
from backend.app.routers.depots import create_depot, list_depots
from backend.app.routers.vehicles import create_vehicle, list_vehicles
from backend.app.routers.customers import create_customer, confirm_location, list_customers
from backend.app.routers.orders import create_order, list_orders
from backend.app.routers.cost_profiles import create_cost_profile, list_cost_profiles
from backend.app.routers.planning import create_planning_request, start_optimization_run, get_run_result
from backend.app.routers.plans import get_plan, get_plan_cost_breakdown
from backend.app.schemas import (
    ConfirmLocationRequest,
    CostProfileCreate,
    CustomerCreate,
    DepotCreate,
    OptimizationRunCreate,
    OrderCreate,
    PlanningRequestCreate,
    VehicleCreate,
)


@pytest.fixture()
def test_db(tmp_path):
    return Database(db_path=str(tmp_path / "test_api_direct.db"))


def test_depots_and_vehicles_api(test_db):
    # 1. Create and list depot
    depot_res = create_depot(
        DepotCreate(
            name="Sayajigunj Central Hub",
            lat=22.3072,
            lon=73.1812,
            opening_time=25200,
            closing_time=64800,
        ),
        db=test_db,
    )
    assert depot_res.name == "Sayajigunj Central Hub"

    depots = list_depots(db=test_db)
    assert len(depots) >= 1

    # 2. Create and list vehicle
    veh_res = create_vehicle(
        VehicleCreate(
            code="VAN-TEST-1",
            capacity=120,
            activation_cost_paise=35000,
            available=True,
        ),
        db=test_db,
    )
    assert veh_res.code == "VAN-TEST-1"
    vehs = list_vehicles(db=test_db)
    assert len(vehs) >= 1


def test_customer_creation_and_location_confirmation(test_db):
    """AC-13 & Phase 4 Section 8: Interactive location selection and confirmation."""
    # 1. Create customer with unconfirmed location
    cust = create_customer(
        CustomerCreate(
            name="Reliance Mart Sayajigunj",
            lat=22.3100,
            lon=73.1850,
            address_text="Opposite Sayaji Garden, Vadodara",
            delivery_instructions="Gate 1",
        ),
        db=test_db,
    )
    assert cust.confirmation_status == "UNCONFIRMED"

    # 2. Confirm pin location (e.g., dispatcher drags pin to delivery gate)
    confirmed = confirm_location(
        id=cust.id,
        req=ConfirmLocationRequest(
            lat=22.3105,
            lon=73.1855,
            delivery_instructions="Gate 2 (Back entrance)",
        ),
        db=test_db,
    )
    assert confirmed.confirmation_status == "CONFIRMED"
    assert confirmed.delivery_instructions == "Gate 2 (Back entrance)"
    assert confirmed.lat == 22.3105


def test_end_to_end_planning_and_run(test_db):
    """End-to-end integration: customer -> order -> planning request -> solve -> plan -> cost breakdown."""
    # 1. Create customer and confirm
    cust = create_customer(
        CustomerCreate(
            name="Sayaji Supermarket",
            lat=22.3120,
            lon=73.1860,
            address_text="Vadodara East",
        ),
        db=test_db,
    )
    confirm_location(cust.id, ConfirmLocationRequest(lat=22.3120, lon=73.1860), db=test_db)

    # 2. Create delivery order
    order = create_order(
        OrderCreate(
            customer_id=cust.id,
            external_ref="ORD-E2E-1",
            demand_units=10,
            service_seconds=60,
            window_open=25200,
            window_close=50000,
            window_type="HARD",
            priority="STANDARD",
        ),
        db=test_db,
    )
    assert order.status == "NEW"

    # 3. Create planning request
    depots = list_depots(db=test_db)
    vehs = list_vehicles(db=test_db)
    profiles = list_cost_profiles(db=test_db)

    pr = create_planning_request(
        PlanningRequestCreate(
            depot_id=depots[0].id,
            order_ids=[order.id],
            vehicle_ids=[v.id for v in vehs],
            planning_mode="COMPLETE",
            fleet_mode="AUTO",
            profile_kind="COST_FOCUSED",
            cost_profile_id=profiles[0].id,
            traffic_scenario_id="traffic-vadodara",
            shift_id="shift-e2e-01",
        ),
        db=test_db,
    )
    assert pr.status == "VALIDATED"
    assert pr.context_hash is not None

    # 4. Trigger optimization run
    run_res = start_optimization_run(
        OptimizationRunCreate(
            planning_request_id=pr.planning_request_id,
            solver_kind="ALNS",
            seed=42,
        ),
        db=test_db,
    )
    assert run_res.run_id is not None

    # 5. Fetch completed run result
    res = get_run_result(run_res.run_id, db=test_db)
    assert res.job_status == "COMPLETED"
    assert res.result_status == "COMPLETE_FEASIBLE"
    assert res.plan_id is not None

    # 6. Fetch published plan
    plan = get_plan(res.plan_id, db=test_db)
    assert plan.plan_id == res.plan_id
    assert len(plan.routes) >= 1
    assert plan.total_cost_paise > 0

    # 7. Verify cost breakdown
    cost_bd = get_plan_cost_breakdown(res.plan_id, db=test_db)
    assert cost_bd.total_paise == plan.total_cost_paise
