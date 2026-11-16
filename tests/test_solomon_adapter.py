import pytest
from pathlib import Path
from routeit_engine.adapters.solomon import parse_solomon_instance
from routeit_engine.contracts import Order, VehicleSpec, SolveRequest, PlanningMode, ProfileKind, ResultStatus, WindowType
from routeit_engine import solve
from routeit_roads import EuclideanOracle


def test_solomon_parse_c101_completion_adjustment():
    solomon_path = Path("benchmarks/instances/C101.txt")
    assert solomon_path.is_file(), "C101.txt benchmark instance must exist"
    content = solomon_path.read_text(encoding="utf-8")

    # 1. With completion adjustment (Route.it D_i <= b'_i = b_i + s_i)
    depot_data, orders, vehicles, coords = parse_solomon_instance(content, due_date_completion_adjustment=True)
    assert depot_data["node_id"] == "0"
    assert depot_data["x"] == 40.0
    assert depot_data["y"] == 50.0
    assert depot_data["open"] == 0
    assert depot_data["close"] == 1236

    assert len(orders) == 100
    assert len(vehicles) == 25
    assert len(coords) == 101  # 1 depot + 100 customers

    # Customer 1 in C101: ready 912, due 967, service 90
    c1 = next(o for o in orders if o.id == "cust_1")
    assert c1.window_open == 912
    assert c1.window_close == 967 + 90
    assert c1.service_seconds == 90
    assert c1.demand == 10

    # 2. Without adjustment (raw arrival due date)
    _, orders_raw, _, _ = parse_solomon_instance(content, due_date_completion_adjustment=False)
    c1_raw = next(o for o in orders_raw if o.id == "cust_1")
    assert c1_raw.window_close == 967


def test_solomon_c101_solve_subproblem():
    solomon_path = Path("benchmarks/instances/C101.txt")
    content = solomon_path.read_text(encoding="utf-8")
    depot_data, orders, vehicles, coords = parse_solomon_instance(content, due_date_completion_adjustment=True)

    # Use first 8 customers
    sub_orders = orders[:8]
    sub_vehicles = vehicles[:3]

    req = SolveRequest(
        orders=sub_orders,
        vehicles=sub_vehicles,
        depot_node_id=depot_data["node_id"],
        planning_mode=PlanningMode.COMPLETE,
        profile_kind=ProfileKind.COST_FOCUSED,
        distance_rate_num=100,
        time_rate_num=100,
        max_wall_seconds=5.0,
    )

    oracle = EuclideanOracle(coords, speed_mps=1.0, distance_scale=1.0)
    res = solve(req, oracle=oracle)
    assert res.result_status in (ResultStatus.COMPLETE_FEASIBLE, ResultStatus.PARTIAL_FEASIBLE)
    assert res.plan is not None
    assert len(res.plan.unserved_orders) == 0
    assert len(res.plan.served_orders) == 8
    assert res.plan.ledger.total_paise > 0
    # Every customer served
    served_ids = {stop.order_id for route in res.plan.routes for stop in route.stops}
    assert served_ids == {o.id for o in sub_orders}
