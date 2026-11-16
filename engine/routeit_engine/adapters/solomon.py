from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from routeit_engine.contracts import (
    Order,
    PriorityTier,
    VehicleSpec,
    WindowType,
)


def parse_solomon_instance(
    content: str,
    vehicle_capacity: Optional[int] = None,
    num_vehicles: Optional[int] = None,
    due_date_completion_adjustment: bool = True,
) -> Tuple[Dict[str, Any], Tuple[Order, ...], Tuple[VehicleSpec, ...], Dict[str, Tuple[float, float]]]:
    """
    Parses a standard Solomon benchmark instance text.
    Returns:
      depot: {"node_id": "0", "x": ..., "y": ..., "open": ..., "close": ...}
      orders: tuple of Order
      vehicles: tuple of VehicleSpec
      coords: dict of node_id -> (x, y)
    """
    lines = content.strip().splitlines()
    depot_data = {}
    orders: List[Order] = []
    coords: Dict[str, Tuple[float, float]] = {}

    # Look for VEHICLE capacity line
    cap = vehicle_capacity or 200
    v_count = num_vehicles or 25

    parsing_customers = False
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        if "VEHICLE" in line_s or "NUMBER" in line_s:
            continue
        if "CUSTOMER" in line_s:
            continue
        if "CUST NO." in line_s:
            parsing_customers = True
            continue

        parts = re.split(r"\s+", line_s)
        if not parts:
            continue

        # Check if line contains vehicle info (e.g., "25      200")
        if not parsing_customers and len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            v_count = int(parts[0])
            cap = int(parts[1])
            continue

        if parsing_customers and len(parts) >= 7:
            cust_id = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            demand = int(float(parts[3]))
            ready = int(float(parts[4]))
            due = int(float(parts[5]))
            service = int(float(parts[6]))

            coords[str(cust_id)] = (x, y)

            if cust_id == "0":
                depot_data = {
                    "node_id": "0",
                    "x": x,
                    "y": y,
                    "open": ready,
                    "close": due,
                }
            else:
                # In Solomon, due date bounds arrival / service start.
                # In Route.it, window_close bounds service completion D_i = B_i + s_i <= b_i'.
                adjusted_due = (due + service) if due_date_completion_adjustment else due
                order = Order(
                    id=f"cust_{cust_id}",
                    node_id=str(cust_id),
                    demand=demand,
                    service_seconds=service,
                    window_open=ready,
                    window_close=adjusted_due,
                    window_type=WindowType.HARD,
                    priority=PriorityTier.STANDARD,
                )
                orders.append(order)

    vehicles = tuple(
        VehicleSpec(
            id=f"veh_{i + 1}",
            capacity=cap,
            available=True,
            activation_cost_paise=50000,
        )
        for i in range(v_count)
    )

    return depot_data, tuple(orders), vehicles, coords
