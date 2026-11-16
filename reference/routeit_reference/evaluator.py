from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def reference_round_half_even(num: int, den: int) -> int:
    """Independent implementation of round half to even using divmod."""
    q, r = divmod(num, den)
    rem_x_2 = 2 * r
    if rem_x_2 < den:
        return q
    elif rem_x_2 > den:
        return q + 1
    else:
        return q if (q % 2 == 0) else (q + 1)


def evaluate_route_reference(
    node_coords: Dict[str, Tuple[float, float]],
    order_dicts: Sequence[Dict[str, Any]],
    vehicle_dict: Dict[str, Any],
    depot_node_id: str,
    departure_time: int,
    distance_rate_paise_per_km: Fraction,
    time_rate_paise_per_hour: Fraction,
    a_shift: Set[str],
    depot_return: bool = True,
    speed_mps: float = 10.0,  # 10 m/s default speed for euclidean test
) -> Dict[str, Any]:
    """
    Independent reference route evaluator using pure Fractions.
    """
    curr_node = depot_node_id
    curr_time = departure_time
    total_dist_m = 0
    total_wait_s = 0
    stops_out = []
    legs_out = []

    for order in order_dicts:
        o_node = order["node_id"]
        c1 = node_coords[curr_node]
        c2 = node_coords[o_node]
        dist_m = int(round(math.hypot(c1[0] - c2[0], c1[1] - c2[1])))
        travel_s = int(math.ceil(dist_m / speed_mps))
        arrive_t = curr_time + travel_s

        w_open = order["window_open"]
        w_close = order["window_close"]
        s_dur = order["service_seconds"]

        start_t = max(arrive_t, w_open)
        wait_s = max(0, w_open - arrive_t)
        end_t = start_t + s_dur
        late_s = max(0, end_t - w_close)

        # Soft window SLA
        sla_paise = 0
        if order.get("window_type") == "SOFT" and late_s > 0:
            sla_rate = Fraction(order.get("sla_rate_num", 0), order.get("sla_rate_den", 1))
            sla_val = Fraction(late_s) * sla_rate / 60
            sla_paise = reference_round_half_even(sla_val.numerator, sla_val.denominator)

        stops_out.append({
            "order_id": order["id"],
            "arrival": arrive_t,
            "wait": wait_s,
            "start": start_t,
            "end": end_t,
            "lateness": late_s,
            "sla_paise": sla_paise,
        })
        legs_out.append({
            "from": curr_node,
            "to": o_node,
            "distance_m": dist_m,
            "depart": curr_time,
            "arrive": arrive_t,
        })

        total_dist_m += dist_m
        total_wait_s += wait_s
        curr_node = o_node
        curr_time = end_t

    return_time = curr_time
    if depot_return and order_dicts:
        c1 = node_coords[curr_node]
        c2 = node_coords[depot_node_id]
        dist_m = int(round(math.hypot(c1[0] - c2[0], c1[1] - c2[1])))
        travel_s = int(math.ceil(dist_m / speed_mps))
        return_time = curr_time + travel_s
        total_dist_m += dist_m
        legs_out.append({
            "from": curr_node,
            "to": depot_node_id,
            "distance_m": dist_m,
            "depart": curr_time,
            "arrive": return_time,
        })

    paid_s = max(0, return_time - departure_time)

    # Line distance cost: RoundEven(d_k * r_d / 1000)
    dist_frac = Fraction(total_dist_m) * distance_rate_paise_per_km / 1000
    dist_cost_paise = reference_round_half_even(dist_frac.numerator, dist_frac.denominator)

    # Line time cost: RoundEven(H_k * r_t / 3600)
    time_frac = Fraction(paid_s) * time_rate_paise_per_hour / 3600
    time_cost_paise = reference_round_half_even(time_frac.numerator, time_frac.denominator)

    # Vehicle activation
    v_id = vehicle_dict["id"]
    is_new = v_id not in a_shift
    act_paise = vehicle_dict["activation_cost_paise"] if is_new else 0

    total_sla_paise = sum(s["sla_paise"] for s in stops_out)
    total_cost_paise = dist_cost_paise + time_cost_paise + act_paise + total_sla_paise

    return {
        "vehicle_id": v_id,
        "distance_m": total_dist_m,
        "paid_seconds": paid_s,
        "idle_seconds": total_wait_s,
        "distance_cost_paise": dist_cost_paise,
        "time_cost_paise": time_cost_paise,
        "activation_cost_paise": act_paise,
        "sla_cost_paise": total_sla_paise,
        "total_cost_paise": total_cost_paise,
        "stops": stops_out,
        "legs": legs_out,
    }
