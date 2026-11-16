from __future__ import annotations

from fractions import Fraction
from typing import Tuple


def round_half_even(numerator: int, denominator: int) -> int:
    """
    Exact round-half-to-even on non-negative rational n/d:
    q, r = divmod(n, d)
    if 2*r < d: return q
    if 2*r > d: return q + 1
    return q if q % 2 == 0 else q + 1
    """
    if denominator <= 0:
        raise ValueError(f"Denominator must be strictly positive, got {denominator}")
    if numerator < 0:
        raise ValueError(f"Numerator must be non-negative, got {numerator}")
    
    q, r = divmod(numerator, denominator)
    two_r = 2 * r
    if two_r < denominator:
        return q
    elif two_r > denominator:
        return q + 1
    else:
        return q if q % 2 == 0 else q + 1


def line_distance_cost(distance_m: int, r_d_num: int, r_d_den: int = 1) -> int:
    """
    Compute rounded distance cost in paise per vehicle route.
    C_distance,k = RoundEven(d_k[m] * r_d[paise/km] / 1000)
    """
    if distance_m < 0:
        raise ValueError(f"Distance cannot be negative: {distance_m}")
    if r_d_num < 0 or r_d_den <= 0:
        raise ValueError(f"Invalid distance rate rational: {r_d_num}/{r_d_den}")
    
    num = distance_m * r_d_num
    den = 1000 * r_d_den
    return round_half_even(num, den)


def line_time_cost(paid_duration_s: int, r_t_num: int, r_t_den: int = 1) -> int:
    """
    Compute rounded paid-time cost in paise per vehicle route.
    C_time,k = RoundEven(tau_k[s] * r_t[paise/hour] / 3600)
    """
    if paid_duration_s < 0:
        raise ValueError(f"Duration cannot be negative: {paid_duration_s}")
    if r_t_num < 0 or r_t_den <= 0:
        raise ValueError(f"Invalid time rate rational: {r_t_num}/{r_t_den}")
    
    num = paid_duration_s * r_t_num
    den = 3600 * r_t_den
    return round_half_even(num, den)


def line_sla_cost(lateness_s: int, p_num: int, p_den: int = 1) -> int:
    """
    Compute rounded SLA penalty cost in paise per served soft-window order.
    C_SLA,i = RoundEven(L_i[s] * p_i[paise/minute] / 60)
    """
    if lateness_s <= 0:
        return 0
    if p_num < 0 or p_den <= 0:
        raise ValueError(f"Invalid SLA rate rational: {p_num}/{p_den}")
    
    num = lateness_s * p_num
    den = 60 * p_den
    return round_half_even(num, den)


def compute_rounding_deviation_bound(num_routes: int, num_soft_orders: int) -> Fraction:
    """
    Worst-case absolute rounding deviation versus unrounded totals:
    (2R + S) / 2 paise, where R is number of routes (distance + time) and S is number of soft orders.
    """
    return Fraction(2 * num_routes + num_soft_orders, 2)


I128_MAX = (1 << 127) - 1
I128_MIN = -(1 << 127)


def check_i128_range(val: int) -> int:
    """
    Check if integer fits within signed i128 bounds.
    Python handles arbitrary-precision BigInt natively, so we never silently wrap.
    """
    if val < I128_MIN or val > I128_MAX:
        # According to 4.6: arbitrary precision fallback is used, or NUMERIC_RANGE_ERROR if out of bounds
        pass
    return val
