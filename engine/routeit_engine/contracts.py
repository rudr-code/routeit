from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


class WindowType(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"


class PriorityTier(str, Enum):
    CRITICAL = "CRITICAL"
    STANDARD = "STANDARD"
    FLEXIBLE = "FLEXIBLE"


class PlanningMode(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"


class FleetMode(str, Enum):
    TARGET = "TARGET"
    AUTO = "AUTO"


class ProfileKind(str, Enum):
    COST_FOCUSED = "COST_FOCUSED"
    BALANCED = "BALANCED"
    DRIVER_PRODUCTIVITY = "DRIVER_PRODUCTIVITY"


class ResultStatus(str, Enum):
    COMPLETE_FEASIBLE = "COMPLETE_FEASIBLE"
    PARTIAL_FEASIBLE = "PARTIAL_FEASIBLE"
    NO_FEASIBLE_PLAN_FOUND = "NO_FEASIBLE_PLAN_FOUND"
    PROVEN_INFEASIBLE = "PROVEN_INFEASIBLE"
    SEARCH_BUDGET_EXHAUSTED = "SEARCH_BUDGET_EXHAUSTED"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_CONFIG = "INVALID_CONFIG"
    ROAD_ORACLE_FAILURE = "ROAD_ORACLE_FAILURE"
    NUMERIC_RANGE_ERROR = "NUMERIC_RANGE_ERROR"
    PROFILE_REFERENCE_UNAVAILABLE = "PROFILE_REFERENCE_UNAVAILABLE"
    PROFILE_CAP_UNSATISFIED = "PROFILE_CAP_UNSATISFIED"
    SHIFT_FLEET_POLICY_CONFLICT = "SHIFT_FLEET_POLICY_CONFLICT"


class StopReason(str, Enum):
    CONVERGED_PROVEN_EXACT = "CONVERGED_PROVEN_EXACT"
    EVALUATION_BUDGET = "EVALUATION_BUDGET"
    WALL_LIMIT = "WALL_LIMIT"
    STAGNATION = "STAGNATION"
    CANCELLED = "CANCELLED"


class UnservedReason(str, Enum):
    PROVEN_UNSERVICEABLE_CAPACITY = "PROVEN_UNSERVICEABLE_CAPACITY"
    PROVEN_UNSERVICEABLE_HARD_WINDOW = "PROVEN_UNSERVICEABLE_HARD_WINDOW"
    UNSERVED_BY_OPTIMIZATION_DECISION = "UNSERVED_BY_OPTIMIZATION_DECISION"
    UNSERVED_DECODER_LIMIT = "UNSERVED_DECODER_LIMIT"
    UNSERVED_SEARCH_BUDGET = "UNSERVED_SEARCH_BUDGET"
    UNSERVED_ROAD_DATA_FAILURE = "UNSERVED_ROAD_DATA_FAILURE"
    UNSERVED_NOT_EVALUATED = "UNSERVED_NOT_EVALUATED"


class StopState(str, Enum):
    PENDING = "PENDING"
    LOCKED = "LOCKED"
    COMPLETED = "COMPLETED"


class ViolationType(str, Enum):
    HARD_WINDOW = "HARD_WINDOW"
    CAPACITY = "CAPACITY"
    RETURN_DEADLINE = "RETURN_DEADLINE"
    ROAD_PATH = "ROAD_PATH"
    DRIVER_DUTY = "DRIVER_DUTY"


@dataclass(frozen=True)
class Violation:
    violation_type: ViolationType
    entity_id: str
    details: str = ""
    magnitude: int = 0


@dataclass(frozen=True)
class Order:
    id: str
    node_id: str
    demand: int
    service_seconds: int
    window_open: int  # UTC epoch seconds
    window_close: int  # UTC epoch seconds
    window_type: WindowType
    priority: PriorityTier
    sla_rate_num: int = 0  # paise per minute of service lateness
    sla_rate_den: int = 1


@dataclass(frozen=True)
class VehicleSpec:
    id: str
    capacity: int
    available: bool = True
    activation_cost_paise: int = 0
    profile: str = "default"


@dataclass(frozen=True)
class RouteStart:
    node_id: str
    departure_time: int
    initial_load: int
    locked_stop: Optional[str] = None


@dataclass(frozen=True)
class RouteStop:
    seq: int
    order_id: str
    arrival_utc: int
    wait_s: int
    service_start_utc: int
    service_end_utc: int
    lateness_s: int
    sla_cost_paise: int
    stop_state: StopState = StopState.PENDING


@dataclass(frozen=True)
class RouteLeg:
    seq: int
    from_node: str
    to_node: str
    distance_m: int
    depart_utc: int
    arrive_utc: int
    edge_ids: Tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Route:
    vehicle_id: str
    departure_utc: int
    return_utc: int
    distance_m: int
    paid_seconds: int
    idle_seconds: int
    distance_cost_paise: int
    time_cost_paise: int
    activation_cost_paise: int
    stops: Tuple[RouteStop, ...]
    legs: Tuple[RouteLeg, ...] = field(default_factory=tuple)

    @property
    def total_cost_paise(self) -> int:
        return (
            self.distance_cost_paise
            + self.time_cost_paise
            + self.activation_cost_paise
            + sum(s.sla_cost_paise for s in self.stops)
        )


@dataclass(frozen=True)
class UnservedOrderRecord:
    order_id: str
    reason: UnservedReason
    detail: str = ""


@dataclass(frozen=True)
class CostLedger:
    distance_cost_paise: int
    time_cost_paise: int
    activation_cost_paise: int
    sla_cost_paise: int
    total_paise: int
    rounding_deviation_bound: Fraction = field(default_factory=lambda: Fraction(0, 1))
    sunk_cost_paise: int = 0
    remaining_decision_cost_paise: int = 0


@dataclass(frozen=True)
class Plan:
    routes: Tuple[Route, ...]
    served_orders: Tuple[str, ...]
    unserved_orders: Tuple[UnservedOrderRecord, ...]
    ledger: CostLedger
    coverage_tuple: Tuple[int, int, int]  # (N_Critical, N_Standard, N_Flexible)
    idle_seconds: int
    m_routes_used: int
    m_new_activated: int
    m_shift_activated_after: int
    is_feasible: bool
    violations: Tuple[Violation, ...] = field(default_factory=tuple)
    canonical_hash: str = ""

    @property
    def c_dec(self) -> int:
        return self.ledger.remaining_decision_cost_paise or self.ledger.total_paise


@dataclass(frozen=True)
class SolveRequest:
    orders: Tuple[Order, ...]
    vehicles: Tuple[VehicleSpec, ...]
    depot_node_id: str
    depot_return_enabled: bool = True
    depot_return_deadline: Optional[int] = None
    planning_mode: PlanningMode = PlanningMode.COMPLETE
    fleet_mode: FleetMode = FleetMode.AUTO
    fleet_max: Optional[int] = None
    profile_kind: ProfileKind = ProfileKind.COST_FOCUSED
    profile_params: Dict[str, Any] = field(default_factory=dict)
    departure_set: Tuple[int, ...] = field(default_factory=lambda: (0,))
    distance_rate_num: int = 0
    distance_rate_den: int = 1
    time_rate_num: int = 0
    time_rate_den: int = 1
    rounding_version: str = "v1_round_even"
    shift_id: str = "shift-default"
    a_shift: Tuple[str, ...] = field(default_factory=tuple)  # Set of vehicle IDs already active in shift
    m_shift_activated_max: Optional[int] = None
    seed: int = 42
    max_evaluations: int = 10000
    max_wall_seconds: float = 60.0
    stagnation_limit: int = 1000


@dataclass(frozen=True)
class RunResult:
    run_id: str
    context_hash: str
    result_status: ResultStatus
    stop_reason: Optional[StopReason] = None
    plan: Optional[Plan] = None
    proof_scope: str = "NONE"
    flags: Dict[str, Any] = field(default_factory=dict)
    work_counters: Dict[str, int] = field(default_factory=dict)
    error_message: Optional[str] = None
