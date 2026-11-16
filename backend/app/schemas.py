from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DepotCreate(BaseModel):
    name: str
    lat: float
    lon: float
    opening_time: int = 25200
    closing_time: int = 64800


class DepotResponse(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    snapped_node_id: str
    opening_time: int
    closing_time: int


class VehicleCreate(BaseModel):
    code: str
    capacity: int
    activation_cost_paise: int = 0
    available: bool = True
    profile: str = "default"


class VehicleResponse(BaseModel):
    id: str
    code: str
    capacity: int
    activation_cost_paise: int
    available: bool
    profile: str


class CustomerCreate(BaseModel):
    name: str
    lat: float
    lon: float
    address_text: str
    delivery_instructions: str = ""


class ConfirmLocationRequest(BaseModel):
    lat: float
    lon: float
    delivery_instructions: str = ""


class CustomerResponse(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    snapped_node_id: str
    address_text: str
    address_source: str
    delivery_instructions: str
    confirmation_status: str


class OrderCreate(BaseModel):
    customer_id: str
    external_ref: str
    demand_units: int
    service_seconds: int = 300
    window_open: int = 28800
    window_close: int = 64800
    window_type: str = "HARD"
    priority: str = "STANDARD"
    sla_rate_num: int = 0
    sla_rate_den: int = 1


class OrderResponse(BaseModel):
    id: str
    customer_id: str
    external_ref: str
    demand_units: int
    service_seconds: int
    window_open: int
    window_close: int
    window_type: str
    priority: str
    sla_rate_num: int
    sla_rate_den: int
    status: str


class CostProfileCreate(BaseModel):
    name: str
    distance_rate_num: int = 800
    distance_rate_den: int = 1
    time_rate_num: int = 10000
    time_rate_den: int = 1
    rounding_policy_version: str = "v1_round_even"


class CostProfileResponse(BaseModel):
    id: str
    name: str
    version: str
    distance_rate_num: int
    distance_rate_den: int
    time_rate_num: int
    time_rate_den: int
    rounding_policy_version: str


class PlanningRequestCreate(BaseModel):
    depot_id: str
    order_ids: List[str]
    vehicle_ids: List[str]
    planning_mode: str = "COMPLETE"
    fleet_mode: str = "AUTO"
    fleet_max: Optional[int] = None
    profile_kind: str = "COST_FOCUSED"
    profile_params: Dict[str, Any] = Field(default_factory=dict)
    departure_set: List[int] = Field(default_factory=lambda: [25200])
    cost_profile_id: str
    traffic_scenario_id: str
    shift_id: str = "shift-01"


class PlanningRequestResponse(BaseModel):
    planning_request_id: str
    shift_id: str
    status: str
    context_hash: str
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    proofs: List[Dict[str, Any]] = Field(default_factory=list)


class OptimizationRunCreate(BaseModel):
    planning_request_id: str
    solver_kind: str = "ALNS"
    seed: int = 42


class OptimizationRunResponse(BaseModel):
    run_id: str
    job_status: str


class RunProgressResponse(BaseModel):
    run_id: str
    job_status: str
    progress: Dict[str, Any] = Field(default_factory=dict)


class RouteStopResponse(BaseModel):
    seq: int
    order_id: str
    arrival_utc: int
    wait_s: int
    service_start_utc: int
    service_end_utc: int
    lateness_s: int
    sla_cost_paise: int
    stop_state: str


class RouteResponse(BaseModel):
    route_id: str
    vehicle_id: str
    departure_utc: int
    return_utc: int
    distance_m: int
    paid_seconds: int
    idle_seconds: int
    distance_cost_paise: int
    time_cost_paise: int
    activation_cost_paise: int
    total_cost_paise: int
    stops: List[RouteStopResponse]


class CostBreakdownResponse(BaseModel):
    distance_cost_paise: int
    time_cost_paise: int
    activation_cost_paise: int
    sla_cost_paise: int
    total_paise: int
    sunk_cost_paise: int
    remaining_decision_cost_paise: int


class PlanResponse(BaseModel):
    plan_id: str
    run_id: str
    version: int
    shift_id: str
    status: str
    result_status: str
    context_hash: str
    coverage: Dict[str, int]
    total_cost_paise: int
    idle_seconds: int
    active_vehicles: int
    routes: List[RouteResponse]
    cost_breakdown: CostBreakdownResponse
    unserved: List[Dict[str, Any]]


class RunResultResponse(BaseModel):
    run_id: str
    job_status: str
    result_status: str
    plan_id: Optional[str] = None
    stop_reason: Optional[str] = None
    proof_scope: str = "NONE"
    coverage: Dict[str, int] = Field(default_factory=dict)
    unserved: List[Dict[str, Any]] = Field(default_factory=list)
    flags: Dict[str, Any] = Field(default_factory=dict)
    context_hash: str = ""
    error_message: Optional[str] = None


class ReplanEventCreate(BaseModel):
    kind: str  # INCIDENT, DELAY, TRAFFIC_UPDATE
    occurred_at: str
    payload: Dict[str, Any]


class ErrorEnvelope(BaseModel):
    error: Dict[str, Any]
