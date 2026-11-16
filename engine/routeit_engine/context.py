from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from routeit_engine.contracts import Order, VehicleSpec


@dataclass(frozen=True)
class EvaluationContext:
    orders: Tuple[Order, ...]
    vehicles: Tuple[VehicleSpec, ...]
    depot_node_id: str
    depot_return_enabled: bool
    depot_return_deadline: Optional[int]
    planning_mode: str
    fleet_mode: str
    fleet_max: Optional[int]
    profile_kind: str
    profile_params: Dict[str, Any]
    departure_set: Tuple[int, ...]
    distance_rate_num: int
    distance_rate_den: int
    time_rate_num: int
    time_rate_den: int
    rounding_version: str
    shift_id: str
    a_shift: Tuple[str, ...]  # Immutable sorted tuple of already-activated vehicle IDs
    m_shift_activated_max: Optional[int] = None
    graph_version: str = "v1"
    traffic_scenario_id: str = "default"
    scenario_revision: int = 0
    fifo_policy: str = "monotone_interpolation"
    partial_tie_policy_version: str = "v1_signature_hash"
    evaluator_version: str = "v5.0_canonical"
    operational_snapshot_version: Optional[int] = None

    def canonical_dict(self) -> Dict[str, Any]:
        """Produce a deterministic, JSON-serializable representation."""
        return {
            "depot_node_id": self.depot_node_id,
            "depot_return_enabled": self.depot_return_enabled,
            "depot_return_deadline": self.depot_return_deadline,
            "planning_mode": self.planning_mode,
            "fleet_mode": self.fleet_mode,
            "fleet_max": self.fleet_max,
            "profile_kind": self.profile_kind,
            "profile_params": dict(sorted(self.profile_params.items())),
            "departure_set": list(self.departure_set),
            "distance_rate": [self.distance_rate_num, self.distance_rate_den],
            "time_rate": [self.time_rate_num, self.time_rate_den],
            "rounding_version": self.rounding_version,
            "shift_id": self.shift_id,
            "a_shift": sorted(list(self.a_shift)),
            "m_shift_activated_max": self.m_shift_activated_max,
            "graph_version": self.graph_version,
            "traffic_scenario_id": self.traffic_scenario_id,
            "scenario_revision": self.scenario_revision,
            "fifo_policy": self.fifo_policy,
            "partial_tie_policy_version": self.partial_tie_policy_version,
            "evaluator_version": self.evaluator_version,
            "operational_snapshot_version": self.operational_snapshot_version,
            "orders": [
                {
                    "id": o.id,
                    "node_id": o.node_id,
                    "demand": o.demand,
                    "service_seconds": o.service_seconds,
                    "window_open": o.window_open,
                    "window_close": o.window_close,
                    "window_type": o.window_type.value,
                    "priority": o.priority.value,
                    "sla_rate": [o.sla_rate_num, o.sla_rate_den],
                }
                for o in sorted(self.orders, key=lambda x: x.id)
            ],
            "vehicles": [
                {
                    "id": v.id,
                    "capacity": v.capacity,
                    "available": v.available,
                    "activation_cost_paise": v.activation_cost_paise,
                    "profile": v.profile,
                }
                for v in sorted(self.vehicles, key=lambda x: x.id)
            ],
        }

    def canonical_bytes(self) -> bytes:
        d = self.canonical_dict()
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def context_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
