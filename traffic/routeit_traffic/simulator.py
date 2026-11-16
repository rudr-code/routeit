from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from routeit_engine.contracts import Plan, StopState
from routeit_engine.dynamic import (
    OperationalSnapshot,
    VehicleActivityState,
    VehicleOperationalState,
)
from routeit_traffic.scenario import TrafficScenario


@dataclass
class SimulationEvent:
    event_id: str
    kind: str  # ARRIVED, SERVICE_COMPLETED, INCIDENT
    timestamp_utc: int
    payload: Dict[str, Any]


class TrafficSimulator:
    """
    World-truth execution simulator that advances vehicle routes through time
    and produces versioned OperationalSnapshots for dynamic replanning.
    """
    def __init__(self, plan: Plan, scenario: TrafficScenario, start_time_utc: int):
        self.plan = plan
        self.scenario = scenario
        self.current_time_utc = start_time_utc
        self.completed_orders: Set[str] = set()
        self.snapshot_version = 1
        self.events_history: List[SimulationEvent] = []

    def advance_time(self, new_time_utc: int) -> List[SimulationEvent]:
        new_events: List[SimulationEvent] = []
        if new_time_utc <= self.current_time_utc:
            return new_events

        self.current_time_utc = new_time_utc

        # Check vehicle route progress
        for route in self.plan.routes:
            for stop in route.stops:
                if stop.order_id not in self.completed_orders:
                    if stop.service_end_utc <= self.current_time_utc:
                        self.completed_orders.add(stop.order_id)
                        ev = SimulationEvent(
                            event_id=f"evt_{len(self.events_history) + len(new_events) + 1}",
                            kind="SERVICE_COMPLETED",
                            timestamp_utc=stop.service_end_utc,
                            payload={"order_id": stop.order_id, "vehicle_id": route.vehicle_id},
                        )
                        new_events.append(ev)

        self.events_history.extend(new_events)
        return new_events

    def get_operational_snapshot(self) -> OperationalSnapshot:
        """Derive current operational state across all vehicles."""
        v_states: Dict[str, VehicleOperationalState] = {}

        for route in self.plan.routes:
            vid = route.vehicle_id
            rem_stops = [s for s in route.stops if s.order_id not in self.completed_orders]

            if not rem_stops:
                v_states[vid] = VehicleOperationalState(
                    vehicle_id=vid,
                    activity=VehicleActivityState.COMPLETED,
                    current_node="0",
                    current_time_utc=self.current_time_utc,
                    residual_load=0,
                )
            else:
                next_stop = rem_stops[0]
                activity = (
                    VehicleActivityState.AT_STOP
                    if next_stop.arrival_utc <= self.current_time_utc < next_stop.service_end_utc
                    else VehicleActivityState.EN_ROUTE
                )
                v_states[vid] = VehicleOperationalState(
                    vehicle_id=vid,
                    activity=activity,
                    current_node=next_stop.order_id,
                    current_time_utc=self.current_time_utc,
                    residual_load=sum(1 for _ in rem_stops),  # simple load proxy
                    locked_stop_order_id=next_stop.order_id,
                    locked_stop_departure_utc=next_stop.service_end_utc,
                )

        snap = OperationalSnapshot(
            snapshot_id=f"snap_{self.snapshot_version}",
            snapshot_version=self.snapshot_version,
            t0_utc=self.current_time_utc,
            traffic_scenario_id=self.scenario.id,
            scenario_revision=self.scenario.revision,
            vehicle_states=v_states,
            completed_order_ids=tuple(sorted(list(self.completed_orders))),
            sunk_cost_paise=sum(r.distance_cost_paise // 2 for r in self.plan.routes),
        )
        self.snapshot_version += 1
        return snap
