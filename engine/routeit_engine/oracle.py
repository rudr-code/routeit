from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class OracleStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PATH_UNAVAILABLE = "PATH_UNAVAILABLE"
    UNSUPPORTED_RESTRICTION = "UNSUPPORTED_RESTRICTION"
    TRAFFIC_DATA_INSUFFICIENT = "TRAFFIC_DATA_INSUFFICIENT"
    ORACLE_FAILURE = "ORACLE_FAILURE"


@dataclass(frozen=True)
class ArrivalResult:
    arrive_t: int
    distance_m: int
    status: OracleStatus = OracleStatus.SUCCESS
    message: str = ""
    edge_ids: tuple[int, ...] = field(default_factory=tuple)

    @property
    def is_success(self) -> bool:
        return self.status == OracleStatus.SUCCESS


@runtime_checkable
class RoadOracle(Protocol):
    def arrival(self, u: str, v: str, depart_t: int, ctx: Any) -> ArrivalResult:
        """Query time-dependent arrival time and distance from node u to node v at depart_t."""
        ...

    def path(self, u: str, v: str, depart_t: int, ctx: Any) -> Optional[List[Dict[str, float]]]:
        """Return full geometry (coordinates list) for visualization."""
        ...

    def identity(self) -> Dict[str, Any]:
        """Return immutable identity metadata (graph hash, scenario revision, etc.)."""
        ...
