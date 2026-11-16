from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from routeit_engine.oracle import ArrivalResult, OracleStatus, RoadOracle


class EuclideanOracle:
    """
    Euclidean Oracle for Solomon benchmarks and geometric test instances.
    """
    def __init__(
        self,
        node_coords: Dict[str, Tuple[float, float]],
        speed_mps: float = 10.0,
        distance_scale: float = 1000.0,  # e.g. 1 unit in Solomon = 1000 meters
    ):
        self.node_coords = node_coords
        self.speed_mps = speed_mps
        self.distance_scale = distance_scale
        self._identity = {
            "kind": "EUCLIDEAN",
            "speed_mps": speed_mps,
            "node_count": len(node_coords),
        }

    def arrival(self, u: str, v: str, depart_t: int, ctx: Any = None) -> ArrivalResult:
        if u not in self.node_coords or v not in self.node_coords:
            return ArrivalResult(
                arrive_t=depart_t,
                distance_m=0,
                status=OracleStatus.PATH_UNAVAILABLE,
                message=f"Node {u} or {v} missing in Euclidean oracle",
            )

        c1 = self.node_coords[u]
        c2 = self.node_coords[v]
        raw_dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        dist_m = int(round(raw_dist * self.distance_scale))

        travel_s = int(math.ceil(dist_m / self.speed_mps)) if self.speed_mps > 0 else 0
        arrive_t = depart_t + travel_s

        return ArrivalResult(
            arrive_t=arrive_t,
            distance_m=dist_m,
            status=OracleStatus.SUCCESS,
        )

    def path(self, u: str, v: str, depart_t: int, ctx: Any = None) -> Optional[List[Dict[str, float]]]:
        if u not in self.node_coords or v not in self.node_coords:
            return None
        c1 = self.node_coords[u]
        c2 = self.node_coords[v]
        return [
            {"lat": c1[1], "lon": c1[0]},
            {"lat": c2[1], "lon": c2[0]},
        ]

    def identity(self) -> Dict[str, Any]:
        return dict(self._identity)
