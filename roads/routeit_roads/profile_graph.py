from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from routeit_engine.oracle import ArrivalResult, OracleStatus, RoadOracle


@dataclass(frozen=True)
class Edge:
    u: str
    v: str
    length_m: int
    road_class: str = "primary"
    edge_id: int = 0


class TimeProfile:
    """
    Speed factors per road class over intervals.
    """
    def __init__(
        self,
        intervals: List[Tuple[int, int]],  # [(t_start, t_end), ...]
        class_speeds_mps: Dict[str, List[float]],  # e.g. {"primary": [10.0, 5.0, 12.0]}
        horizon_start: int = 0,
        horizon_end: int = 86400,
    ):
        self.intervals = intervals
        self.class_speeds_mps = class_speeds_mps
        self.horizon_start = horizon_start
        self.horizon_end = horizon_end

    def get_speed(self, road_class: str, t: float) -> float:
        speeds = self.class_speeds_mps.get(road_class, [10.0])
        for idx, (t_start, t_end) in enumerate(self.intervals):
            if t_start <= t < t_end:
                return speeds[min(idx, len(speeds) - 1)]
        return speeds[-1]

    def get_next_break(self, t: float) -> Optional[float]:
        for t_start, t_end in self.intervals:
            if t < t_end:
                return float(t_end)
        return None


def traverse_edge(
    t: float,
    length_m: int,
    road_class: str,
    profile: TimeProfile,
) -> float:
    """
    Link-travel model of Ichoua, Gendreau and Potvin (2003):
    Integrates speed along edge traversal, guaranteeing FIFO by construction.
    """
    remaining = float(length_m)
    curr_t = float(t)

    while remaining > 1e-6:
        speed = max(0.1, profile.get_speed(road_class, curr_t))
        t_next = profile.get_next_break(curr_t)

        if t_next is None or t_next <= curr_t:
            curr_t += remaining / speed
            break

        reach = speed * (t_next - curr_t)
        if reach >= remaining:
            curr_t += remaining / speed
            break
        else:
            remaining -= reach
            curr_t = t_next

    return curr_t


class ProfileGraphOracle:
    """
    Graph-based Road Oracle with time-dependent profile tables.
    """
    def __init__(
        self,
        nodes: Dict[str, Tuple[float, float]],  # id -> (lat, lon)
        edges: List[Edge],
        profile: TimeProfile,
        grid_seconds: int = 900,  # 15 min grid
    ):
        self.nodes = nodes
        self.profile = profile
        self.grid_seconds = grid_seconds
        self.closed_edges: Set[int] = set()

        # Build adjacency graph
        self.adj: Dict[str, List[Edge]] = {nid: [] for nid in nodes}
        for e in edges:
            if e.u in self.adj and e.v in self.nodes:
                self.adj[e.u].append(e)

        # Precompute shortest paths
        self.paths: Dict[Tuple[str, str], Tuple[int, List[Edge]]] = {}
        self._precompute_fixed_paths()

    def _precompute_fixed_paths(self):
        """Precompute reference fastest path for all node pairs using Dijkstra."""
        for u in self.nodes:
            dist = {n: float("inf") for n in self.nodes}
            dist[u] = 0
            pred_edge: Dict[str, Edge] = {}
            pq = [(0, u)]

            while pq:
                d, curr = heapq.heappop(pq)
                if d > dist[curr]:
                    continue
                for e in self.adj.get(curr, []):
                    if e.edge_id in self.closed_edges:
                        continue
                    new_d = d + e.length_m
                    if new_d < dist[e.v]:
                        dist[e.v] = new_d
                        pred_edge[e.v] = e
                        heapq.heappush(pq, (new_d, e.v))

            for v in self.nodes:
                if dist[v] < float("inf"):
                    # Reconstruct path
                    curr = v
                    path_edges = []
                    while curr != u and curr in pred_edge:
                        pe = pred_edge[curr]
                        path_edges.append(pe)
                        curr = pe.u
                    path_edges.reverse()
                    self.paths[(u, v)] = (int(dist[v]), path_edges)

    def close_edge(self, edge_id: int):
        self.closed_edges.add(edge_id)
        self._precompute_fixed_paths()

    def arrival(self, u: str, v: str, depart_t: int, ctx: Any = None) -> ArrivalResult:
        if depart_t < self.profile.horizon_start or depart_t > self.profile.horizon_end:
            return ArrivalResult(
                arrive_t=depart_t,
                distance_m=0,
                status=OracleStatus.TRAFFIC_DATA_INSUFFICIENT,
                message=f"Departure time {depart_t} is outside traffic horizon [{self.profile.horizon_start}, {self.profile.horizon_end}]",
            )

        if u == v:
            return ArrivalResult(arrive_t=depart_t, distance_m=0, status=OracleStatus.SUCCESS)

        if (u, v) not in self.paths:
            return ArrivalResult(
                arrive_t=depart_t,
                distance_m=0,
                status=OracleStatus.PATH_UNAVAILABLE,
                message=f"No routable path from {u} to {v}",
            )

        dist_m, edge_list = self.paths[(u, v)]
        if not edge_list and u != v:
            return ArrivalResult(
                arrive_t=depart_t,
                distance_m=0,
                status=OracleStatus.PATH_UNAVAILABLE,
                message=f"Path empty from {u} to {v}",
            )

        # Monotone piecewise linear interpolation on grid
        g_s = self.grid_seconds
        t_low = (depart_t // g_s) * g_s
        t_high = t_low + g_s

        def eval_exact_departure(d_t: int) -> int:
            t_curr = float(d_t)
            for e in edge_list:
                t_curr = traverse_edge(t_curr, e.length_m, e.road_class, self.profile)
            return int(math.ceil(t_curr))

        a_low = eval_exact_departure(t_low)
        if t_high == t_low or depart_t == t_low:
            arr_t = a_low
        else:
            a_high = eval_exact_departure(t_high)
            # Monotone linear interpolation
            weight = float(depart_t - t_low) / float(t_high - t_low)
            arr_t = int(math.ceil(a_low + weight * (a_high - a_low)))

        edge_ids = tuple(e.edge_id for e in edge_list)
        return ArrivalResult(
            arrive_t=arr_t,
            distance_m=dist_m,
            status=OracleStatus.SUCCESS,
            edge_ids=edge_ids,
        )

    def path(self, u: str, v: str, depart_t: int, ctx: Any = None) -> Optional[List[Dict[str, float]]]:
        if (u, v) not in self.paths:
            return None
        _, edge_list = self.paths[(u, v)]
        coords = []
        if u in self.nodes:
            coords.append({"lat": self.nodes[u][0], "lon": self.nodes[u][1]})
        for e in edge_list:
            if e.v in self.nodes:
                coords.append({"lat": self.nodes[e.v][0], "lon": self.nodes[e.v][1]})
        return coords

    def identity(self) -> Dict[str, Any]:
        return {
            "kind": "PROFILE_GRAPH",
            "node_count": len(self.nodes),
            "edge_count": sum(len(el) for el in self.adj.values()),
            "grid_seconds": self.grid_seconds,
            "horizon": [self.profile.horizon_start, self.profile.horizon_end],
        }
