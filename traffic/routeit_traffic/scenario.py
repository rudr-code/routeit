from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Incident:
    id: str
    affected_edge_ids: Tuple[int, ...]
    speed_multiplier: float  # e.g., 0.3 for heavy congestion
    is_closure: bool
    start_utc: int
    end_utc: int


@dataclass
class TrafficScenario:
    id: str
    revision: int = 0
    intervals: List[Tuple[int, int]] = field(
        default_factory=lambda: [(27000, 37800), (37800, 59400), (59400, 72000)]
    )  # 07:30-10:30, 10:30-16:30, 16:30-20:00
    class_factors: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "primary": [0.55, 0.85, 0.50],
            "secondary": [0.60, 0.90, 0.55],
            "residential": [0.80, 0.95, 0.80],
        }
    )
    incidents: List[Incident] = field(default_factory=list)
    grid_seconds: int = 900
