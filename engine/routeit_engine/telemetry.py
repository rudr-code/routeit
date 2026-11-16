from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WorkCounters:
    candidates_generated: int = 0
    unique_permutations: int = 0
    canonical_evaluations: int = 0
    cache_hits: int = 0
    ls_evaluations: int = 0
    road_lookups: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "candidates_generated": self.candidates_generated,
            "unique_permutations": self.unique_permutations,
            "canonical_evaluations": self.canonical_evaluations,
            "cache_hits": self.cache_hits,
            "ls_evaluations": self.ls_evaluations,
            "road_lookups": self.road_lookups,
        }


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    planning_request_id: str
    context_hash: str
    engine_version: str
    solver: str
    seed: int
    runtime_s: float
    counters: Dict[str, int]
    result_status: str
    stop_reason: str
    best_plan_hash: str
    flags: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
