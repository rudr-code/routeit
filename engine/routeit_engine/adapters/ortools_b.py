from __future__ import annotations

from typing import Any, Dict, Optional

from routeit_engine.contracts import Plan, SolveRequest


class ORToolsAdapter:
    @staticmethod
    def check_capabilities(req: SolveRequest) -> Dict[str, Any]:
        """
        Emits capability and equivalence report (Q-HGALS 5.0 §6.10).
        Flags unsupported time-dependence, priority soft-windows, cap, or commitment semantics.
        """
        has_soft_windows = any(o.window_type.value == "SOFT" for o in req.orders)
        is_partial = req.planning_mode.value == "PARTIAL"
        has_time_profiles = len(req.departure_set) > 1
        is_productivity = req.profile_kind.value == "DRIVER_PRODUCTIVITY"

        return {
            "ortools_available": True,
            "supports_exact_time_dependence": False,
            "supports_lexicographic_priorities": False,
            "supports_productivity_cost_cap": False,
            "can_claim_equivalence": not (has_soft_windows or is_partial or has_time_profiles or is_productivity),
            "feature_flags": {
                "has_soft_windows": has_soft_windows,
                "is_partial": is_partial,
                "has_time_profiles": has_time_profiles,
                "is_productivity": is_productivity,
            },
        }
