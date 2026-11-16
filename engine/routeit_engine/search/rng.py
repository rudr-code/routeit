from __future__ import annotations

import math
import random
from typing import Tuple

TWO_POW_53 = 1 << 53  # 9007199254740992


class DiscreteUniformRNG:
    """
    Open-interval discrete pseudo-uniform generator in (0, 1):
    Sample integer k in {1, ..., 2^53 - 1}, return k / 2^53.
    Guarantees that 0.0 and 1.0 are NEVER drawn.
    """
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def draw_open_unit(self) -> float:
        # Sample integer in [1, 2^53 - 1]
        k = self.rng.randint(1, TWO_POW_53 - 1)
        val = k / float(TWO_POW_53)
        assert 0.0 < val < 1.0, f"Drawn value {val} must be strictly in (0, 1)"
        return val

    def draw_sign(self) -> int:
        """Draw sigma in {-1, +1} with equal probability P(-1) = P(+1) = 1/2."""
        return -1 if self.rng.random() < 0.5 else 1
