from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from routeit_engine.contracts import (
    Order,
    Plan,
    PlanningMode,
    ProfileKind,
    ResultStatus,
    RunResult,
    StopReason,
)
from routeit_engine.context import EvaluationContext
from routeit_engine.oracle import RoadOracle
from routeit_engine.policy import Calibration, compare_plans
from routeit_engine.search.rng import DiscreteUniformRNG
from routeit_engine.split import SplitDecoder


def reflect(y: float) -> float:
    """
    Triangular-wave reflection mapping into [0, 1]:
    r = y - 2 * floor(y / 2)
    return r if r <= 1 else 2 - r, with 0 <= r < 2.
    """
    r = y - 2.0 * math.floor(y / 2.0)
    return r if r <= 1.0 else (2.0 - r)


def compute_mbest(pbest_positions: Sequence[Sequence[float]]) -> List[float]:
    """mbest_d(t) = (1/P) sum_{i=1..P} pbest_{i,d}(t)"""
    P = len(pbest_positions)
    if P == 0:
        return []
    n_dim = len(pbest_positions[0])
    mbest = [0.0] * n_dim
    for d in range(n_dim):
        mbest[d] = sum(pbest_positions[i][d] for i in range(P)) / float(P)
    return mbest


def contraction_schedule(t: int, T: int, alpha_max: float, alpha_min: float) -> float:
    """Linear contraction schedule: alpha_t = alpha_max - (alpha_max - alpha_min)*t / (T - 1)"""
    if T <= 1 or t <= 0:
        return alpha_max
    if t >= T - 1:
        return alpha_min
    return alpha_max - (alpha_max - alpha_min) * (float(t) / float(T - 1))


def qpso_position_update(
    x: float,
    pbest: float,
    gbest: float,
    mbest: float,
    phi: float,
    u: float,
    sigma: int,
    alpha: float,
) -> Tuple[float, float]:
    """
    Computes local attractor p and updated position y:
    p = phi * pbest + (1 - phi) * gbest
    y = p + sigma * alpha * |mbest - x| * ln(1 / u)
    Returns (p, y)
    """
    if u <= 0.0 or u >= 1.0:
        raise ValueError(f"u must be in open interval (0, 1), got {u}")
    if math.isnan(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be finite positive, got {alpha}")

    p = phi * pbest + (1.0 - phi) * gbest
    step_val = sigma * alpha * abs(mbest - x) * math.log(1.0 / u)
    y = p + step_val
    return p, y


@dataclass
class Particle:
    position: List[float]
    pbest_position: List[float]
    pbest_plan: Plan
    current_plan: Optional[Plan] = None


class QPSOStrategy:
    @staticmethod
    def solve(
        orders: Sequence[Order],
        ctx: EvaluationContext,
        oracle: RoadOracle,
        seed_plans: Sequence[Plan],
        calibration: Optional[Calibration] = None,
        c_cap: Optional[int] = None,
        pop_size: int = 15,
        generations: int = 20,
        alpha_max: float = 1.0,
        alpha_min: float = 0.5,
        seed: int = 42,
    ) -> Plan:
        order_map = {o.id: o for o in orders}
        n = len(orders)
        if n == 0:
            return evaluate_plan([], orders, ctx, oracle)

        # Check for eligible feasible seed
        feasible_seeds = [p for p in seed_plans if p.is_feasible]
        if ctx.profile_kind == ProfileKind.DRIVER_PRODUCTIVITY and c_cap is not None:
            feasible_seeds = [p for p in feasible_seeds if p.c_dec <= c_cap]

        if not feasible_seeds:
            # Cannot launch QPSO without an eligible attractor
            raise ValueError("QPSO requires at least one eligible feasible seed attractor")

        rng = DiscreteUniformRNG(seed=seed)

        # Determine initial gbest
        gbest_plan = feasible_seeds[0]
        for sp in feasible_seeds[1:]:
            if compare_plans(sp, gbest_plan, PlanningMode(ctx.planning_mode), ProfileKind(ctx.profile_kind), order_map, calibration, c_cap) < 0:
                gbest_plan = sp

        # Decode helper: given position vector x in [0, 1]^n, sort orders by (x_i, o.id)
        def decode_position(pos: Sequence[float]) -> Plan:
            paired = sorted(zip(pos, orders), key=lambda item: (item[0], item[1].id))
            tour = [item[1] for item in paired]
            return SplitDecoder.split(
                tour,
                ctx,
                oracle,
                calibration=calibration,
                c_cap=c_cap,
                exact_tiny_mode=(n <= 6),
            )

        # Initialize particles
        particles: List[Particle] = []
        for p_idx in range(pop_size):
            init_pos = [rng.draw_open_unit() for _ in range(n)]
            plan = decode_position(init_pos)

            if plan.is_feasible and (ctx.profile_kind != ProfileKind.DRIVER_PRODUCTIVITY or c_cap is None or plan.c_dec <= c_cap):
                pbest_plan = plan
                pbest_pos = list(init_pos)
            else:
                pbest_plan = gbest_plan
                # Seed position from gbest order
                order_index = {o.id: idx for idx, o in enumerate(orders)}
                gbest_pos = [0.5] * n
                pbest_pos = gbest_pos

            particles.append(Particle(position=init_pos, pbest_position=pbest_pos, pbest_plan=pbest_plan, current_plan=plan))

            if compare_plans(pbest_plan, gbest_plan, PlanningMode(ctx.planning_mode), ProfileKind(ctx.profile_kind), order_map, calibration, c_cap) < 0:
                gbest_plan = pbest_plan

        # Synchronous generations loop
        for gen in range(generations):
            alpha_t = contraction_schedule(gen, generations, alpha_max, alpha_min)
            mbest = compute_mbest([p.pbest_position for p in particles])
            # Best position from gbest_plan
            gbest_pos = particles[0].pbest_position  # Fallback

            # Find which particle owns gbest
            for p in particles:
                if p.pbest_plan == gbest_plan:
                    gbest_pos = p.pbest_position
                    break

            new_positions: List[List[float]] = []
            for p_idx, particle in enumerate(particles):
                new_pos = [0.0] * n
                for d in range(n):
                    phi = rng.draw_open_unit()
                    u = rng.draw_open_unit()
                    sigma = rng.draw_sign()
                    _, y = qpso_position_update(
                        x=particle.position[d],
                        pbest=particle.pbest_position[d],
                        gbest=gbest_pos[d],
                        mbest=mbest[d],
                        phi=phi,
                        u=u,
                        sigma=sigma,
                        alpha=alpha_t,
                    )
                    # Apply reflection
                    new_pos[d] = reflect(y)
                new_positions.append(new_pos)

            # Evaluate synchronous candidate batch
            for p_idx, particle in enumerate(particles):
                particle.position = new_positions[p_idx]
                plan = decode_position(particle.position)
                particle.current_plan = plan

                if plan.is_feasible and (ctx.profile_kind != ProfileKind.DRIVER_PRODUCTIVITY or c_cap is None or plan.c_dec <= c_cap):
                    # Compare with pbest
                    if compare_plans(plan, particle.pbest_plan, PlanningMode(ctx.planning_mode), ProfileKind(ctx.profile_kind), order_map, calibration, c_cap) < 0:
                        particle.pbest_plan = plan
                        particle.pbest_position = list(particle.position)

                        if compare_plans(plan, gbest_plan, PlanningMode(ctx.planning_mode), ProfileKind(ctx.profile_kind), order_map, calibration, c_cap) < 0:
                            gbest_plan = plan

        return gbest_plan
