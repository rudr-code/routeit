from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from routeit_engine.contracts import (
    Order,
    Plan,
    PlanningMode,
    ProfileKind,
)
from routeit_engine.context import EvaluationContext
from routeit_engine.oracle import RoadOracle
from routeit_engine.policy import Calibration, compare_plans
from routeit_engine.split import SplitDecoder


class GAStrategy:
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
        mutation_rate: float = 0.2,
        seed: int = 42,
    ) -> Plan:
        order_map = {o.id: o for o in orders}
        n = len(orders)
        if n == 0:
            from routeit_engine.evaluator import evaluate_plan
            return evaluate_plan([], orders, ctx, oracle)

        rng = random.Random(seed)

        # Generate initial population of permutations
        population: List[List[Order]] = []

        # From seeds
        for sp in seed_plans:
            tour: List[Order] = []
            for r in sp.routes:
                for s in r.stops:
                    if s.order_id in order_map:
                        tour.append(order_map[s.order_id])
            for o in orders:
                if o not in tour:
                    tour.append(o)
            if len(tour) == n:
                population.append(tour)

        while len(population) < pop_size:
            p = list(orders)
            rng.shuffle(p)
            population.append(p)

        def decode_and_eval(tour: Sequence[Order]) -> Plan:
            return SplitDecoder.split(
                tour,
                ctx,
                oracle,
                calibration=calibration,
                c_cap=c_cap,
                exact_tiny_mode=(n <= 6),
            )

        def order_crossover(p1: Sequence[Order], p2: Sequence[Order]) -> List[Order]:
            if n <= 2:
                return list(p1)
            cx1 = rng.randint(0, n - 2)
            cx2 = rng.randint(cx1 + 1, n - 1)
            child = [None] * n
            child[cx1:cx2] = p1[cx1:cx2]
            p1_set = set(p1[cx1:cx2])
            fill_items = [item for item in p2 if item not in p1_set]
            fill_idx = 0
            for i in range(n):
                if child[i] is None:
                    child[i] = fill_items[fill_idx]
                    fill_idx += 1
            return child

        # Initial evaluation
        scored_pop = [(decode_and_eval(tour), tour) for tour in population]
        best_plan = scored_pop[0][0]

        for plan, _ in scored_pop:
            if compare_plans(plan, best_plan, PlanningMode(ctx.planning_mode), ProfileKind(ctx.profile_kind), order_map, calibration, c_cap) < 0:
                best_plan = plan

        for gen in range(generations):
            # Sort population by Plan quality
            scored_pop.sort(
                key=lambda item: item[0],
                # Custom comparator sorting helper
            )
            # Tournament selection & elitism
            new_pop: List[List[Order]] = []
            # Keep top 2 elites
            elites = [item[1] for item in scored_pop[:2]]
            new_pop.extend(elites)

            while len(new_pop) < pop_size:
                # Tournament 3
                t1 = rng.sample(scored_pop, min(3, len(scored_pop)))
                t1.sort(key=lambda item: item[0].c_dec)
                parent1 = t1[0][1]

                t2 = rng.sample(scored_pop, min(3, len(scored_pop)))
                t2.sort(key=lambda item: item[0].c_dec)
                parent2 = t2[0][1]

                child = order_crossover(parent1, parent2)

                # Mutation
                if rng.random() < mutation_rate and n >= 2:
                    m_type = rng.choice(["swap", "invert"])
                    i, j = rng.sample(range(n), 2)
                    if m_type == "swap":
                        child[i], child[j] = child[j], child[i]
                    else:
                        start_idx, end_idx = min(i, j), max(i, j)
                        child[start_idx:end_idx + 1] = reversed(child[start_idx:end_idx + 1])

                new_pop.append(child)

            population = new_pop
            scored_pop = [(decode_and_eval(tour), tour) for tour in population]
            for plan, _ in scored_pop:
                if compare_plans(plan, best_plan, PlanningMode(ctx.planning_mode), ProfileKind(ctx.profile_kind), order_map, calibration, c_cap) < 0:
                    best_plan = plan

        return best_plan
