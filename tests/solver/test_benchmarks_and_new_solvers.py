"""Unit tests for Genetic Algorithm, Dijkstra Heuristic, and Multi-Algorithm Benchmarking."""

import pytest
from solver.benchmark import (
    compute_multi_objective_metrics,
    run_algorithm_benchmark,
    run_scalability_analysis,
)
from solver.dijkstra import DijkstraConfig, solve_dijkstra
from solver.ga import GAConfig, solve_ga
from solver.models import Customer, Depot, Problem, Vehicle


@pytest.fixture
def sample_problem() -> Problem:
    depot = Depot(id="depot", lat=22.3072, lon=73.1812, ready_time=0.0, due_time=25000.0)
    customers = [
        Customer(id="c1", lat=22.3125, lon=73.1785, demand=5.0, ready_time=0.0, due_time=20000.0, service_time=60.0),
        Customer(id="c2", lat=22.2980, lon=73.1650, demand=8.0, ready_time=0.0, due_time=20000.0, service_time=60.0),
        Customer(id="c3", lat=22.3250, lon=73.1900, demand=6.0, ready_time=0.0, due_time=20000.0, service_time=60.0),
        Customer(id="c4", lat=22.3150, lon=73.1420, demand=4.0, ready_time=0.0, due_time=20000.0, service_time=60.0),
    ]
    vehicles = [
        Vehicle(id="V1", capacity=20.0, shift_start=0.0, shift_end=25000.0),
        Vehicle(id="V2", capacity=20.0, shift_start=0.0, shift_end=25000.0),
    ]
    prob = Problem(
        id="unit_test_prob",
        depot=depot,
        customers=customers,
        vehicles=vehicles,
        coordinate_system="geographic",
    )
    prob.compute_euclidean_matrices()
    return prob


def test_genetic_algorithm_solve(sample_problem: Problem):
    cfg = GAConfig(population_size=12, max_generations=10, seed=42)
    sol, hist = solve_ga(sample_problem, cfg)

    assert "GeneticAlgorithm" in sol.algorithm
    assert sol.is_feasible is True
    assert len(sol.routes) > 0
    assert len(hist) == cfg.max_generations + 1
    # Check that demand served matches total customer demand
    total_load = sum(r.load for r in sol.routes)
    expected_load = sum(c.demand for c in sample_problem.customers)
    assert pytest.approx(total_load) == expected_load


def test_dijkstra_heuristic_solve(sample_problem: Problem):
    cfg = DijkstraConfig(seed=42)
    sol, hist = solve_dijkstra(sample_problem, cfg)

    assert sol.algorithm == "DijkstraHeuristic"
    assert sol.is_feasible is True
    assert len(sol.routes) > 0
    assert len(hist) > 0

    visited_stops = {s_id for r in sol.routes for s_id in r.stop_ids}
    assert visited_stops == {c.id for c in sample_problem.customers}


def test_benchmark_suite_execution(sample_problem: Problem):
    bench_result = run_algorithm_benchmark(
        problem=sample_problem,
        algorithms=["qpso", "pso", "ga", "dijkstra", "ortools"],
        num_particles_or_pop=10,
        max_iters_or_gen=5,
        seed=42,
    )

    assert bench_result.problem_id == sample_problem.id
    assert len(bench_result.results) == 5
    algo_names = {r.algorithm for r in bench_result.results}
    assert algo_names == {"qpso", "pso", "ga", "dijkstra", "ortools"}

    for r in bench_result.results:
        assert r.total_distance_km > 0.0
        assert r.operational_cost_inr > 0.0
        assert r.execution_time_ms >= 0.0
        assert len(r.convergence_history) > 0


def test_multi_objective_metrics_calculation(sample_problem: Problem):
    cfg = DijkstraConfig(seed=42)
    sol, _ = solve_dijkstra(sample_problem, cfg)

    metrics = compute_multi_objective_metrics(sample_problem, sol)
    assert "dist_km" in metrics
    assert "dur_min" in metrics
    assert "traffic_delay_min" in metrics
    assert "operational_inr" in metrics
    assert metrics["dist_km"] > 0
    assert metrics["operational_inr"] > 0
