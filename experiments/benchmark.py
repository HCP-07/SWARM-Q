"""Comparative benchmark evaluation across 4 multi-agent algorithms and 7 scenarios.

Evaluates:
1. Greedy Goal Navigation (Baseline 1)
2. Static Priority Navigation (Baseline 2)
3. Non-Adaptive Swarm Navigation (Baseline 3)
4. Proposed Adaptive Swarm Navigation
"""

from __future__ import annotations
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
from typing import Dict, Any, List
from config import SimulationConfig, ScenarioType
from scenarios import get_scenario
from algorithms.greedy import GreedyAgent
from algorithms.static_priority import StaticPriorityAgent
from algorithms.non_adaptive_swarm import NonAdaptiveSwarmAgent
from algorithms.adaptive_swarm import AdaptiveSwarmAgent


def run_benchmark(
    seed: int = 42,
    agent_count: int = 15,
    max_ticks: int = 70,
    output_dir: str = "results"
) -> pd.DataFrame:
    """Executes comparative evaluation across algorithms and scenarios."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    scenarios = [
        ScenarioType.OPEN,
        ScenarioType.STATIC_OBSTACLES,
        ScenarioType.DENSE_SWARM,
        ScenarioType.DYNAMIC_OBSTACLES,
        ScenarioType.COMMUNICATION_DROPOUT,
        ScenarioType.AGENT_FAILURE,
        ScenarioType.COMBINED,
    ]

    algorithms = [
        ("Greedy Goal", GreedyAgent),
        ("Static Priority", StaticPriorityAgent),
        ("Non-Adaptive Swarm", NonAdaptiveSwarmAgent),
        ("Proposed Adaptive Swarm", AdaptiveSwarmAgent),
    ]

    records: List[Dict[str, Any]] = []

    print("=" * 80)
    print("STARTING DECENTRALIZED MULTI-AGENT BENCHMARK EVALUATION")
    print(f"Agents: {agent_count} | Ticks: {max_ticks} | Seed: {seed}")
    print("=" * 80)

    for sc in scenarios:
        sc_name = sc.value if hasattr(sc, "value") else str(sc)
        for algo_name, agent_cls in algorithms:
            cfg = SimulationConfig(
                seed=seed,
                agent_count=agent_count,
                max_ticks=max_ticks,
                scenario=sc,
                enable_adaptation=(algo_name == "Proposed Adaptive Swarm")
            )

            coordinator = get_scenario(sc, config=cfg, agent_cls=agent_cls)
            metrics = coordinator.run()

            rec = {
                "scenario": sc_name,
                "algorithm": algo_name,
                "task_completion_rate": metrics.task_completion_rate,
                "collision_count": metrics.collision_count,
                "collision_rate": metrics.collision_rate,
                "avg_latency_ms": metrics.average_decision_latency_ms,
                "p95_latency_ms": metrics.p95_decision_latency_ms,
                "real_time_pass": metrics.real_time_pass,
                "deadlock_count": metrics.deadlock_count,
                "deadlock_resolution_rate": metrics.deadlock_resolution_rate,
                "avg_path_length": metrics.average_path_length,
                "total_energy": metrics.total_energy_cost,
                "task_throughput": metrics.task_throughput,
                "coverage_velocity": metrics.coverage_velocity,
                "final_objective": metrics.final_objective_value,
            }
            records.append(rec)
            print(
                f"[{sc_name.upper():<10}] {algo_name:<24} | "
                f"Completion: {metrics.task_completion_rate:>5.1f}% | "
                f"Collisions: {metrics.collision_count:>2} | "
                f"Latency: {metrics.average_decision_latency_ms:>5.2f}ms | "
                f"Deadlocks Resolved: {metrics.deadlock_resolution_rate:>5.1f}%"
            )

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "benchmark.csv")
    json_path = os.path.join(output_dir, "benchmark.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("=" * 80)
    print(f"Benchmark completed successfully! Results written to:\n - {csv_path}\n - {json_path}")
    print("=" * 80)
    return df


if __name__ == "__main__":
    run_benchmark()
