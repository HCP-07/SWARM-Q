"""Micro-benchmark comparing Proposed AD-QPSO against Classical PSO Baseline."""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure workspace root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from configs.config import SimulationConfig
from src.benchmarking.baselines import BaselineRunner


def run_micro_benchmark(scenario: str = "static", agent_count: int = 15, ticks: int = 60, seed: int = 42) -> pd.DataFrame:
    """Executes side-by-side run of Proposed AD-QPSO vs Classical PSO Baseline."""
    print("=" * 70)
    print(f"MICRO-BENCHMARK: Proposed AD-QPSO vs Classical PSO Baseline")
    print(f"Scenario: {scenario}, Agents: {agent_count}, Ticks: {ticks}, Seed: {seed}")
    print("=" * 70)

    cfg = SimulationConfig(seed=seed, agent_count=agent_count)
    cfg.arena.max_ticks = ticks

    # 1. Classical PSO (BASELINE)
    print("Running Classical PSO (BASELINE)...")
    m_pso, _ = BaselineRunner.run_classical_pso(scenario, cfg)

    # 2. Proposed AD-QPSO (PROPOSED)
    print("Running Proposed AD-QPSO (PROPOSED)...")
    m_adqpso, _ = BaselineRunner.run_proposed_adso(scenario, cfg)

    rows = [
        {
            "Role": "BASELINE",
            "Solver": "Classical PSO",
            "Task Completion (%)": round(m_pso.task_completion_rate, 2),
            "Throughput (/min)": round(m_pso.swarm_throughput, 2),
            "Hard Collisions": m_pso.collision_count,
            "Near Collisions": m_pso.near_collision_count,
            "Mean Latency (ms)": round(m_pso.average_tick_latency_ms, 2),
            "P95 Latency (ms)": round(m_pso.p95_latency_ms, 2),
            "Energy (units)": round(m_pso.total_energy_consumed, 1),
            "Objective Fitness": round(m_pso.final_objective_value, 4)
        },
        {
            "Role": "PROPOSED",
            "Solver": "AD-QPSO",
            "Task Completion (%)": round(m_adqpso.task_completion_rate, 2),
            "Throughput (/min)": round(m_adqpso.swarm_throughput, 2),
            "Hard Collisions": m_adqpso.collision_count,
            "Near Collisions": m_adqpso.near_collision_count,
            "Mean Latency (ms)": round(m_adqpso.average_tick_latency_ms, 2),
            "P95 Latency (ms)": round(m_adqpso.p95_latency_ms, 2),
            "Energy (units)": round(m_adqpso.total_energy_consumed, 1),
            "Objective Fitness": round(m_adqpso.final_objective_value, 4)
        }
    ]

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    print("=" * 70)
    return df


if __name__ == "__main__":
    run_micro_benchmark()
