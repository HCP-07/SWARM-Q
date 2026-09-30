"""Multi-seed repeated evaluation for statistical significance and variance analysis."""

from __future__ import annotations
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from config import SimulationConfig, ScenarioType
from scenarios import get_scenario
from algorithms.adaptive_swarm import AdaptiveSwarmAgent


def run_statistics(
    runs: int = 30,
    agent_count: int = 15,
    max_ticks: int = 70,
    scenario: ScenarioType = ScenarioType.STATIC_OBSTACLES,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """Executes N runs across independent seeds and computes statistical metrics."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"STARTING STATISTICAL SIGNIFICANCE STUDY ({runs} REPEATED RUNS)")
    print(f"Scenario: {scenario.value} | Agents: {agent_count} | Ticks: {max_ticks}")
    print("=" * 80)

    run_records: List[Dict[str, Any]] = []

    for seed in range(1, runs + 1):
        cfg = SimulationConfig(
            seed=seed,
            agent_count=agent_count,
            max_ticks=max_ticks,
            scenario=scenario,
            enable_adaptation=True
        )

        coord = get_scenario(scenario, config=cfg, agent_cls=AdaptiveSwarmAgent)
        metrics = coord.run()

        m_dict = metrics.to_dict()
        m_dict["seed"] = seed
        run_records.append(m_dict)

        if seed % 5 == 0 or seed == runs:
            print(f"Run {seed:>2}/{runs}: Completion={metrics.task_completion_rate:>5.1f}% | Collisions={metrics.collision_count} | Latency={metrics.average_decision_latency_ms:>5.2f}ms")

    # Compute statistical aggregates
    keys_to_aggregate = [
        "task_completion_rate",
        "collision_count",
        "average_decision_latency_ms",
        "p95_decision_latency_ms",
        "deadlock_count",
        "deadlock_resolution_rate",
        "average_path_length",
        "total_energy_cost",
        "task_throughput",
        "coverage_velocity",
        "final_objective_value"
    ]

    stats_summary: Dict[str, Dict[str, float]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for k in keys_to_aggregate:
        vals = [r[k] for r in run_records]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        median_val = float(np.median(vals))

        stats_summary[k] = {
            "mean": round(mean_val, 3),
            "std": round(std_val, 3),
            "min": round(min_val, 3),
            "max": round(max_val, 3),
            "median": round(median_val, 3)
        }

        summary_rows.append({
            "metric": k,
            "mean": round(mean_val, 3),
            "std": round(std_val, 3),
            "min": round(min_val, 3),
            "max": round(max_val, 3),
            "median": round(median_val, 3)
        })

    # Save outputs
    results_json = {
        "metadata": {
            "runs": runs,
            "agent_count": agent_count,
            "max_ticks": max_ticks,
            "scenario": scenario.value,
        },
        "summary": stats_summary,
        "raw_runs": run_records
    }

    json_path = os.path.join(output_dir, "results.json")
    csv_path = os.path.join(output_dir, "summary.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    print("=" * 80)
    print(f"Statistical study complete! Results written to:\n - {json_path}\n - {csv_path}")
    print("=" * 80)
    return results_json


if __name__ == "__main__":
    run_statistics(runs=30)
