"""5-stage component ablation study isolating the contribution of adaptive heuristics."""

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
from algorithms.non_adaptive_swarm import NonAdaptiveSwarmAgent
from algorithms.adaptive_swarm import AdaptiveSwarmAgent


def run_ablation(
    seed: int = 42,
    agent_count: int = 15,
    max_ticks: int = 70,
    output_dir: str = "results"
) -> pd.DataFrame:
    """Runs the 5-stage ablation matrix."""
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        {
            "id": "A",
            "name": "Greedy Baseline",
            "description": "Pure Euclidean greedy goal navigation without swarm coordination",
            "scenario": ScenarioType.STATIC_OBSTACLES,
            "agent_cls": GreedyAgent,
            "adaptation": False
        },
        {
            "id": "B",
            "name": "Non-Adaptive Swarm",
            "description": "Decentralized swarm heuristic with static, unadapted weights",
            "scenario": ScenarioType.STATIC_OBSTACLES,
            "agent_cls": NonAdaptiveSwarmAgent,
            "adaptation": False
        },
        {
            "id": "C",
            "name": "Proposed Adaptive Swarm",
            "description": "Full adaptive heuristic policy (collision threat, stagnation, clear path)",
            "scenario": ScenarioType.STATIC_OBSTACLES,
            "agent_cls": AdaptiveSwarmAgent,
            "adaptation": True
        },
        {
            "id": "D",
            "name": "Adaptive + Dynamic Obstacles",
            "description": "Proposed adaptive swarm under dynamic obstacle intercepts",
            "scenario": ScenarioType.DYNAMIC_OBSTACLES,
            "agent_cls": AdaptiveSwarmAgent,
            "adaptation": True
        },
        {
            "id": "E",
            "name": "Adaptive + Agent Failure",
            "description": "Proposed adaptive swarm operating through mid-mission hardware crash",
            "scenario": ScenarioType.AGENT_FAILURE,
            "agent_cls": AdaptiveSwarmAgent,
            "adaptation": True
        },
    ]

    records: List[Dict[str, Any]] = []

    print("=" * 80)
    print("STARTING 5-STAGE COMPONENT ABLATION STUDY")
    print(f"Agents: {agent_count} | Ticks: {max_ticks} | Seed: {seed}")
    print("=" * 80)

    for cfg_info in configs:
        cfg = SimulationConfig(
            seed=seed,
            agent_count=agent_count,
            max_ticks=max_ticks,
            scenario=cfg_info["scenario"],
            enable_adaptation=cfg_info["adaptation"]
        )

        coord = get_scenario(
            cfg_info["scenario"],
            config=cfg,
            agent_cls=cfg_info["agent_cls"]
        )
        metrics = coord.run()

        rec = {
            "config_id": cfg_info["id"],
            "config_name": cfg_info["name"],
            "description": cfg_info["description"],
            "scenario": cfg_info["scenario"].value,
            "task_completion_rate": metrics.task_completion_rate,
            "collision_count": metrics.collision_count,
            "avg_latency_ms": metrics.average_decision_latency_ms,
            "deadlock_count": metrics.deadlock_count,
            "deadlock_resolution_rate": metrics.deadlock_resolution_rate,
            "avg_path_length": metrics.average_path_length,
            "total_energy": metrics.total_energy_cost,
            "final_objective": metrics.final_objective_value,
        }
        records.append(rec)
        print(
            f"Config {cfg_info['id']}: {cfg_info['name']:<30} | "
            f"Completion: {metrics.task_completion_rate:>5.1f}% | "
            f"Collisions: {metrics.collision_count:>2} | "
            f"Deadlocks: {metrics.deadlock_count:>2} (Resolved: {metrics.deadlock_resolution_rate:>5.1f}%)"
        )

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "ablation.csv")
    json_path = os.path.join(output_dir, "ablation.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("=" * 80)
    print(f"Ablation study complete! Results written to:\n - {csv_path}\n - {json_path}")
    print("=" * 80)
    return df


if __name__ == "__main__":
    run_ablation()
