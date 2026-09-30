"""Multi-seed statistical evaluation script producing results/benchmark_stats.csv.
Computes Mean ± Std across 10 fixed random seeds for Proposed AD-QPSO, Classical PSO, and Greedy Nearest.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from configs.config import SimulationConfig
from src.benchmarking.baselines import BaselineRunner


def run_multi_seed_evaluation(seeds=list(range(42, 52)), scenario="static", ticks=40, agents=10):
    print(f"Executing Multi-Seed Benchmark across {len(seeds)} seeds: {seeds}")
    print(f"Scenario: {scenario}, Agents: {agents}, Ticks: {ticks}")
    
    results = []
    
    for seed in seeds:
        print(f"  -> Running seed {seed}...")
        cfg = SimulationConfig(seed=seed, agent_count=agents)
        cfg.arena.max_ticks = ticks
        
        # 1. Greedy Nearest
        m_greedy, _ = BaselineRunner.run_greedy_nearest(scenario, cfg)
        d_greedy = m_greedy.to_dict()
        d_greedy["Solver"] = "Greedy Nearest"
        d_greedy["seed"] = seed
        results.append(d_greedy)
        
        # 2. Classical PSO
        m_pso, _ = BaselineRunner.run_classical_pso(scenario, cfg)
        d_pso = m_pso.to_dict()
        d_pso["Solver"] = "Classical PSO"
        d_pso["seed"] = seed
        results.append(d_pso)
        
        # 3. Proposed AD-QPSO
        m_adso, _ = BaselineRunner.run_proposed_adso(scenario, cfg)
        d_adso = m_adso.to_dict()
        d_adso["Solver"] = "Proposed AD-QPSO"
        d_adso["seed"] = seed
        results.append(d_adso)

    raw_df = pd.DataFrame(results)
    
    # Compute aggregate Mean ± Std
    metrics_to_agg = [
        "task_completion_rate",
        "swarm_throughput",
        "collision_count",
        "near_collision_count",
        "average_tick_latency_ms",
        "p95_latency_ms",
        "total_energy_consumed",
        "total_path_length",
        "final_objective_value"
    ]
    
    summary_rows = []
    for solver in ["Greedy Nearest", "Classical PSO", "Proposed AD-QPSO"]:
        sub = raw_df[raw_df["Solver"] == solver]
        row = {"Solver": solver, "Sample_Count": len(sub)}
        for m in metrics_to_agg:
            mean_v = float(sub[m].mean())
            std_v = float(sub[m].std())
            row[f"{m}_mean"] = round(mean_v, 3)
            row[f"{m}_std"] = round(std_v, 3)
            row[f"{m}_formatted"] = f"{mean_v:.2f} ± {std_v:.2f}"
        summary_rows.append(row)
        
    stats_df = pd.DataFrame(summary_rows)
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "benchmark_stats.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"\nSuccessfully generated {out_path}:")
    print(stats_df[["Solver", "task_completion_rate_formatted", "collision_count_formatted", "p95_latency_ms_formatted"]].to_string(index=False))


if __name__ == "__main__":
    run_multi_seed_evaluation()
