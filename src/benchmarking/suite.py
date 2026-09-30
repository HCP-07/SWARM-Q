"""Automated benchmark suite comparing all baselines across scenarios and seeds."""

from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd

from configs.config import SimulationConfig
from src.benchmarking.baselines import BaselineRunner
from src.metrics.collector import SwarmMetrics


class BenchmarkSuite:
    """Executes comparative evaluation across Greedy, Classical PSO, Random, and ADSO."""

    @staticmethod
    def run_comparison(scenario_name: str, config: SimulationConfig) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []

        # 1. Greedy Nearest Baseline
        m_greedy, _ = BaselineRunner.run_greedy_nearest(scenario_name, config)
        records.append({"Solver": "Greedy Nearest", **m_greedy.to_dict()})

        # 2. Classical PSO Baseline
        m_pso, _ = BaselineRunner.run_classical_pso(scenario_name, config)
        records.append({"Solver": "Classical PSO", **m_pso.to_dict()})

        # 3. Random Local Search Baseline
        m_rand, _ = BaselineRunner.run_random_local_search(scenario_name, config)
        records.append({"Solver": "Random Local Search", **m_rand.to_dict()})

        # 4. Proposed AD-QPSO
        m_adso, _ = BaselineRunner.run_proposed_adso(scenario_name, config)
        records.append({"Solver": "Proposed AD-QPSO", **m_adso.to_dict()})

        df = pd.DataFrame(records)
        return df
