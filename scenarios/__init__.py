"""Scenario registry and factory functions for benchmark environments."""

from __future__ import annotations
from typing import Type, Optional
from config import ScenarioType, SimulationConfig
from core.swarm import SwarmCoordinator
from core.agent import AutonomousAgent
from algorithms.adaptive_swarm import AdaptiveSwarmAgent
from scenarios.open import build_open_scenario
from scenarios.obstacles import build_obstacles_scenario, build_dense_scenario
from scenarios.dynamic import build_dynamic_scenario, build_dropout_scenario
from scenarios.failure import build_failure_scenario
from scenarios.combined import build_combined_scenario


def get_scenario(
    scenario_type: ScenarioType | str,
    config: Optional[SimulationConfig] = None,
    agent_cls: Type[AutonomousAgent] = AdaptiveSwarmAgent
) -> SwarmCoordinator:
    """Builds and returns the configured scenario coordinator.
    
    Args:
        scenario_type: Name or ScenarioType enum value.
        config: Simulation configuration parameters.
        agent_cls: Agent class to instantiate (e.g. AdaptiveSwarmAgent, GreedyAgent).
    
    Returns:
        SwarmCoordinator initialized with the selected scenario.
    """
    cfg = config or SimulationConfig()
    st = str(scenario_type).lower()
    if "." in st:
        st = st.split(".")[-1]

    if st in ("open", "scenario1", "1"):
        return build_open_scenario(cfg, agent_cls=agent_cls)
    elif st in ("obstacles", "static_obstacles", "scenario2", "2"):
        return build_obstacles_scenario(cfg, agent_cls=agent_cls)
    elif st in ("dense", "dense_swarm", "scenario3", "3"):
        return build_dense_scenario(cfg, agent_cls=agent_cls)
    elif st in ("dynamic", "dynamic_obstacles", "scenario4", "4"):
        return build_dynamic_scenario(cfg, agent_cls=agent_cls)
    elif st in ("dropout", "communication_dropout", "scenario5", "5"):
        return build_dropout_scenario(cfg, agent_cls=agent_cls)
    elif st in ("failure", "agent_failure", "scenario6", "6"):
        return build_failure_scenario(cfg, agent_cls=agent_cls)
    elif st in ("combined", "scenario7", "7"):
        return build_combined_scenario(cfg, agent_cls=agent_cls)
    else:
        raise ValueError(f"Unknown scenario: {scenario_type}. Supported: open, obstacles, dense, dynamic, dropout, failure, combined")


__all__ = [
    "get_scenario",
    "build_open_scenario",
    "build_obstacles_scenario",
    "build_dense_scenario",
    "build_dynamic_scenario",
    "build_dropout_scenario",
    "build_failure_scenario",
    "build_combined_scenario",
]
