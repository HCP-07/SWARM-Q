"""Scenario definitions and builder facade delegating to modular scenario builders."""

from __future__ import annotations
from enum import Enum
from typing import Dict, Any, List
import numpy as np

from configs.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from scenarios.static import populate_base_world, build_static_scenario
from scenarios.dynamic_obstacle import build_dynamic_obstacle_scenario
from scenarios.multiple_moving_obstacles import build_multiple_moving_obstacles_scenario
from scenarios.communication_dropout import build_communication_dropout_scenario
from scenarios.agent_failure import build_agent_failure_scenario
from scenarios.stress import build_stress_scenario


class ScenarioType(str, Enum):
    """Enumeration of standard benchmark scenario profiles."""
    STATIC = "static"
    DYNAMIC_OBSTACLE = "dynamic_obstacle"
    MULTIPLE_MOVING_OBSTACLES = "multiple_moving_obstacles"
    COMMUNICATION_DROPOUT = "communication_dropout"
    AGENT_FAILURE = "agent_failure"
    STRESS = "stress"
    FULL_STRESS = "full_stress"  # Supported alias for canonical 'stress'


class ScenarioBuilder:
    """Facade constructing deterministic benchmark environments."""

    populate_base_world = staticmethod(populate_base_world)

    @classmethod
    def build(
        cls,
        scenario: str | ScenarioType,
        config: SimulationConfig
    ) -> SimulationEngine:
        """Constructs a fully configured SimulationEngine for given scenario name."""
        if isinstance(scenario, str):
            scenario_name = scenario.lower()
        else:
            scenario_name = scenario.value

        if scenario_name == ScenarioType.STATIC.value:
            return build_static_scenario(config)
        elif scenario_name == ScenarioType.DYNAMIC_OBSTACLE.value:
            return build_dynamic_obstacle_scenario(config)
        elif scenario_name == ScenarioType.MULTIPLE_MOVING_OBSTACLES.value:
            return build_multiple_moving_obstacles_scenario(config)
        elif scenario_name == ScenarioType.COMMUNICATION_DROPOUT.value:
            return build_communication_dropout_scenario(config)
        elif scenario_name == ScenarioType.AGENT_FAILURE.value:
            return build_agent_failure_scenario(config)
        elif scenario_name in (ScenarioType.STRESS.value, ScenarioType.FULL_STRESS.value):
            return build_stress_scenario(config)
        else:
            valid_names = [s.value for s in ScenarioType]
            raise ValueError(f"Unknown scenario: '{scenario_name}'. Valid options: {valid_names}")
