"""Configuration parameters for Decentralized Adaptive Swarm Navigation.

Defines typed dataclasses for simulation bounds, agent physical parameters,
adaptive heuristic scoring weights, and real-time execution budgets.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List, Dict, Any


class ScenarioType(str, Enum):
    """Supported multi-agent benchmark scenarios."""
    OPEN = "open"
    STATIC_OBSTACLES = "obstacles"
    DENSE_SWARM = "dense"
    DYNAMIC_OBSTACLES = "dynamic"
    COMMUNICATION_DROPOUT = "dropout"
    AGENT_FAILURE = "failure"
    COMBINED = "combined"


class Action(Enum):
    """9 candidate discrete movement actions in 2D space."""
    WAIT = (0, 0)
    UP = (0, 1)
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP_LEFT = (-1, 1)
    UP_RIGHT = (1, 1)
    DOWN_LEFT = (-1, -1)
    DOWN_RIGHT = (1, -1)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]

    @property
    def step_cost(self) -> float:
        """Euclidean movement cost (diagonal moves cost ~1.414)."""
        if self == Action.WAIT:
            return 0.1
        if self.dx != 0 and self.dy != 0:
            return 1.4142
        return 1.0


@dataclass(frozen=True)
class EnvironmentConfig:
    """2D spatial grid world environment limits."""
    width: int = 40
    height: int = 40
    boundary_padding: int = 1

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """(x_min, x_max, y_min, y_max)"""
        return (0, self.width - 1, 0, self.height - 1)


@dataclass(frozen=True)
class AgentConfig:
    """Individual autonomous agent physical and operational specifications."""
    communication_radius: float = 6.0
    sensing_radius: float = 5.0
    safety_margin: float = 1.0
    initial_energy: float = 100.0
    energy_move_cost: float = 1.0
    energy_wait_cost: float = 0.1
    stagnation_threshold: int = 4  # Consecutive ticks without progress before stagnation trigger
    max_history_length: int = 100


@dataclass
class AdaptiveWeightsConfig:
    """Base weights for the candidate action evaluation heuristic."""
    w_goal: float = 3.0
    w_collision: float = 4.5
    w_obstacle: float = 4.0
    w_conflict: float = 2.5
    w_energy: float = 0.5
    w_momentum: float = 0.8
    w_exploration: float = 0.5

    # Adaptation scale limits
    min_goal_weight: float = 0.5
    max_goal_weight: float = 5.0
    max_collision_weight: float = 10.0
    max_exploration_weight: float = 3.0


@dataclass
class SimulationConfig:
    """Global simulation execution parameters."""
    seed: int = 42
    agent_count: int = 15
    max_ticks: int = 80
    real_time_budget_ms: float = 100.0  # SLA latency limit
    scenario: ScenarioType = ScenarioType.OPEN
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    weights: AdaptiveWeightsConfig = field(default_factory=AdaptiveWeightsConfig)
    enable_adaptation: bool = True
