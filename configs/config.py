"""Configuration specifications for the Adaptive Decentralized Swarm Intelligence system."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class ArenaConfig:
    """Continuous 2D environment boundaries and discretization."""
    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = 0.0
    y_max: float = 100.0
    time_step: float = 0.1  # seconds (dt)
    max_ticks: int = 200

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)


@dataclass
class AgentPhysicalConfig:
    """Physical, kinematic, and sensory attributes of autonomous swarm agents."""
    safety_radius: float = 1.5       # Hard collision boundary (meters)
    operational_margin: float = 0.8  # Extra buffer around obstacles (meters)
    sensor_radius: float = 18.0      # Vision/sensing horizon (meters)
    communication_radius: float = 25.0 # Wireless mesh radius (meters)
    max_speed: float = 3.0           # m/s
    max_acceleration: float = 4.0    # m/s^2
    initial_energy: float = 100.0    # Battery units (Joules/normalized)
    idle_energy_rate: float = 0.02   # Energy drained per tick idling
    motion_energy_rate: float = 0.15 # Energy per unit velocity squared per tick


@dataclass
class FitnessWeightsConfig:
    """Configurable multi-objective fitness weights. Normalized in evaluation."""
    w_distance: float = 0.35        # Target attraction / relative distance penalty
    w_delay: float = 0.20           # Progress shortfall / delay penalty
    w_collision: float = 0.20       # Inter-agent proximity risk
    w_obstacle: float = 0.15        # Obstacle proximity risk
    w_energy: float = 0.05          # Maneuver acceleration penalty
    w_conflict: float = 0.03        # Duplicate task claiming penalty
    w_communication: float = 0.02   # Coordination / message overhead

    def validate(self) -> None:
        total = (self.w_distance + self.w_energy + self.w_collision +
                 self.w_obstacle + self.w_conflict + self.w_delay +
                 self.w_communication)
        if not np.isclose(total, 1.0, atol=1e-3):
            # Normalizing weights if not strictly summing to 1.0
            scale = 1.0 / total
            self.w_distance *= scale
            self.w_energy *= scale
            self.w_collision *= scale
            self.w_obstacle *= scale
            self.w_conflict *= scale
            self.w_delay *= scale
            self.w_communication *= scale


@dataclass
class OptimizationParameters:
    """Adaptive Swarm and QPSO algorithmic search parameters."""
    horizon_steps: int = 10         # Lookahead trajectory horizon (steps)
    candidate_count: int = 16       # Number of candidate trajectories evaluated
    beta_initial: float = 0.70      # QPSO contraction-expansion coefficient
    beta_min: float = 0.30          # Minimum beta (pure exploitation)
    beta_max: float = 1.20          # Maximum beta (aggressive exploration)
    diversity_threshold: float = 3.5 # Minimum local swarm diversity (meters)
    stagnation_limit: int = 4       # Consecutive stagnant ticks before Cauchy mutation
    cauchy_scale: float = 0.5       # Mutation scale parameter
    collision_escalation_factor: float = 2.5 # Multiplier when in critical safety zone


@dataclass
class SafetyConstraintsConfig:
    """Hard safety invariant thresholds."""
    min_agent_separation: float = 3.0  # 2 * safety_radius
    min_obstacle_distance: float = 1.5 # Obstacle surface to agent center
    avoidance_margin: float = 3.2      # Proactive buffer added to separation for RVO braking
    apf_repulsive_gain: float = 12.0   # Repulsion strength for safety repair
    max_repair_iterations: int = 5    # Max attempts to steer trajectory away
    emergency_brake_decel: float = 3.5 # Deceleration rate on fail-safe trigger


@dataclass
class LatencyBudgetConfig:
    """Real-time latency monitoring and SLA budget."""
    budget_ms: float = 25.0           # Maximum permissible P95 latency per tick
    strict_enforcement: bool = True   # Fail simulation if P95 exceeds budget
    warn_threshold_ms: float = 18.0   # Warning flag for approaching SLA ceiling


@dataclass
class SimulationConfig:
    """Complete master configuration for a simulation run."""
    seed: int = 42
    agent_count: int = 15
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    agent_params: AgentPhysicalConfig = field(default_factory=AgentPhysicalConfig)
    fitness_weights: FitnessWeightsConfig = field(default_factory=FitnessWeightsConfig)
    opt_params: OptimizationParameters = field(default_factory=OptimizationParameters)
    safety_params: SafetyConstraintsConfig = field(default_factory=SafetyConstraintsConfig)
    latency_budget: LatencyBudgetConfig = field(default_factory=LatencyBudgetConfig)

    def __post_init__(self) -> None:
        self.fitness_weights.validate()
