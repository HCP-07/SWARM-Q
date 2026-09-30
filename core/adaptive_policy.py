"""Explicit adaptive heuristic policy governing local decision-making weights."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from config import AdaptiveWeightsConfig


@dataclass
class LocalPerceptionState:
    """Local environmental and swarm conditions perceived by a single agent."""
    nearby_agent_count: int = 0
    min_neighbor_dist: float = float("inf")
    min_obstacle_dist: float = float("inf")
    stagnation_ticks: int = 0
    recent_conflicts: int = 0
    communication_quality: float = 1.0  # 1.0 = full mesh, 0.0 = complete dropout
    target_distance: float = 0.0
    progress_last_5_ticks: float = 0.0


class AdaptivePolicy:
    """Dynamically adjusts decision weights based on local swarm conditions."""

    def __init__(self, base_weights: Optional[AdaptiveWeightsConfig] = None):
        self.base_weights = base_weights or AdaptiveWeightsConfig()

    def adapt_weights(self, state: LocalPerceptionState) -> AdaptiveWeightsConfig:
        """Computes adapted weights using transparent, deterministic heuristic rules."""
        w = AdaptiveWeightsConfig(
            w_goal=self.base_weights.w_goal,
            w_collision=self.base_weights.w_collision,
            w_obstacle=self.base_weights.w_obstacle,
            w_conflict=self.base_weights.w_conflict,
            w_energy=self.base_weights.w_energy,
            w_momentum=self.base_weights.w_momentum,
            w_exploration=self.base_weights.w_exploration
        )

        # Rule 1: Collision risk escalation
        # If neighbors or obstacles are encroaching within 2 grid units
        if state.min_neighbor_dist <= 2.0 or state.min_obstacle_dist <= 1.5:
            threat_scale = max(1.0, 3.0 - min(state.min_neighbor_dist, state.min_obstacle_dist))
            w.w_collision = min(self.base_weights.max_collision_weight, w.w_collision * threat_scale)
            w.w_conflict += 1.5 * state.recent_conflicts
            # Dampen aggressive goal-seeking when in close quarters to prevent head-on crashes
            w.w_goal = max(self.base_weights.min_goal_weight, w.w_goal * 0.6)

        # Rule 2: Stagnation / Deadlock breaking
        # If the agent has made zero or negative progress for multiple ticks
        if state.stagnation_ticks >= 3:
            stagnation_factor = min(3.0, 1.0 + 0.5 * (state.stagnation_ticks - 2))
            w.w_exploration = min(self.base_weights.max_exploration_weight, w.w_exploration * stagnation_factor)
            # Temporarily relax goal attraction to encourage wide lateral detours
            w.w_goal = max(self.base_weights.min_goal_weight, w.w_goal * (0.8 ** state.stagnation_ticks))
            w.w_momentum = 0.2  # Allow sharp direction changes away from deadlock

        # Rule 3: Clear path acceleration
        # When no agents or obstacles are nearby and path is unobstructed
        if state.min_neighbor_dist > 4.0 and state.min_obstacle_dist > 3.0 and state.stagnation_ticks == 0:
            w.w_goal = min(self.base_weights.max_goal_weight, w.w_goal * 1.3)
            w.w_exploration = 0.2
            w.w_momentum = 1.2  # Cruise efficiently in consistent direction

        # Rule 4: Communication dropout resilience
        # If neighbor telemetry is degraded or dropped, expand conservative safety margins
        if state.communication_quality < 0.8:
            comm_penalty = 1.0 - state.communication_quality
            w.w_collision *= (1.0 + 0.8 * comm_penalty)
            w.w_obstacle *= (1.0 + 0.5 * comm_penalty)
            # Reduce dependence on neighbor intention assumptions
            w.w_conflict *= state.communication_quality

        return w
