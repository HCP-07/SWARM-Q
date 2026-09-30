"""Baseline 1: Greedy Goal Navigation.

Agents greedily choose the action minimizing raw Euclidean distance to target,
with minimal local obstacle collision avoidance and no swarm adaptation.
"""

from __future__ import annotations
import math
from typing import Tuple, List, Set, Dict, Optional, Any
from config import Action, EnvironmentConfig, AgentConfig, AdaptiveWeightsConfig
from core.agent import AutonomousAgent, SwarmMessage
from core.collision import CollisionChecker


class GreedyAgent(AutonomousAgent):
    """Greedy baseline agent moving directly toward its target."""

    def __init__(
        self,
        agent_id: int,
        initial_position: Tuple[int, int],
        target: Tuple[int, int],
        config: Optional[AgentConfig] = None,
        base_weights: Optional[AdaptiveWeightsConfig] = None,
        enable_adaptation: bool = False,
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            initial_position=initial_position,
            target=target,
            config=config,
            base_weights=base_weights,
            enable_adaptation=False
        )

    def plan_candidates(
        self,
        neighbor_messages: List[SwarmMessage],
        static_obstacles: Set[Tuple[int, int]],
        dynamic_obstacles: Set[Tuple[int, int]],
        env_cfg: EnvironmentConfig,
        communication_quality: float = 1.0,
        rng: Optional[Any] = None
    ) -> Tuple[int, int]:
        if not self.state.active or self.state.completed_target:
            self.state.intended_position = self.state.position
            self._ranked_candidates = [(Action.WAIT, 0.0, self.state.position)]
            return self.state.position

        candidates = []
        for act in Action:
            next_pos = (self.state.position[0] + act.dx, self.state.position[1] + act.dy)
            if not CollisionChecker.is_within_bounds(next_pos, env_cfg):
                continue
            if not CollisionChecker.is_obstacle_free(next_pos, static_obstacles, dynamic_obstacles):
                continue

            dist = math.hypot(next_pos[0] - self.state.target[0], next_pos[1] - self.state.target[1])
            score = -dist  # greedy: minimize Euclidean distance
            candidates.append((act, score, next_pos))

        if not any(c[0] == Action.WAIT for c in candidates):
            candidates.append((Action.WAIT, -999.0, self.state.position))

        candidates.sort(key=lambda item: item[1], reverse=True)
        self._ranked_candidates = candidates
        self.state.intended_position = candidates[0][2]
        return self.state.intended_position
