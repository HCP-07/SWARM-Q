"""Baseline 3: Non-Adaptive Swarm Navigation.

Agents execute the decentralized heuristic (goal attraction, neighbor repulsion,
obstacle repulsion, momentum, exploration) using fixed, unadapted weights.
"""

from __future__ import annotations
from typing import Tuple, Optional, Any
from config import AgentConfig, AdaptiveWeightsConfig
from core.agent import AutonomousAgent


class NonAdaptiveSwarmAgent(AutonomousAgent):
    """Swarm agent utilizing fixed weights without environmental adaptation."""

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
