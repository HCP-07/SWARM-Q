"""Baseline 2: Static Priority Navigation.

Agents use fixed static weights without environmental adaptation,
and resolve conflicts strictly through pre-assigned agent IDs.
"""

from __future__ import annotations
from typing import Tuple, List, Set, Dict, Optional, Any
from config import Action, EnvironmentConfig, AgentConfig, AdaptiveWeightsConfig
from core.agent import AutonomousAgent, SwarmMessage


class StaticPriorityAgent(AutonomousAgent):
    """Static priority agent utilizing unadapted weights and fixed ID priority."""

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
            base_weights=base_weights or AdaptiveWeightsConfig(),
            enable_adaptation=False
        )

    def get_priority(self) -> float:
        """Static priority strictly based on inverse agent ID (lower ID has higher priority)."""
        return 1000.0 - float(self.state.agent_id)

    def create_broadcast_message(self) -> SwarmMessage:
        return SwarmMessage(
            agent_id=self.state.agent_id,
            position=self.state.position,
            intended_next_position=self.state.intended_position,
            target=self.state.target,
            priority=self.get_priority(),
            local_risk=0.0,
            active=self.state.active
        )
