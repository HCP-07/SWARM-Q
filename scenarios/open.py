"""Scenario 1: Open environment with unobstructed spatial coordinates."""

from __future__ import annotations
import random
from typing import Tuple, List, Type
from config import SimulationConfig
from core.environment import Environment
from core.swarm import SwarmCoordinator
from core.agent import AutonomousAgent
from algorithms.adaptive_swarm import AdaptiveSwarmAgent


def build_open_scenario(
    config: SimulationConfig,
    agent_cls: Type[AutonomousAgent] = AdaptiveSwarmAgent
) -> SwarmCoordinator:
    """Builds an open 2D arena with distributed agent start points and targets."""
    env = Environment(config.env, seed=config.seed)
    coordinator = SwarmCoordinator(config, env=env)
    rng = random.Random(config.seed)

    w, h = config.env.width, config.env.height
    used_starts = set()
    used_targets = set()

    for aid in range(config.agent_count):
        # Generate non-overlapping starting positions
        while True:
            sx = rng.randint(2, w // 3)
            sy = rng.randint(2, h - 3)
            if (sx, sy) not in used_starts:
                used_starts.add((sx, sy))
                break

        # Generate targets on opposite side of arena
        while True:
            tx = rng.randint(2 * w // 3, w - 3)
            ty = rng.randint(2, h - 3)
            if (tx, ty) not in used_targets:
                used_targets.add((tx, ty))
                break

        agent = agent_cls(
            agent_id=aid,
            initial_position=(sx, sy),
            target=(tx, ty),
            config=config.agent,
            base_weights=config.weights,
            enable_adaptation=config.enable_adaptation
        )
        coordinator.add_agent(agent)

    return coordinator
