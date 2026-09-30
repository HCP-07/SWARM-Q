"""Scenario 6: Hardware agent failure and autonomous swarm recovery."""

from __future__ import annotations
import random
from typing import Tuple, List, Type
from config import SimulationConfig
from core.environment import Environment, PerturbationEvent, PerturbationType
from core.swarm import SwarmCoordinator
from core.agent import AutonomousAgent
from algorithms.adaptive_swarm import AdaptiveSwarmAgent


def build_failure_scenario(
    config: SimulationConfig,
    agent_cls: Type[AutonomousAgent] = AdaptiveSwarmAgent
) -> SwarmCoordinator:
    """Scenario 6: Abrupt hardware crash of an agent testing decentralized continuity."""
    env = Environment(config.env, seed=config.seed)
    w, h = config.env.width, config.env.height

    # Trigger failure of agent 1 at tick 25
    fail_agent_id = 1 if config.agent_count > 1 else 0
    env.add_perturbation(PerturbationEvent(
        trigger_tick=25,
        event_type=PerturbationType.AGENT_FAILURE,
        payload={"agent_id": fail_agent_id}
    ))

    coordinator = SwarmCoordinator(config, env=env)
    coordinator.metrics_collector.failure_ticks.append(25)
    rng = random.Random(config.seed)

    used_starts = set()
    used_targets = set()

    for aid in range(config.agent_count):
        while True:
            sx = rng.randint(2, w // 3)
            sy = rng.randint(2, h - 3)
            tx = rng.randint(2 * w // 3, w - 3)
            ty = rng.randint(2, h - 3)
            if (sx, sy) not in used_starts and (tx, ty) not in used_targets:
                used_starts.add((sx, sy))
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
