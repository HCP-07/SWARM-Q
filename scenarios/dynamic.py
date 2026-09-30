"""Scenario 4 & 5: Dynamic Obstacles and Communication Dropout Scenarios."""

from __future__ import annotations
import random
from typing import Tuple, List, Type
from config import SimulationConfig
from core.environment import Environment, PerturbationEvent, PerturbationType
from core.swarm import SwarmCoordinator
from core.agent import AutonomousAgent
from algorithms.adaptive_swarm import AdaptiveSwarmAgent


def build_dynamic_scenario(
    config: SimulationConfig,
    agent_cls: Type[AutonomousAgent] = AdaptiveSwarmAgent
) -> SwarmCoordinator:
    """Scenario 4: Moving dynamic obstacles that intercept agent navigation trajectories."""
    env = Environment(config.env, seed=config.seed)
    w, h = config.env.width, config.env.height

    # Add dynamic patrol obstacles crossing vertical corridors
    env.add_dynamic_obstacle("dyn_1", (w // 2, 5), (0, 1))
    env.add_dynamic_obstacle("dyn_2", (w // 2, h - 5), (0, -1))
    env.add_dynamic_obstacle("dyn_3", (w // 3, h // 2), (1, 0))

    # Add mid-mission dynamic obstacle appearance event at tick 20
    env.add_perturbation(PerturbationEvent(
        trigger_tick=20,
        event_type=PerturbationType.OBSTACLE_APPEAR,
        payload={"positions": [(w // 2, y) for y in range(h // 2 - 4, h // 2 + 5)]}
    ))

    coordinator = SwarmCoordinator(config, env=env)
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


def build_dropout_scenario(
    config: SimulationConfig,
    agent_cls: Type[AutonomousAgent] = AdaptiveSwarmAgent
) -> SwarmCoordinator:
    """Scenario 5: Wireless communication blackout injecting packet loss on selected agents."""
    env = Environment(config.env, seed=config.seed)
    w, h = config.env.width, config.env.height

    # Target agent IDs 0, 1, 2 for communication blackout between tick 15 and 45
    dropout_targets = list(range(min(4, config.agent_count)))
    env.add_perturbation(PerturbationEvent(
        trigger_tick=15,
        event_type=PerturbationType.COMMUNICATION_DROPOUT,
        payload={"agent_ids": dropout_targets, "duration": 30}
    ))

    coordinator = SwarmCoordinator(config, env=env)
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
