"""Scenario 7: Combined concurrent dynamic perturbations."""

from __future__ import annotations
import random
from typing import Tuple, List, Type
from config import SimulationConfig
from core.environment import Environment, PerturbationEvent, PerturbationType
from core.swarm import SwarmCoordinator
from core.agent import AutonomousAgent
from algorithms.adaptive_swarm import AdaptiveSwarmAgent


def build_combined_scenario(
    config: SimulationConfig,
    agent_cls: Type[AutonomousAgent] = AdaptiveSwarmAgent
) -> SwarmCoordinator:
    """Scenario 7: Extreme scenario combining moving obstacles, comms blackout, and agent failure."""
    env = Environment(config.env, seed=config.seed)
    w, h = config.env.width, config.env.height

    # 1. Static obstacles (central barrier column with gaps)
    for y in range(h):
        if not (h // 2 - 3 <= y <= h // 2 + 3):
            env.add_static_obstacle((w // 2, y))

    # 2. Moving dynamic obstacle patrolling passage
    env.add_dynamic_obstacle("dyn_patrol", (w // 2, h // 2), (0, 1))

    # 3. Communication dropout at tick 15
    dropout_targets = [0, 1] if config.agent_count > 1 else [0]
    env.add_perturbation(PerturbationEvent(
        trigger_tick=15,
        event_type=PerturbationType.COMMUNICATION_DROPOUT,
        payload={"agent_ids": dropout_targets, "duration": 25}
    ))

    # 4. Agent failure at tick 30
    fail_id = 2 if config.agent_count > 2 else 0
    env.add_perturbation(PerturbationEvent(
        trigger_tick=30,
        event_type=PerturbationType.AGENT_FAILURE,
        payload={"agent_id": fail_id}
    ))

    # 5. Obstacle appearance at tick 45
    env.add_perturbation(PerturbationEvent(
        trigger_tick=45,
        event_type=PerturbationType.OBSTACLE_APPEAR,
        payload={"positions": [(w // 2, y) for y in range(h // 2 - 2, h // 2 + 2)]}
    ))

    coordinator = SwarmCoordinator(config, env=env)
    coordinator.metrics_collector.failure_ticks.append(30)
    rng = random.Random(config.seed)

    used_starts = set(env.static_obstacles)
    used_targets = set(env.static_obstacles)

    for aid in range(config.agent_count):
        while True:
            sx = rng.randint(2, w // 2 - 2)
            sy = rng.randint(2, h - 3)
            tx = rng.randint(w // 2 + 2, w - 3)
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
