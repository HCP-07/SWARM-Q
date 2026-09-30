"""Scenario 2 & 3: Static Obstacles and Dense Swarm Scenarios."""

from __future__ import annotations
import random
from typing import Tuple, List, Type
from config import SimulationConfig
from core.environment import Environment
from core.swarm import SwarmCoordinator
from core.agent import AutonomousAgent
from algorithms.adaptive_swarm import AdaptiveSwarmAgent


def build_obstacles_scenario(
    config: SimulationConfig,
    agent_cls: Type[AutonomousAgent] = AdaptiveSwarmAgent
) -> SwarmCoordinator:
    """Scenario 2: Arena with geometric static obstacle barriers and corridors."""
    env = Environment(config.env, seed=config.seed)
    w, h = config.env.width, config.env.height

    # Create vertical barrier walls with narrow passage gaps
    wall_x1 = w // 3
    wall_x2 = 2 * w // 3

    for y in range(h):
        # Gap at center (y in 16..24)
        if not (h // 2 - 4 <= y <= h // 2 + 4):
            env.add_static_obstacle((wall_x1, y))

        # Staggered gaps on second barrier
        if not (h // 4 - 2 <= y <= h // 4 + 2 or 3 * h // 4 - 2 <= y <= 3 * h // 4 + 2):
            env.add_static_obstacle((wall_x2, y))

    coordinator = SwarmCoordinator(config, env=env)
    rng = random.Random(config.seed)

    used_starts = set(env.static_obstacles)
    used_targets = set(env.static_obstacles)

    for aid in range(config.agent_count):
        while True:
            sx = rng.randint(1, wall_x1 - 2)
            sy = rng.randint(2, h - 3)
            if (sx, sy) not in used_starts:
                used_starts.add((sx, sy))
                break

        while True:
            tx = rng.randint(wall_x2 + 2, w - 2)
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


def build_dense_scenario(
    config: SimulationConfig,
    agent_cls: Type[AutonomousAgent] = AdaptiveSwarmAgent
) -> SwarmCoordinator:
    """Scenario 3: Dense swarm packed in a constricted arena to induce high coordination pressure."""
    dense_cfg = SimulationConfig(
        seed=config.seed,
        agent_count=max(30, config.agent_count),
        max_ticks=config.max_ticks,
        scenario=config.scenario,
        env=config.env,
        agent=config.agent,
        weights=config.weights,
        enable_adaptation=config.enable_adaptation
    )

    env = Environment(dense_cfg.env, seed=dense_cfg.seed)
    coordinator = SwarmCoordinator(dense_cfg, env=env)
    rng = random.Random(dense_cfg.seed)

    w, h = dense_cfg.env.width, dense_cfg.env.height
    used_starts = set()
    used_targets = set()

    for aid in range(dense_cfg.agent_count):
        # Spatially interleave starts across top and bottom
        while True:
            if aid % 2 == 0:
                sx = rng.randint(2, w // 2 - 2)
                sy = rng.randint(2, h // 2 - 2)
                tx = rng.randint(w // 2 + 2, w - 3)
                ty = rng.randint(h // 2 + 2, h - 3)
            else:
                sx = rng.randint(w // 2 + 2, w - 3)
                sy = rng.randint(2, h // 2 - 2)
                tx = rng.randint(2, w // 2 - 2)
                ty = rng.randint(h // 2 + 2, h - 3)

            if (sx, sy) not in used_starts and (tx, ty) not in used_targets:
                used_starts.add((sx, sy))
                used_targets.add((tx, ty))
                break

        agent = agent_cls(
            agent_id=aid,
            initial_position=(sx, sy),
            target=(tx, ty),
            config=dense_cfg.agent,
            base_weights=dense_cfg.weights,
            enable_adaptation=dense_cfg.enable_adaptation
        )
        coordinator.add_agent(agent)

    return coordinator
