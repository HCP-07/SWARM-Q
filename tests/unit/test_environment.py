"""Unit tests for environment representation, dynamic obstacles, and deterministic resets."""

import numpy as np
import pytest
from configs.config import ArenaConfig
from src.environment.world import Environment
from src.environment.obstacles import CircularObstacle, RectangularObstacle, DynamicObstacle, HazardZone
from src.environment.targets import TaskTarget


def test_environment_initialization_and_entities():
    arena = ArenaConfig(x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0)
    env = Environment(arena, seed=42)

    circ = CircularObstacle(np.array([20.0, 20.0]), radius=5.0)
    rect = RectangularObstacle(40.0, 40.0, 60.0, 60.0)
    dyn = DynamicObstacle(np.array([50.0, 50.0]), radius=3.0, velocity=np.array([1.0, 0.0]), bounds=arena.bounds)
    hazard = HazardZone(np.array([80.0, 80.0]), radius=10.0)
    target = TaskTarget(task_id=1, position=np.array([90.0, 90.0]))

    env.add_obstacle(circ)
    env.add_obstacle(rect)
    env.add_obstacle(dyn)
    env.add_hazard_zone(hazard)
    env.add_target(target)

    assert len(env.obstacles) == 3
    assert len(env.dynamic_obstacles) == 1
    assert len(env.hazard_zones) == 1
    assert len(env.targets) == 1


def test_dynamic_obstacle_bounces_at_arena_boundary():
    bounds = (0.0, 100.0, 0.0, 100.0)
    # Heading right, near boundary
    dyn = DynamicObstacle(
        center=np.array([98.0, 50.0]),
        radius=2.0,
        velocity=np.array([3.0, 0.0]),
        bounds=bounds
    )
    # Move forward
    dyn.step(dt=1.0)
    # Center reached 100 - radius = 98.0 and velocity flipped to negative
    assert dyn.velocity[0] < 0.0
    assert dyn.center[0] <= 98.0


def test_environment_reset_determinism():
    arena = ArenaConfig()
    env = Environment(arena, seed=123)
    dyn = DynamicObstacle(
        center=np.array([50.0, 50.0]),
        radius=3.0,
        velocity=np.array([2.0, 2.0]),
        bounds=arena.bounds
    )
    env.add_obstacle(dyn)

    for _ in range(10):
        env.step({})
    pos_after_10 = dyn.center.copy()

    # Recreate with same seed
    env2 = Environment(arena, seed=123)
    dyn2 = DynamicObstacle(
        center=np.array([50.0, 50.0]),
        radius=3.0,
        velocity=np.array([2.0, 2.0]),
        bounds=arena.bounds
    )
    env2.add_obstacle(dyn2)
    for _ in range(10):
        env2.step({})

    assert np.allclose(pos_after_10, dyn2.center)
