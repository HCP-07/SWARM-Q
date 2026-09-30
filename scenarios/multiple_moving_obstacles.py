"""Scenario with multiple moving dynamic obstacles."""

from __future__ import annotations
import numpy as np
from configs.config import SimulationConfig
from src.environment.obstacles import DynamicObstacle
from src.environment.perturbations import PerturbationEvent, PerturbationType
from src.simulation.engine import SimulationEngine
from scenarios.static import populate_base_world


def build_multiple_moving_obstacles_scenario(config: SimulationConfig) -> SimulationEngine:
    """Builds environment with 3 distinct dynamic moving obstacles."""
    engine = SimulationEngine(config)
    populate_base_world(engine, target_count=max(15, int(config.agent_count * 1.5)))

    dyn1 = DynamicObstacle(
        center=np.array([45.0, 95.0]),
        radius=3.0,
        velocity=np.array([1.5, -2.5]),
        bounds=config.arena.bounds,
        obs_id="dyn_1"
    )
    dyn2 = DynamicObstacle(
        center=np.array([70.0, 10.0]),
        radius=3.0,
        velocity=np.array([-2.0, 2.0]),
        bounds=config.arena.bounds,
        obs_id="dyn_2"
    )
    dyn3 = DynamicObstacle(
        center=np.array([20.0, 45.0]),
        radius=2.5,
        velocity=np.array([2.5, 0.5]),
        bounds=config.arena.bounds,
        obs_id="dyn_3"
    )

    engine.env.perturbation_engine.add_event(PerturbationEvent(
        trigger_tick=15,
        event_type=PerturbationType.DYNAMIC_OBSTACLE_SPAWN,
        payload={"obstacle": dyn1}
    ))
    engine.env.perturbation_engine.add_event(PerturbationEvent(
        trigger_tick=25,
        event_type=PerturbationType.DYNAMIC_OBSTACLE_SPAWN,
        payload={"obstacle": dyn2}
    ))
    engine.env.perturbation_engine.add_event(PerturbationEvent(
        trigger_tick=35,
        event_type=PerturbationType.DYNAMIC_OBSTACLE_SPAWN,
        payload={"obstacle": dyn3}
    ))

    engine.initialize_swarm(config.agent_count)
    return engine
