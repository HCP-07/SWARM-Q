"""Dynamic obstacle scenario with moving intruder crossing corridors."""

from __future__ import annotations
import numpy as np
from configs.config import SimulationConfig
from src.environment.obstacles import DynamicObstacle
from src.environment.perturbations import PerturbationEvent, PerturbationType
from src.simulation.engine import SimulationEngine
from scenarios.static import populate_base_world


def build_dynamic_obstacle_scenario(config: SimulationConfig) -> SimulationEngine:
    """Builds environment with dynamic obstacle crossing central corridor."""
    engine = SimulationEngine(config)
    populate_base_world(engine, target_count=max(15, int(config.agent_count * 1.5)))

    dyn_obs = DynamicObstacle(
        center=np.array([50.0, 95.0]),
        radius=3.5,
        velocity=np.array([0.0, -3.0]),
        bounds=config.arena.bounds,
        obs_id="dyn_cutter"
    )
    engine.env.perturbation_engine.add_event(PerturbationEvent(
        trigger_tick=20,
        event_type=PerturbationType.DYNAMIC_OBSTACLE_SPAWN,
        payload={"obstacle": dyn_obs}
    ))

    engine.initialize_swarm(config.agent_count)
    return engine
