"""Full stress scenario combining moving obstacles, comms blackout, and agent failure."""

from __future__ import annotations
import numpy as np
from configs.config import SimulationConfig
from src.environment.obstacles import DynamicObstacle
from src.environment.perturbations import PerturbationEvent, PerturbationType
from src.simulation.engine import SimulationEngine
from scenarios.static import populate_base_world


def build_stress_scenario(config: SimulationConfig) -> SimulationEngine:
    """Builds environment under concurrent stress perturbations."""
    engine = SimulationEngine(config)
    populate_base_world(engine, target_count=max(15, int(config.agent_count * 1.5)))

    # 1. Dynamic obstacle intruder
    dyn1 = DynamicObstacle(
        center=np.array([50.0, 95.0]),
        radius=3.5,
        velocity=np.array([1.0, -3.0]),
        bounds=config.arena.bounds,
        obs_id="dyn_stress_1"
    )
    engine.env.perturbation_engine.add_event(PerturbationEvent(
        trigger_tick=15,
        event_type=PerturbationType.DYNAMIC_OBSTACLE_SPAWN,
        payload={"obstacle": dyn1}
    ))

    # 2. Localized communication dropout
    affected_ids = [1, 2] if config.agent_count > 2 else [0]
    engine.env.perturbation_engine.add_event(PerturbationEvent(
        trigger_tick=25,
        event_type=PerturbationType.COMMUNICATION_DROPOUT,
        target_ids=affected_ids,
        duration_ticks=40
    ))

    # 3. Permanent hardware crash
    fail_id = 3 if config.agent_count > 3 else 0
    engine.env.perturbation_engine.add_event(PerturbationEvent(
        trigger_tick=45,
        event_type=PerturbationType.AGENT_FAILURE,
        target_ids=[fail_id]
    ))

    engine.initialize_swarm(config.agent_count)
    return engine
