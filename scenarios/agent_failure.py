"""Scenario modeling abrupt hardware failure and orphan task reallocation."""

from __future__ import annotations
from configs.config import SimulationConfig
from src.environment.perturbations import PerturbationEvent, PerturbationType
from src.simulation.engine import SimulationEngine
from scenarios.static import populate_base_world


def build_agent_failure_scenario(config: SimulationConfig) -> SimulationEngine:
    """Builds environment with abrupt hardware crash of an agent."""
    engine = SimulationEngine(config)
    populate_base_world(engine, target_count=max(15, int(config.agent_count * 1.5)))

    fail_id = 2 if config.agent_count > 2 else 0
    engine.env.perturbation_engine.add_event(PerturbationEvent(
        trigger_tick=35,
        event_type=PerturbationType.AGENT_FAILURE,
        target_ids=[fail_id]
    ))

    engine.initialize_swarm(config.agent_count)
    return engine
