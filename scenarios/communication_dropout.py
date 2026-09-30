"""Scenario modeling temporary RF communication dropouts."""

from __future__ import annotations
from configs.config import SimulationConfig
from src.environment.perturbations import PerturbationEvent, PerturbationType
from src.simulation.engine import SimulationEngine
from scenarios.static import populate_base_world


def build_communication_dropout_scenario(config: SimulationConfig) -> SimulationEngine:
    """Builds environment with localized temporary RF blackout."""
    engine = SimulationEngine(config)
    populate_base_world(engine, target_count=max(15, int(config.agent_count * 1.5)))

    affected_ids = [1, 2, 3] if config.agent_count > 3 else [0]
    engine.env.perturbation_engine.add_event(PerturbationEvent(
        trigger_tick=25,
        event_type=PerturbationType.COMMUNICATION_DROPOUT,
        target_ids=affected_ids,
        duration_ticks=40
    ))

    engine.initialize_swarm(config.agent_count)
    return engine
