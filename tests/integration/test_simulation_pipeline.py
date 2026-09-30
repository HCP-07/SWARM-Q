"""Integration tests for end-to-end simulation pipelines under dynamic perturbations."""

import pytest
import numpy as np
from configs.config import SimulationConfig
from scenarios.scenario_definitions import ScenarioBuilder, ScenarioType


def test_static_scenario_pipeline():
    config = SimulationConfig(seed=42, agent_count=5)
    config.arena.max_ticks = 30
    engine = ScenarioBuilder.build(ScenarioType.STATIC, config)

    metrics = engine.run(config.arena.max_ticks)
    assert metrics.collision_count == 0
    assert metrics.total_path_length > 0.0
    assert metrics.average_tick_latency_ms > 0.0
    assert len(engine.latency_tracker.tick_latencies_ms) == 30


def test_dynamic_obstacle_perturbation_pipeline():
    config = SimulationConfig(seed=42, agent_count=5)
    config.arena.max_ticks = 40
    engine = ScenarioBuilder.build(ScenarioType.DYNAMIC_OBSTACLE, config)

    # Dynamic obstacle scheduled at tick 20
    metrics = engine.run(config.arena.max_ticks)
    assert metrics.collision_count == 0
    assert any(p["type"] == "dynamic_obstacle_spawn" for p in engine.env.perturbation_engine.history)


def test_communication_dropout_pipeline():
    config = SimulationConfig(seed=42, agent_count=6)
    config.arena.max_ticks = 40
    engine = ScenarioBuilder.build(ScenarioType.COMMUNICATION_DROPOUT, config)

    metrics = engine.run(config.arena.max_ticks)
    assert metrics.collision_count == 0
    # Perturbation engine recorded comm dropout
    assert any(p["type"] == "communication_dropout" for p in engine.env.perturbation_engine.history)


def test_agent_failure_pipeline():
    config = SimulationConfig(seed=42, agent_count=6)
    config.arena.max_ticks = 50
    engine = ScenarioBuilder.build(ScenarioType.AGENT_FAILURE, config)

    metrics = engine.run(config.arena.max_ticks)
    # The failed agent must be inactive
    failed_agents = [ag for ag in engine.agents.values() if not ag.state.active]
    assert len(failed_agents) >= 1
    # Surviving agents maintain zero collisions
    assert metrics.collision_count == 0
