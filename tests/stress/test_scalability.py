"""Stress tests evaluating swarm scalability across 10, 25, 50, and 100 agents."""

import pytest
import numpy as np
from configs.config import SimulationConfig
from scenarios.scenario_definitions import ScenarioBuilder, ScenarioType


@pytest.mark.parametrize("swarm_size", [10, 25, 50, 100])
def test_swarm_scalability(swarm_size: int):
    """Verifies that the swarm scales stably up to 100 agents without crashing or deadlock."""
    config = SimulationConfig(seed=42, agent_count=swarm_size)
    config.arena.max_ticks = 20  # Fast execution for stress evaluation

    engine = ScenarioBuilder.build(ScenarioType.STATIC, config)
    metrics = engine.run(config.arena.max_ticks)

    assert len(engine.agents) == swarm_size
    assert metrics.collision_count == 0
    assert metrics.total_path_length > 0.0
    # Average tick latency must remain bounded on single-threaded CPU without runaway overhead
    max_permissible_latency = max(150.0, swarm_size * 35.0)
    assert metrics.average_tick_latency_ms < max_permissible_latency
