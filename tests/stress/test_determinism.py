"""Determinism verification test ensuring identical results with identical seeds."""

import pytest
import numpy as np
from configs.config import SimulationConfig
from scenarios.scenario_definitions import ScenarioBuilder, ScenarioType


def test_simulation_determinism():
    """Validates that two distinct simulation instances with the same seed yield bitwise identical outputs."""
    seed = 1007
    cfg1 = SimulationConfig(seed=seed, agent_count=8)
    cfg1.arena.max_ticks = 30
    engine1 = ScenarioBuilder.build(ScenarioType.DYNAMIC_OBSTACLE, cfg1)
    metrics1 = engine1.run(cfg1.arena.max_ticks)

    cfg2 = SimulationConfig(seed=seed, agent_count=8)
    cfg2.arena.max_ticks = 30
    engine2 = ScenarioBuilder.build(ScenarioType.DYNAMIC_OBSTACLE, cfg2)
    metrics2 = engine2.run(cfg2.arena.max_ticks)

    # 1. Compare aggregate metrics
    assert np.isclose(metrics1.task_completion_rate, metrics2.task_completion_rate)
    assert np.isclose(metrics1.total_path_length, metrics2.total_path_length)
    assert np.isclose(metrics1.total_energy_consumed, metrics2.total_energy_consumed)
    assert metrics1.collision_count == metrics2.collision_count

    # 2. Compare exact agent trajectory coordinates
    for aid in engine1.agents.keys():
        path1 = engine1.agents[aid].state.history_path
        path2 = engine2.agents[aid].state.history_path
        assert len(path1) == len(path2)
        for p1, p2 in zip(path1, path2):
            assert np.allclose(p1, p2, atol=1e-5)
