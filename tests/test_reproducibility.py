"""Tests verifying simulation determinism and reproducibility under identical seeds."""

import pytest
from config import SimulationConfig, ScenarioType
from scenarios import get_scenario


def test_identical_seed_exact_determinism():
    cfg1 = SimulationConfig(agent_count=8, max_ticks=25, seed=123)
    coord1 = get_scenario(ScenarioType.STATIC_OBSTACLES, config=cfg1)
    m1 = coord1.run()

    cfg2 = SimulationConfig(agent_count=8, max_ticks=25, seed=123)
    coord2 = get_scenario(ScenarioType.STATIC_OBSTACLES, config=cfg2)
    m2 = coord2.run()

    # Exact equality of primary metrics
    assert m1.task_completion_rate == m2.task_completion_rate
    assert m1.collision_count == m2.collision_count
    assert m1.average_path_length == m2.average_path_length
    assert m1.total_energy_cost == m2.total_energy_cost
    assert m1.deadlock_count == m2.deadlock_count

    # Exact equality of agent trajectory histories
    for aid in coord1.agents:
        hist1 = coord1.agents[aid].state.history
        hist2 = coord2.agents[aid].state.history
        assert hist1 == hist2


def test_different_seeds_produce_varied_trajectories():
    cfg1 = SimulationConfig(agent_count=8, max_ticks=25, seed=100)
    coord1 = get_scenario(ScenarioType.OPEN, config=cfg1)
    coord1.run()

    cfg2 = SimulationConfig(agent_count=8, max_ticks=25, seed=200)
    coord2 = get_scenario(ScenarioType.OPEN, config=cfg2)
    coord2.run()

    # Different seeds should start at different positions
    starts1 = [ag.state.history[0] for ag in coord1.agents.values()]
    starts2 = [ag.state.history[0] for ag in coord2.agents.values()]
    assert starts1 != starts2


def test_dynamic_obstacle_determinism():
    cfg1 = SimulationConfig(agent_count=4, max_ticks=20, seed=42)
    coord1 = get_scenario(ScenarioType.DYNAMIC_OBSTACLES, config=cfg1)
    coord1.run()

    cfg2 = SimulationConfig(agent_count=4, max_ticks=20, seed=42)
    coord2 = get_scenario(ScenarioType.DYNAMIC_OBSTACLES, config=cfg2)
    coord2.run()

    for oid in coord1.env.dynamic_obstacles:
        p1 = coord1.env.dynamic_obstacles[oid].position
        p2 = coord2.env.dynamic_obstacles[oid].position
        assert p1 == p2
