"""Tests for hardware agent failure handling and autonomous swarm resilience."""

import pytest
from config import SimulationConfig, ScenarioType
from core.environment import Environment, PerturbationEvent, PerturbationType
from core.agent import AutonomousAgent
from core.swarm import SwarmCoordinator
from scenarios import get_scenario


def test_agent_failure_perturbation_triggers():
    env = Environment()
    agent0 = AutonomousAgent(agent_id=0, initial_position=(5, 5), target=(10, 10))
    agent1 = AutonomousAgent(agent_id=1, initial_position=(6, 6), target=(10, 10))
    agents_map = {0: agent0, 1: agent1}

    # Inject failure for agent 0 at tick 5
    env.add_perturbation(PerturbationEvent(
        trigger_tick=5,
        event_type=PerturbationType.AGENT_FAILURE,
        payload={"agent_id": 0}
    ))

    # Ticks 0..4: both active
    for t in range(5):
        env.step_environment(t, agents_map)
        assert agent0.state.active is True
        assert agent1.state.active is True

    # Tick 5: agent 0 becomes inactive
    affected = env.step_environment(5, agents_map)
    assert 0 in affected
    assert agent0.state.active is False
    assert agent1.state.active is True


def test_failed_agent_stops_movement():
    agent = AutonomousAgent(agent_id=0, initial_position=(5, 5), target=(10, 10))
    agent.state.active = False

    # Attempt to plan and execute
    action = agent.perceive_and_plan([], set(), set(), Environment().config)
    from config import Action
    assert action == Action.WAIT

    agent.execute_action(Action.RIGHT)
    # Position must remain unchanged
    assert agent.state.position == (5, 5)


def test_failure_scenario_continuity():
    # Run full failure scenario where agent crashes mid-mission
    cfg = SimulationConfig(agent_count=6, max_ticks=40, seed=42)
    coord = get_scenario(ScenarioType.AGENT_FAILURE, config=cfg)
    metrics = coord.run()

    # The simulation must complete cleanly with 0 collisions
    assert metrics.collision_count == 0
    # Surviving agents must successfully reach their targets
    assert metrics.task_completion_rate > 0.0
    # Failure recovery metrics populated
    assert metrics.recovery_time_after_failure >= 0.0


def test_combined_scenario_resilience():
    # Combined scenario: moving obstacles + comms blackout + agent failure
    cfg = SimulationConfig(agent_count=6, max_ticks=45, seed=42)
    coord = get_scenario(ScenarioType.COMBINED, config=cfg)
    metrics = coord.run()

    assert metrics.collision_count == 0
    assert metrics.real_time_pass is True
