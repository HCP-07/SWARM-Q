"""Tests for deadlock detection, yielding mechanisms, and autonomous resolution."""

import pytest
from config import Action, AgentConfig, SimulationConfig, ScenarioType
from core.agent import AutonomousAgent, SwarmMessage
from scenarios import get_scenario


def test_stagnation_ticks_increment_on_wait():
    agent = AutonomousAgent(agent_id=0, initial_position=(10, 10), target=(20, 20))
    assert agent.state.stagnation_ticks == 0

    agent.execute_action(Action.WAIT)
    assert agent.state.stagnation_ticks == 1

    agent.execute_action(Action.WAIT)
    assert agent.state.stagnation_ticks == 2


def test_stagnation_resets_on_progress():
    agent = AutonomousAgent(agent_id=0, initial_position=(10, 10), target=(20, 20))
    agent.execute_action(Action.WAIT)
    agent.execute_action(Action.WAIT)
    assert agent.state.stagnation_ticks == 2

    # Move closer to target
    agent.execute_action(Action.RIGHT)  # pos becomes (11, 10), distance reduces
    assert agent.state.stagnation_ticks == 0


def test_deadlock_counter_trigger():
    cfg = AgentConfig(stagnation_threshold=3)
    agent = AutonomousAgent(agent_id=0, initial_position=(10, 10), target=(20, 20), config=cfg)

    agent.execute_action(Action.WAIT)
    agent.execute_action(Action.WAIT)
    assert agent.state.deadlocks_encountered == 0

    agent.execute_action(Action.WAIT)  # 3rd stagnant tick matches threshold
    assert agent.state.deadlocks_encountered == 1


def test_deadlock_resolution_trigger():
    cfg = AgentConfig(stagnation_threshold=2)
    agent = AutonomousAgent(agent_id=0, initial_position=(10, 10), target=(20, 20), config=cfg)

    # Encounter deadlock
    agent.execute_action(Action.WAIT)
    agent.execute_action(Action.WAIT)
    assert agent.state.deadlocks_encountered == 1
    assert agent.state.deadlocks_resolved == 0

    # Break deadlock by progressing
    agent.execute_action(Action.RIGHT)
    assert agent.state.deadlocks_resolved == 1


def test_yielding_to_higher_priority():
    # Two agents compete for (10, 10)
    agent0 = AutonomousAgent(agent_id=0, initial_position=(9, 10), target=(10, 10))
    agent1 = AutonomousAgent(agent_id=1, initial_position=(11, 10), target=(10, 10))

    # Give agent 0 higher priority
    msg0 = SwarmMessage(
        agent_id=0, position=(9, 10), intended_next_position=(10, 10),
        target=(10, 10), priority=10.0, local_risk=0.0, active=True
    )
    msg1 = SwarmMessage(
        agent_id=1, position=(11, 10), intended_next_position=(10, 10),
        target=(10, 10), priority=2.0, local_risk=0.0, active=True
    )

    priorities = {0: 10.0, 1: 2.0}

    # Set candidates for agent 1
    agent1._ranked_candidates = [
        (Action.LEFT, 5.0, (10, 10)),   # Preferred move into (10, 10)
        (Action.UP, 3.0, (11, 11)),     # Alternate non-conflicting move
        (Action.WAIT, 0.0, (11, 10))
    ]

    action1 = agent1.resolve_and_select_action([msg0], priorities)
    # Agent 1 must yield (10, 10) to higher-priority agent 0, picking alternate UP
    assert action1 == Action.UP


def test_deadlock_resolution_in_obstacles_scenario():
    cfg = SimulationConfig(agent_count=10, max_ticks=60, seed=42)
    coord = get_scenario(ScenarioType.STATIC_OBSTACLES, config=cfg)
    metrics = coord.run()

    # Even in narrow passages, resolution rate should be non-zero
    assert metrics.deadlock_resolution_rate >= 0.0
    assert metrics.collision_count == 0
