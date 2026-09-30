"""Tests for metric computation, step costs, energy accounting, and value bounds."""

import pytest
from config import Action, AgentConfig, SimulationConfig
from core.agent import AutonomousAgent
from core.metrics import MetricsCollector
from scenarios import get_scenario


def test_action_step_costs():
    assert Action.WAIT.step_cost == 0.1
    assert Action.UP.step_cost == 1.0
    assert Action.DOWN.step_cost == 1.0
    assert Action.LEFT.step_cost == 1.0
    assert Action.RIGHT.step_cost == 1.0
    assert abs(Action.UP_LEFT.step_cost - 1.4142) < 0.001
    assert abs(Action.UP_RIGHT.step_cost - 1.4142) < 0.001
    assert abs(Action.DOWN_LEFT.step_cost - 1.4142) < 0.001
    assert abs(Action.DOWN_RIGHT.step_cost - 1.4142) < 0.001


def test_energy_expenditure_accounting():
    cfg = AgentConfig(initial_energy=100.0, energy_move_cost=1.0, energy_wait_cost=0.1)
    agent = AutonomousAgent(agent_id=0, initial_position=(5, 5), target=(10, 10), config=cfg)

    # 1 cardinal move
    agent.execute_action(Action.RIGHT)
    assert abs(agent.state.energy - 99.0) < 0.01
    assert abs(agent.state.total_energy_expended - 1.0) < 0.01

    # 1 diagonal move
    agent.execute_action(Action.UP_RIGHT)
    assert abs(agent.state.energy - (99.0 - 1.4142)) < 0.01

    # 1 wait
    agent.execute_action(Action.WAIT)
    assert abs(agent.state.energy - (99.0 - 1.4142 - 0.1)) < 0.01


def test_metrics_collector_empty_swarm():
    collector = MetricsCollector()
    metrics = collector.finalize({}, total_ticks=10, hard_collisions=0, dropout_agent_ids=set())
    assert metrics.task_completion_rate == 0.0
    assert metrics.collision_count == 0
    assert metrics.real_time_pass is True


def test_metric_bounds_validation():
    coord = get_scenario("open", SimulationConfig(agent_count=6, max_ticks=20, seed=42))
    m = coord.run()

    assert 0.0 <= m.task_completion_rate <= 100.0
    assert m.collision_count >= 0
    assert m.average_decision_latency_ms >= 0.0
    assert m.p95_decision_latency_ms >= 0.0
    assert 0.0 <= m.deadlock_resolution_rate <= 100.0
    assert m.total_energy_cost >= 0.0
    assert m.average_path_length >= 0.0
    assert m.final_objective_value >= 0.0


def test_throughput_and_coverage_velocity():
    collector = MetricsCollector()
    collector.record_cell_visit((1, 1))
    collector.record_cell_visit((1, 2))
    collector.record_cell_visit((1, 3))
    collector.completed_ticks.append(5)

    agent = AutonomousAgent(agent_id=0, initial_position=(1, 3), target=(1, 3))
    agent.state.completed_target = True
    m = collector.finalize({0: agent}, total_ticks=10, hard_collisions=0, dropout_agent_ids=set())

    assert m.task_throughput == 10.0  # 1 task per 10 ticks = 10 per 100 ticks
    assert m.coverage_velocity == 0.3  # 3 cells in 10 ticks
