"""Unit tests ensuring all reported metrics strictly conform to defined mathematical bounds."""

import pytest
import numpy as np
from src.metrics.collector import MetricsCollector, SwarmMetrics
from src.metrics.latency import LatencyReport


def _create_mock_agent(agent_id: int, active: bool = True, comm: bool = True, moves: int = 10):
    class MockConfig:
        initial_energy = 100.0

    class MockState:
        def __init__(self):
            self.active = active
            self.communication_status = comm
            self.energy = 80.0
            self.target_id = 0
            self.history_path = [np.array([float(i), float(i)]) for i in range(moves)]

    class MockAgent:
        def __init__(self):
            self.config = MockConfig()
            self.state = MockState()

    return MockAgent()


def _create_mock_target(target_id: int, completed: bool = False, tick: int = 10):
    class MockTarget:
        def __init__(self):
            self.target_id = target_id
            self.completed = completed
            self.completed_at_tick = tick if completed else None

    return MockTarget()


def test_agent_failure_recovery_rate_is_ratio_within_0_and_1():
    """Confirms agent_failure_recovery_rate is bounded in [0.0, 1.0]."""
    collector = MetricsCollector(arena_bounds=(-50.0, 50.0, -50.0, 50.0))
    lat_rep = LatencyReport(10.0, 10.0, 15.0, 20.0, 25.0, 5.0, 50.0, True, 10)

    # 2 out of 3 orphan tasks reallocated
    metrics = collector.compute_metrics(
        agents_map={0: _create_mock_agent(0)},
        targets=[_create_mock_target(0, True)],
        total_ticks=10,
        latency_report=lat_rep,
        hard_collisions=0,
        near_collisions=0,
        recovery_times=[5],
        orphan_tasks_reallocated=2,
        total_orphan_tasks=3,
        final_fitness_values=[0.2]
    )

    assert 0.0 <= metrics.agent_failure_recovery_rate <= 1.0
    assert metrics.agent_failure_recovery_rate == pytest.approx(2.0 / 3.0)


def test_task_completion_rate_percentage_bounds():
    """Confirms task completion rate is bounded in [0.0, 100.0]."""
    collector = MetricsCollector(arena_bounds=(-50.0, 50.0, -50.0, 50.0))
    lat_rep = LatencyReport(10.0, 10.0, 15.0, 20.0, 25.0, 5.0, 50.0, True, 10)

    targets = [_create_mock_target(i, completed=(i < 7)) for i in range(10)]
    metrics = collector.compute_metrics(
        agents_map={0: _create_mock_agent(0)},
        targets=targets,
        total_ticks=10,
        latency_report=lat_rep,
        hard_collisions=0,
        near_collisions=0,
        recovery_times=[],
        orphan_tasks_reallocated=0,
        total_orphan_tasks=0,
        final_fitness_values=[0.1]
    )

    assert 0.0 <= metrics.task_completion_rate <= 100.0
    assert metrics.task_completion_rate == 70.0


def test_zero_task_edge_cases():
    """Ensures safe behavior and non-NaN metrics when zero tasks are assigned."""
    collector = MetricsCollector(arena_bounds=(-50.0, 50.0, -50.0, 50.0))
    lat_rep = LatencyReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, True, 0)

    metrics = collector.compute_metrics(
        agents_map={},
        targets=[],
        total_ticks=0,
        latency_report=lat_rep,
        hard_collisions=0,
        near_collisions=0,
        recovery_times=[],
        orphan_tasks_reallocated=0,
        total_orphan_tasks=0,
        final_fitness_values=[]
    )

    assert metrics.task_completion_rate == 0.0
    assert metrics.agent_failure_recovery_rate == 1.0
    assert not any(np.isnan(val) for val in [
        metrics.task_completion_rate,
        metrics.agent_failure_recovery_rate,
        metrics.communication_robustness_score,
        metrics.average_tick_latency_ms
    ])
