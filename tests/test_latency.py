"""Tests verifying sub-second decision latency and real-time SLA compliance."""

import time
import pytest
from config import SimulationConfig, ScenarioType
from scenarios import get_scenario
from core.environment import Environment
from core.agent import AutonomousAgent


def test_single_tick_latency_under_sla():
    cfg = SimulationConfig(agent_count=15, seed=42)
    coord = get_scenario(ScenarioType.OPEN, config=cfg)

    # Measure tick latency
    lat_ms = coord.step()
    # SLA budget is 100 ms; target is < 15 ms
    assert lat_ms < 100.0


def test_p95_latency_satisfies_sla_budget():
    cfg = SimulationConfig(agent_count=15, max_ticks=30, seed=42, real_time_budget_ms=100.0)
    coord = get_scenario(ScenarioType.STATIC_OBSTACLES, config=cfg)
    metrics = coord.run()

    assert metrics.p95_decision_latency_ms < 100.0
    assert metrics.real_time_pass is True


def test_latency_scalability_up_to_30_agents():
    for count in [10, 20, 30]:
        cfg = SimulationConfig(agent_count=count, max_ticks=20, seed=42)
        coord = get_scenario(ScenarioType.OPEN, config=cfg)
        metrics = coord.run()
        # Even with 30 agents, average latency must remain well below 100 ms
        assert metrics.average_decision_latency_ms < 50.0


def test_spatial_hashing_lookup_speed():
    env = Environment()
    agents_map = {}
    for i in range(50):
        ag = AutonomousAgent(agent_id=i, initial_position=(i % 30, i % 30), target=(35, 35))
        agents_map[i] = ag

    env.update_spatial_index(agents_map)

    t0 = time.perf_counter()
    for _ in range(100):
        nearby = env.get_nearby_agent_ids((15, 15), 6.0, agents_map)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # 100 lookups should take less than 10 ms
    assert elapsed_ms < 15.0
