"""Tests for hard safety constraints: zero cell collisions, zero swap collisions, and obstacle avoidance."""

import pytest
from config import Action, EnvironmentConfig, AgentConfig
from core.collision import CollisionChecker
from core.agent import AutonomousAgent, SwarmMessage
from core.swarm import SwarmCoordinator
from scenarios import get_scenario


def test_within_bounds():
    env_cfg = EnvironmentConfig(width=20, height=20)
    assert CollisionChecker.is_within_bounds((0, 0), env_cfg)
    assert CollisionChecker.is_within_bounds((19, 19), env_cfg)
    assert not CollisionChecker.is_within_bounds((-1, 5), env_cfg)
    assert not CollisionChecker.is_within_bounds((20, 5), env_cfg)
    assert not CollisionChecker.is_within_bounds((5, -1), env_cfg)
    assert not CollisionChecker.is_within_bounds((5, 20), env_cfg)


def test_obstacle_free_validation():
    static_obs = {(5, 5), (5, 6)}
    dynamic_obs = {(10, 10)}
    assert CollisionChecker.is_obstacle_free((4, 5), static_obs, dynamic_obs)
    assert not CollisionChecker.is_obstacle_free((5, 5), static_obs, dynamic_obs)
    assert not CollisionChecker.is_obstacle_free((10, 10), static_obs, dynamic_obs)


def test_detect_swap_conflict():
    curr_pos = {0: (5, 5), 1: (5, 6)}
    intended_pos = {0: (5, 6), 1: (5, 5)}

    is_swap = CollisionChecker.detect_swap_conflict(
        agent_id=0,
        current_pos=(5, 5),
        candidate_pos=(5, 6),
        other_current_positions=curr_pos,
        other_intended_positions=intended_pos
    )
    assert is_swap is True


def test_no_swap_when_other_moves_elsewhere():
    curr_pos = {0: (5, 5), 1: (5, 6)}
    intended_pos = {0: (5, 6), 1: (6, 6)}

    is_swap = CollisionChecker.detect_swap_conflict(
        agent_id=0,
        current_pos=(5, 5),
        candidate_pos=(5, 6),
        other_current_positions=curr_pos,
        other_intended_positions=intended_pos
    )
    assert is_swap is False


def test_cell_occupied_priority_resolution():
    priorities = {0: 1.0, 1: 5.0}
    intended = {1: (5, 5)}
    current = {1: (5, 4)}

    occupied = CollisionChecker.is_cell_occupied(
        agent_id=0,
        candidate_pos=(5, 5),
        other_intended_positions=intended,
        other_current_positions=current,
        priorities=priorities
    )
    assert occupied is True


def test_cell_occupied_tie_breaking_by_id():
    priorities = {0: 3.0, 1: 3.0}
    intended = {0: (5, 5)}
    current = {0: (5, 4)}

    occupied = CollisionChecker.is_cell_occupied(
        agent_id=1,
        candidate_pos=(5, 5),
        other_intended_positions=intended,
        other_current_positions=current,
        priorities=priorities
    )
    assert occupied is True


def test_zero_collisions_in_open_scenario():
    coord = get_scenario("open")
    metrics = coord.run(max_ticks=40)
    assert metrics.collision_count == 0


def test_zero_collisions_in_obstacles_scenario():
    coord = get_scenario("obstacles")
    metrics = coord.run(max_ticks=40)
    assert metrics.collision_count == 0


def test_zero_collisions_in_dense_scenario():
    coord = get_scenario("dense")
    metrics = coord.run(max_ticks=30)
    assert metrics.collision_count == 0


def test_zero_collisions_in_dynamic_obstacles_scenario():
    coord = get_scenario("dynamic")
    metrics = coord.run(max_ticks=30)
    assert metrics.collision_count == 0


def test_zero_collisions_in_dropout_scenario():
    coord = get_scenario("dropout")
    metrics = coord.run(max_ticks=30)
    assert metrics.collision_count == 0


def test_zero_collisions_in_failure_scenario():
    coord = get_scenario("failure")
    metrics = coord.run(max_ticks=30)
    assert metrics.collision_count == 0


def test_zero_collisions_in_combined_scenario():
    coord = get_scenario("combined")
    metrics = coord.run(max_ticks=30)
    assert metrics.collision_count == 0
