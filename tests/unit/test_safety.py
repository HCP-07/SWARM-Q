"""Unit tests for safety validation and trajectory repair modules."""

import numpy as np
import pytest
from configs.config import SafetyConstraintsConfig, AgentPhysicalConfig, ArenaConfig
from src.environment.obstacles import CircularObstacle, RectangularObstacle, DynamicObstacle
from src.safety.validator import SafetyValidator
from src.safety.repair import TrajectoryRepair


@pytest.fixture
def safety_environment():
    arena_cfg = ArenaConfig(x_min=0.0, x_max=100.0, y_min=0.0, y_max=100.0)
    agent_cfg = AgentPhysicalConfig(safety_radius=1.5, operational_margin=0.8, max_speed=3.0)
    safety_cfg = SafetyConstraintsConfig(min_agent_separation=3.0, min_obstacle_distance=1.5)

    validator = SafetyValidator(safety_cfg, agent_cfg, arena_cfg)
    repair = TrajectoryRepair(validator, safety_cfg, agent_cfg, arena_cfg)
    return validator, repair, arena_cfg, agent_cfg


def test_circular_obstacle_distance():
    obs = CircularObstacle(center=np.array([50.0, 50.0]), radius=5.0)
    # Outside
    assert np.isclose(obs.distance_to(np.array([50.0, 60.0])), 5.0)
    # On surface
    assert np.isclose(obs.distance_to(np.array([50.0, 55.0])), 0.0)
    # Inside
    assert obs.distance_to(np.array([50.0, 52.0])) < 0.0


def test_rectangular_obstacle_distance():
    obs = RectangularObstacle(x_min=20.0, y_min=20.0, x_max=40.0, y_max=40.0)
    # Outside directly right
    assert np.isclose(obs.distance_to(np.array([50.0, 30.0])), 10.0)
    # On corner
    assert np.isclose(obs.distance_to(np.array([40.0, 40.0])), 0.0)
    # Inside
    assert obs.distance_to(np.array([30.0, 30.0])) < 0.0


def test_boundary_validation(safety_environment):
    validator, _, _, agent_cfg = safety_environment
    # Safe trajectory inside arena
    safe_traj = [np.array([10.0, 10.0]), np.array([12.0, 12.0])]
    res_safe = validator.validate_boundary(safe_traj)
    assert res_safe.is_valid

    # Trajectory exiting left boundary
    violating_traj = [np.array([5.0, 10.0]), np.array([0.5, 10.0])]
    res_violating = validator.validate_boundary(violating_traj)
    assert not res_violating.is_valid
    assert res_violating.violation_type == "boundary"


def test_kinematic_limits_validation(safety_environment):
    validator, _, _, agent_cfg = safety_environment
    dt = 0.1
    # Move 0.2m in 0.1s -> speed 2.0 m/s (valid, limit is 3.0 m/s)
    valid_traj = [np.array([10.0, 10.0]), np.array([10.2, 10.0])]
    res_valid = validator.validate_kinematics(valid_traj, dt)
    assert res_valid.is_valid

    # Move 1.0m in 0.1s -> speed 10.0 m/s (exceeds limit 3.0 m/s)
    invalid_traj = [np.array([10.0, 10.0]), np.array([11.0, 10.0])]
    res_invalid = validator.validate_kinematics(invalid_traj, dt)
    assert not res_invalid.is_valid
    assert res_invalid.violation_type == "kinematics"


def test_inter_agent_separation_validation(safety_environment):
    validator, _, _, _ = safety_environment
    traj_a = [np.array([10.0, 10.0]), np.array([11.0, 10.0])]

    # Agent B is 5m away -> safe
    other_trajs_safe = {1: [np.array([15.0, 10.0]), np.array([16.0, 10.0])]}
    res_safe = validator.validate_agent_separation(0, traj_a, other_trajs_safe)
    assert res_safe.is_valid

    # Agent B is 1m away -> below min separation of 3m
    other_trajs_close = {1: [np.array([10.5, 10.0]), np.array([11.5, 10.0])]}
    res_close = validator.validate_agent_separation(0, traj_a, other_trajs_close)
    assert not res_close.is_valid
    assert res_close.violation_type == "agent_agent"


def test_trajectory_repair_deflects_obstacle(safety_environment):
    validator, repair, _, agent_cfg = safety_environment
    dt = 0.1
    obs = CircularObstacle(center=np.array([20.0, 10.0]), radius=3.0)

    # Infeasible trajectory straight through the center of obstacle
    infeasible_traj = [
        np.array([15.0, 10.0]),
        np.array([17.0, 10.0]),
        np.array([20.0, 10.0]), # Directly in obstacle center!
        np.array([23.0, 10.0]),
        np.array([25.0, 10.0])
    ]

    initial_check = validator.validate_full(0, infeasible_traj, [obs], {}, dt)
    assert not initial_check.is_valid

    repaired_traj, is_safe = repair.repair(0, infeasible_traj, [obs], {}, dt)
    # The repaired trajectory must be valid and collision-free
    final_check = validator.validate_full(0, repaired_traj, [obs], {}, dt)
    assert final_check.is_valid
    assert is_safe
