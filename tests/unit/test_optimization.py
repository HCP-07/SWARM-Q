"""Unit tests for fitness evaluator, QPSO engine, and candidate generation."""

import numpy as np
import pytest
from configs.config import FitnessWeightsConfig, ArenaConfig, AgentPhysicalConfig, OptimizationParameters, SafetyConstraintsConfig
from src.optimization.fitness import FitnessEvaluator
from src.agents.kinematics import KinematicModel
from src.safety.validator import SafetyValidator
from src.safety.repair import TrajectoryRepair
from src.optimization.adaptation import SwarmAdaptationController
from src.optimization.trajectory_generator import TrajectoryGenerator
from src.optimization.qpso_engine import QPSOTrajectoryEngine
from src.environment.obstacles import CircularObstacle


def test_fitness_weights_normalization():
    weights = FitnessWeightsConfig(
        w_distance=2.0, w_energy=1.0, w_collision=1.0,
        w_obstacle=1.0, w_conflict=1.0, w_delay=1.0, w_communication=1.0
    )
    weights.validate()
    total = (weights.w_distance + weights.w_energy + weights.w_collision +
             weights.w_obstacle + weights.w_conflict + weights.w_delay +
             weights.w_communication)
    assert np.isclose(total, 1.0)


def test_fitness_evaluator_scores():
    arena_cfg = ArenaConfig()
    agent_cfg = AgentPhysicalConfig()
    weights = FitnessWeightsConfig()
    evaluator = FitnessEvaluator(weights, arena_cfg, agent_cfg)

    target_pos = np.array([50.0, 50.0])
    # Closer trajectory should have lower distance cost than distant trajectory
    close_traj = [np.array([45.0, 45.0]), np.array([49.0, 49.0])]
    far_traj = [np.array([10.0, 10.0]), np.array([11.0, 11.0])]

    bd_close = evaluator.evaluate(close_traj, target_pos, [], {})
    bd_far = evaluator.evaluate(far_traj, target_pos, [], {})

    assert bd_close.distance_cost < bd_far.distance_cost
    assert bd_close.total_fitness < bd_far.total_fitness


def test_qpso_engine_optimization():
    arena_cfg = ArenaConfig()
    agent_cfg = AgentPhysicalConfig()
    opt_cfg = OptimizationParameters()
    safety_cfg = SafetyConstraintsConfig()
    weights = FitnessWeightsConfig()

    kinematics = KinematicModel(agent_cfg, dt=0.1)
    validator = SafetyValidator(safety_cfg, agent_cfg, arena_cfg)
    repair = TrajectoryRepair(validator, safety_cfg, agent_cfg, arena_cfg)
    fitness_eval = FitnessEvaluator(weights, arena_cfg, agent_cfg)
    adapt_ctrl = SwarmAdaptationController(opt_cfg, weights)
    traj_gen = TrajectoryGenerator(opt_cfg, agent_cfg, arena_cfg, kinematics, seed=42)

    engine = QPSOTrajectoryEngine(
        agent_id=0,
        opt_cfg=opt_cfg,
        fitness_evaluator=fitness_eval,
        safety_validator=validator,
        trajectory_repair=repair,
        adaptation_controller=adapt_ctrl,
        trajectory_generator=traj_gen,
        kinematics=kinematics,
        seed=42
    )

    start_pos = np.array([10.0, 10.0])
    start_vel = np.zeros(2)
    target_pos = np.array([20.0, 10.0])

    best_traj, bd = engine.optimize_trajectory(
        current_pos=start_pos,
        current_vel=start_vel,
        target_pos=target_pos,
        obstacles=[],
        other_agent_trajectories={},
        neighbor_states=[],
        dt=0.1
    )

    assert len(best_traj) >= 2
    assert bd.total_fitness < float("inf")
    assert engine.local_best_trajectory is not None
