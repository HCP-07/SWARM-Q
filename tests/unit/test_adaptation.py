"""Unit tests for parameter adaptation rules."""

import numpy as np
import pytest
from configs.config import OptimizationParameters, FitnessWeightsConfig
from src.optimization.adaptation import SwarmAdaptationController, SwarmAdaptationState


def test_diversity_feedback_increases_exploration():
    opt_cfg = OptimizationParameters(
        beta_initial=0.7,
        beta_min=0.3,
        beta_max=1.2,
        diversity_threshold=5.0
    )
    weights = FitnessWeightsConfig()
    ctrl = SwarmAdaptationController(opt_cfg, weights)

    agent_pos = np.array([10.0, 10.0])
    # Clustered neighbors (low diversity < 5.0)
    clustered_neighbors = [np.array([10.2, 10.1]), np.array([10.1, 10.3])]

    init_state = SwarmAdaptationState(beta=0.7, diversity=5.0)
    new_state, _ = ctrl.update_adaptation(
        current_state=init_state,
        agent_pos=agent_pos,
        neighbor_positions=clustered_neighbors,
        current_fitness=0.5,
        best_fitness=0.4,
        min_obstacle_dist=20.0,
        min_agent_dist=20.0,
        comm_active=True
    )

    # Beta should have expanded for exploration
    assert new_state.beta > 0.7


def test_collision_risk_escalation():
    opt_cfg = OptimizationParameters()
    base_weights = FitnessWeightsConfig(w_collision=0.25, w_obstacle=0.15)
    ctrl = SwarmAdaptationController(opt_cfg, base_weights)

    agent_pos = np.array([10.0, 10.0])
    init_state = SwarmAdaptationState(beta=0.7, diversity=5.0)

    # Critical distance deficit: 0.5m (below critical safety margin 3.0m)
    _, adapted_weights = ctrl.update_adaptation(
        current_state=init_state,
        agent_pos=agent_pos,
        neighbor_positions=[],
        current_fitness=0.5,
        best_fitness=0.4,
        min_obstacle_dist=0.5,
        min_agent_dist=10.0,
        comm_active=True,
        critical_safety_margin=3.0
    )

    # Collision & obstacle weights should escalate
    assert adapted_weights.w_collision > base_weights.w_collision
    assert adapted_weights.w_obstacle > base_weights.w_obstacle


def test_stagnation_triggers_cauchy_mutation():
    opt_cfg = OptimizationParameters(stagnation_limit=3)
    ctrl = SwarmAdaptationController(opt_cfg, FitnessWeightsConfig())

    agent_pos = np.array([10.0, 10.0])
    state = SwarmAdaptationState(beta=0.7, diversity=5.0, stagnant_ticks=2)

    # Fitness has not improved
    new_state, _ = ctrl.update_adaptation(
        current_state=state,
        agent_pos=agent_pos,
        neighbor_positions=[],
        current_fitness=0.8,
        best_fitness=0.5,
        min_obstacle_dist=20.0,
        min_agent_dist=20.0,
        comm_active=True
    )

    assert new_state.mutation_triggered
    assert new_state.stagnant_ticks == 0
