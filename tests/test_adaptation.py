"""Tests for the transparent adaptive heuristic policy rules and weight adaptations."""

import pytest
from config import AdaptiveWeightsConfig
from core.adaptive_policy import AdaptivePolicy, LocalPerceptionState
from core.agent import AutonomousAgent


def test_base_weights_defaults():
    policy = AdaptivePolicy()
    base = policy.base_weights
    assert base.w_goal > 0
    assert base.w_collision > 0
    assert base.w_obstacle > 0
    assert base.w_energy > 0
    assert base.w_momentum > 0
    assert base.w_exploration > 0


def test_collision_threat_escalation():
    policy = AdaptivePolicy()
    state_threat = LocalPerceptionState(
        min_neighbor_dist=1.0,
        min_obstacle_dist=1.0,
        recent_conflicts=2
    )
    adapted = policy.adapt_weights(state_threat)

    assert adapted.w_collision > policy.base_weights.w_collision
    assert adapted.w_goal < policy.base_weights.w_goal
    assert adapted.w_conflict > policy.base_weights.w_conflict


def test_stagnation_breaking_boost():
    policy = AdaptivePolicy()
    state_stagnant = LocalPerceptionState(
        stagnation_ticks=5,
        min_neighbor_dist=5.0,
        min_obstacle_dist=5.0
    )
    adapted = policy.adapt_weights(state_stagnant)

    assert adapted.w_exploration > policy.base_weights.w_exploration
    assert adapted.w_momentum < policy.base_weights.w_momentum
    assert adapted.w_goal < policy.base_weights.w_goal


def test_clear_path_acceleration():
    policy = AdaptivePolicy()
    state_clear = LocalPerceptionState(
        min_neighbor_dist=8.0,
        min_obstacle_dist=6.0,
        stagnation_ticks=0
    )
    adapted = policy.adapt_weights(state_clear)

    assert adapted.w_goal > policy.base_weights.w_goal
    assert adapted.w_exploration < policy.base_weights.w_exploration
    assert adapted.w_momentum > policy.base_weights.w_momentum


def test_communication_dropout_conservatism():
    policy = AdaptivePolicy()
    state_dropout = LocalPerceptionState(
        communication_quality=0.1,
        min_neighbor_dist=4.0,
        min_obstacle_dist=4.0
    )
    adapted = policy.adapt_weights(state_dropout)

    assert adapted.w_collision > policy.base_weights.w_collision
    assert adapted.w_obstacle > policy.base_weights.w_obstacle
    assert adapted.w_conflict < policy.base_weights.w_conflict


def test_adaptation_bounds_respected():
    policy = AdaptivePolicy()
    extreme_state = LocalPerceptionState(
        min_neighbor_dist=0.1,
        min_obstacle_dist=0.1,
        stagnation_ticks=20
    )
    adapted = policy.adapt_weights(extreme_state)

    assert adapted.w_collision <= policy.base_weights.max_collision_weight
    assert adapted.w_exploration <= policy.base_weights.max_exploration_weight
    assert adapted.w_goal >= policy.base_weights.min_goal_weight


def test_deterministic_adaptation():
    policy = AdaptivePolicy()
    state = LocalPerceptionState(
        min_neighbor_dist=1.5,
        min_obstacle_dist=2.0,
        stagnation_ticks=3,
        communication_quality=0.5
    )
    w1 = policy.adapt_weights(state)
    w2 = policy.adapt_weights(state)

    assert w1.w_goal == w2.w_goal
    assert w1.w_collision == w2.w_collision
    assert w1.w_exploration == w2.w_exploration


def test_adaptation_disabled_in_agent():
    # When enable_adaptation is False, last_weights should remain base_weights
    base = AdaptiveWeightsConfig(w_goal=3.0, w_collision=4.5)
    agent = AutonomousAgent(
        agent_id=0,
        initial_position=(5, 5),
        target=(10, 10),
        base_weights=base,
        enable_adaptation=False
    )
    from core.environment import Environment
    agent.perceive_and_plan([], set(), set(), Environment().config)
    assert agent.last_weights.w_goal == base.w_goal
    assert agent.last_weights.w_collision == base.w_collision
