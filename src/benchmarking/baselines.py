"""Benchmark baselines: Greedy Nearest, Classical PSO, Random Local Search, and Proposed ADSO."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np

from configs.config import SimulationConfig, OptimizationParameters, FitnessWeightsConfig
from src.simulation.engine import SimulationEngine
from scenarios.scenario_definitions import ScenarioBuilder, ScenarioType
from src.metrics.collector import SwarmMetrics


class BaselineRunner:
    """Orchestrates benchmark runs across baseline solvers and the proposed method."""

    @staticmethod
    def run_proposed_adso(scenario_name: str, config: SimulationConfig) -> Tuple[SwarmMetrics, SimulationEngine]:
        """Runs the complete Proposed Adaptive Decentralized Swarm Optimizer (ADSO)."""
        engine = ScenarioBuilder.build(scenario_name, config)
        metrics = engine.run(config.arena.max_ticks)
        return metrics, engine

    @staticmethod
    def run_classical_pso(scenario_name: str, config: SimulationConfig) -> Tuple[SwarmMetrics, SimulationEngine]:
        """Classical PSO: static inertia/cognitive/social weights, no diversity feedback, no Cauchy mutation."""
        pso_config = SimulationConfig(
            seed=config.seed,
            agent_count=config.agent_count,
            arena=config.arena,
            agent_params=config.agent_params,
            fitness_weights=config.fitness_weights,
            opt_params=OptimizationParameters(
                horizon_steps=config.opt_params.horizon_steps,
                candidate_count=config.opt_params.candidate_count,
                beta_initial=0.7,
                beta_min=0.7,      # Static beta (no diversity adaptation)
                beta_max=0.7,
                diversity_threshold=0.0,
                stagnation_limit=9999, # Disable Cauchy mutation
                cauchy_scale=0.0
            ),
            safety_params=config.safety_params,
            latency_budget=config.latency_budget
        )
        engine = ScenarioBuilder.build(scenario_name, pso_config)

        # Force continuous periodic replanning (no event-triggered caching)
        original_step = engine.step
        def periodic_step() -> float:
            for ag in engine.agents.values():
                ag.state.current_trajectory.clear()  # Force replan every tick
            return original_step()
        engine.step = periodic_step

        metrics = engine.run(config.arena.max_ticks)
        return metrics, engine

    @staticmethod
    def run_greedy_nearest(scenario_name: str, config: SimulationConfig) -> Tuple[SwarmMetrics, SimulationEngine]:
        """Greedy Nearest: agents head straight to closest target, simple reactive stop, no swarm optimization."""
        engine = ScenarioBuilder.build(scenario_name, config)

        # Override agent planning with direct line-of-sight greedy target heading
        for ag in engine.agents.values():
            def make_greedy_plan(agent_ref):
                def greedy_plan(local_obs, local_targets, neighbors, other_trajs, dt, force_replan=False):
                    if not agent_ref.state.active:
                        return False
                    # Pick nearest active target
                    if local_targets:
                        nearest = min(local_targets, key=lambda t: float(np.linalg.norm(agent_ref.state.position - t.position)))
                        agent_ref.state.target_id = nearest.task_id
                        agent_ref.state.target_position = nearest.position.copy()
                    else:
                        agent_ref.state.target_id = None
                        agent_ref.state.target_position = None

                    # Direct trajectory
                    traj = agent_ref.optimizer.traj_gen.generate_direct(
                        agent_ref.state.position, agent_ref.state.velocity, agent_ref.state.target_position
                    )
                    # Pass through safety repair
                    safe_traj, _ = agent_ref.repair.repair(agent_ref.agent_id, traj, local_obs, other_trajs, dt)
                    agent_ref.state.current_trajectory = safe_traj
                    agent_ref.state.local_best = safe_traj
                    return True
                return greedy_plan
            ag.plan_step = make_greedy_plan(ag)

        metrics = engine.run(config.arena.max_ticks)
        return metrics, engine

    @staticmethod
    def run_random_local_search(scenario_name: str, config: SimulationConfig) -> Tuple[SwarmMetrics, SimulationEngine]:
        """Random Local Search: APF guidance + stochastic random walk perturbation."""
        engine = ScenarioBuilder.build(scenario_name, config)

        for ag in engine.agents.values():
            def make_random_plan(agent_ref):
                def random_plan(local_obs, local_targets, neighbors, other_trajs, dt, force_replan=False):
                    if not agent_ref.state.active:
                        return False
                    if local_targets:
                        agent_ref.state.target_id = local_targets[0].task_id
                        agent_ref.state.target_position = local_targets[0].position.copy()

                    direct = agent_ref.optimizer.traj_gen.generate_direct(
                        agent_ref.state.position, agent_ref.state.velocity, agent_ref.state.target_position
                    )
                    # Random walk perturbations
                    perturbed = agent_ref.optimizer.traj_gen.generate_perturbed_candidates(
                        direct, agent_ref.state.position, agent_ref.state.velocity, count=8, perturbation_scale=3.0
                    )
                    best_cand = direct
                    best_c = float("inf")
                    for c in [direct] + perturbed:
                        repaired, is_safe = agent_ref.repair.repair(agent_ref.agent_id, c, local_obs, other_trajs, dt)
                        bd = agent_ref.optimizer.fitness_eval.evaluate(repaired, agent_ref.state.target_position, local_obs, other_trajs)
                        if bd.total_fitness < best_c:
                            best_c = bd.total_fitness
                            best_cand = repaired

                    agent_ref.state.current_trajectory = best_cand
                    agent_ref.state.local_best = best_cand
                    return True
                return random_plan
            ag.plan_step = make_random_plan(ag)

        metrics = engine.run(config.arena.max_ticks)
        return metrics, engine
