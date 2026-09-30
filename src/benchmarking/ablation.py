"""Ablation study evaluating the incremental contributions of each architectural module."""

from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from configs.config import SimulationConfig, OptimizationParameters, FitnessWeightsConfig, SafetyConstraintsConfig
from src.simulation.engine import SimulationEngine
from scenarios.scenario_definitions import ScenarioBuilder, ScenarioType
from src.metrics.collector import SwarmMetrics


def _bind_forced_replan(agent):
    orig = agent.plan_step
    def forced_step(obs, tgts, nbs, trajs, dt, force_replan=True):
        return orig(obs, tgts, nbs, trajs, dt, force_replan=True)
    agent.plan_step = forced_step


def _bind_dummy_allocator(agent, aid):
    agent.allocator.allocate_task = lambda a_id, pos, e, cur, targets, nb: (
        targets[aid % len(targets)] if targets else None, 1.0
    )


class AblationStudy:
    """Executes the 6-stage component ablation study under identical seeds."""

    @staticmethod
    def run_all(config: SimulationConfig, scenario_name: str = "full_stress") -> pd.DataFrame:
        results = []

        # -------------------------------------------------------------
        # Stage A: Base Optimizer
        # Fixed parameters, no diversity feedback, no hard repair, no negotiation, replan every tick
        # -------------------------------------------------------------
        cfg_a = SimulationConfig(
            seed=config.seed,
            agent_count=config.agent_count,
            arena=config.arena,
            agent_params=config.agent_params,
            fitness_weights=config.fitness_weights,
            opt_params=OptimizationParameters(
                beta_initial=0.7, beta_min=0.7, beta_max=0.7,
                diversity_threshold=0.0, stagnation_limit=9999, cauchy_scale=0.0
            ),
            safety_params=SafetyConstraintsConfig(max_repair_iterations=0), # Disable active repair
            latency_budget=config.latency_budget
        )
        engine_a = ScenarioBuilder.build(scenario_name, cfg_a)
        for aid, ag in engine_a.agents.items():
            _bind_dummy_allocator(ag, aid)
            _bind_forced_replan(ag)

        metrics_a = engine_a.run(config.arena.max_ticks)
        results.append({"Stage": "A: Base Optimizer", **metrics_a.to_dict()})

        # -------------------------------------------------------------
        # Stage B: + Adaptive Exploration
        # Adds diversity feedback and Cauchy mutation
        # -------------------------------------------------------------
        cfg_b = SimulationConfig(
            seed=config.seed,
            agent_count=config.agent_count,
            arena=config.arena,
            agent_params=config.agent_params,
            fitness_weights=config.fitness_weights,
            opt_params=config.opt_params, # Full adaptive beta + Cauchy
            safety_params=SafetyConstraintsConfig(max_repair_iterations=0),
            latency_budget=config.latency_budget
        )
        engine_b = ScenarioBuilder.build(scenario_name, cfg_b)
        for aid, ag in engine_b.agents.items():
            _bind_dummy_allocator(ag, aid)
            _bind_forced_replan(ag)

        metrics_b = engine_b.run(config.arena.max_ticks)
        results.append({"Stage": "B: + Adaptive Exploration", **metrics_b.to_dict()})

        # -------------------------------------------------------------
        # Stage C: + Collision-Aware Safety
        # Adds hard validation and APF detour repair
        # -------------------------------------------------------------
        cfg_c = SimulationConfig(
            seed=config.seed,
            agent_count=config.agent_count,
            arena=config.arena,
            agent_params=config.agent_params,
            fitness_weights=config.fitness_weights,
            opt_params=config.opt_params,
            safety_params=config.safety_params, # Full safety repair enabled
            latency_budget=config.latency_budget
        )
        engine_c = ScenarioBuilder.build(scenario_name, cfg_c)
        for aid, ag in engine_c.agents.items():
            _bind_dummy_allocator(ag, aid)
            _bind_forced_replan(ag)

        metrics_c = engine_c.run(config.arena.max_ticks)
        results.append({"Stage": "C: + Collision-Aware Safety", **metrics_c.to_dict()})

        # -------------------------------------------------------------
        # Stage D: + Decentralized Negotiation
        # Adds consensus auction / local task negotiation
        # -------------------------------------------------------------
        engine_d = ScenarioBuilder.build(scenario_name, cfg_c)
        for ag in engine_d.agents.values():
            _bind_forced_replan(ag)

        metrics_d = engine_d.run(config.arena.max_ticks)
        results.append({"Stage": "D: + Decentralized Negotiation", **metrics_d.to_dict()})

        # -------------------------------------------------------------
        # Stage E: + Event-Triggered Re-optimization
        # Adds selective replanning (only affected agents replan)
        # -------------------------------------------------------------
        engine_e = ScenarioBuilder.build(scenario_name, cfg_c)
        metrics_e = engine_e.run(config.arena.max_ticks)
        results.append({"Stage": "E: + Event-Triggered Replanning", **metrics_e.to_dict()})

        # -------------------------------------------------------------
        # Stage F: Full Proposed System
        # Complete system with comm loss discount and failure recovery
        # -------------------------------------------------------------
        engine_f = ScenarioBuilder.build(scenario_name, config)
        metrics_f = engine_f.run(config.arena.max_ticks)
        results.append({"Stage": "F: Full Proposed System", **metrics_f.to_dict()})

        df = pd.DataFrame(results)
        return df
