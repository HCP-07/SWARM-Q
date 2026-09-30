"""Decentralized Quantum-behaved Particle Swarm Optimization (QPSO) trajectory engine."""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from scipy.special import expit

from configs.config import OptimizationParameters, FitnessWeightsConfig, AgentPhysicalConfig, ArenaConfig
from src.environment.obstacles import Obstacle
from src.safety.validator import SafetyValidator
from src.safety.repair import TrajectoryRepair
from src.optimization.fitness import FitnessEvaluator, FitnessBreakdown
from src.optimization.adaptation import SwarmAdaptationController, SwarmAdaptationState
from src.optimization.trajectory_generator import TrajectoryGenerator
from src.agents.kinematics import KinematicModel


def qpso_update(
    x: np.ndarray,
    pbest: np.ndarray,
    gbest: np.ndarray,
    mbest: np.ndarray,
    beta: float,
    rng: np.random.Generator
) -> np.ndarray:
    """One QPSO step: attractor + sign * beta * |mbest - x| * ln(1/u)."""
    phi = rng.random(x.shape)
    attractor = phi * pbest + (1.0 - phi) * gbest
    u = np.clip(rng.random(x.shape), 1e-12, 1.0)
    sign = np.where(rng.random(x.shape) < 0.5, 1.0, -1.0)
    return attractor + sign * beta * np.abs(mbest - x) * np.log(1.0 / u)


def conflict_penalty(bid_self: float, bid_other: float, k: float = 10.0) -> float:
    """Smooth version of the boolean conflict flag: 1 / (1 + exp(k * (b_i - b_j)))."""
    return float(expit(-k * (bid_self - bid_other)))


class QPSOTrajectoryEngine:
    """Decentralized QPSO optimizer running locally on an autonomous agent."""

    def __init__(
        self,
        agent_id: int,
        opt_cfg: OptimizationParameters,
        fitness_evaluator: FitnessEvaluator,
        safety_validator: SafetyValidator,
        trajectory_repair: TrajectoryRepair,
        adaptation_controller: SwarmAdaptationController,
        trajectory_generator: TrajectoryGenerator,
        kinematics: KinematicModel,
        seed: int = 42
    ):
        self.agent_id = agent_id
        self.opt_cfg = opt_cfg
        self.fitness_eval = fitness_evaluator
        self.safety_val = safety_validator
        self.repair = trajectory_repair
        self.adaptation_ctrl = adaptation_controller
        self.traj_gen = trajectory_generator
        self.kinematics = kinematics
        self.rng = np.random.default_rng(seed + agent_id * 101)

        self.adaptation_state = SwarmAdaptationState(
            beta=opt_cfg.beta_initial,
            diversity=opt_cfg.diversity_threshold
        )
        self.local_best_trajectory: Optional[List[np.ndarray]] = None
        self.local_best_fitness: float = float("inf")

    def optimize_trajectory(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        target_pos: Optional[np.ndarray],
        obstacles: List[Obstacle],
        other_agent_trajectories: Dict[int, List[np.ndarray]],
        neighbor_states: List[Any],
        dt: float,
        has_task_conflict: bool = False
    ) -> Tuple[List[np.ndarray], FitnessBreakdown]:
        """Synthesizes candidate trajectories, applies QPSO sampling, safety repair, and selects best."""
        # 1. Update adaptation parameters
        self.adaptation_state = self.adaptation_ctrl.update_parameters(
            current_pos=current_pos,
            neighbor_states=neighbor_states,
            obstacles=obstacles,
            current_best_fitness=self.local_best_fitness
        )

        # 2. Extract neighborhood best and mean best
        neighborhood_best_traj: Optional[List[np.ndarray]] = self.local_best_trajectory
        neighborhood_best_fit: float = self.local_best_fitness

        for nb in neighbor_states:
            if nb.neighborhood_best is not None and getattr(nb, "local_best_fitness", float("inf")) < neighborhood_best_fit:
                neighborhood_best_fit = nb.local_best_fitness
                neighborhood_best_traj = nb.neighborhood_best

        # Compute mean best trajectory across communicating neighbors
        mbest_points: List[np.ndarray] = []
        if self.local_best_trajectory is not None:
            horizon = len(self.local_best_trajectory)
            for step_idx in range(horizon):
                valid_pts = [self.local_best_trajectory[step_idx]]
                for nb in neighbor_states:
                    if nb.neighborhood_best is not None and step_idx < len(nb.neighborhood_best):
                        valid_pts.append(nb.neighborhood_best[step_idx])
                mbest_points.append(np.mean(valid_pts, axis=0))

        # 3. Generate candidate trajectories
        candidates: List[List[np.ndarray]] = []

        # A. Direct candidate towards target
        direct_traj = self.traj_gen.generate_direct(current_pos, current_vel, target_pos)
        candidates.append(direct_traj)

        # B. Tangential detours around nearest obstacles
        detours = self.traj_gen.generate_detours(current_pos, current_vel, target_pos, obstacles)
        candidates.extend(detours)

        # C. QPSO quantum potential well candidates
        beta = self.adaptation_state.beta
        if self.local_best_trajectory is not None and mbest_points:
            num_qpso_samples = max(4, self.opt_cfg.candidate_count - len(candidates))
            for _ in range(num_qpso_samples):
                sample_waypoints: List[np.ndarray] = []
                for s in range(1, len(self.local_best_trajectory)):
                    pbest_pt = self.local_best_trajectory[s]
                    if neighborhood_best_traj is not None and s < len(neighborhood_best_traj):
                        gbest_pt = neighborhood_best_traj[s]
                    else:
                        gbest_pt = pbest_pt

                    mbest_pt = mbest_points[s]
                    sample_pt = qpso_update(
                        x=pbest_pt,
                        pbest=pbest_pt,
                        gbest=gbest_pt,
                        mbest=mbest_pt,
                        beta=beta,
                        rng=self.rng
                    )
                    sample_waypoints.append(sample_pt)

                # If stagnation triggered, apply Cauchy mutation
                if self.adaptation_state.mutation_triggered:
                    sample_waypoints = self.adaptation_ctrl.apply_cauchy_mutation(sample_waypoints, self.rng)

                cand = self.kinematics.generate_kinematic_trajectory(current_pos, current_vel, sample_waypoints)
                candidates.append(cand)
        else:
            # Cold-start stochastic perturbations of direct trajectory
            stochastic_samples = self.traj_gen.generate_perturbed_candidates(
                direct_traj, current_pos, current_vel,
                count=max(6, self.opt_cfg.candidate_count - len(candidates)),
                perturbation_scale=1.5
            )
            candidates.extend(stochastic_samples)

        # 4. Safety Validation, Repair, and Fitness Evaluation
        best_candidate: Optional[List[np.ndarray]] = None
        best_breakdown: Optional[FitnessBreakdown] = None
        min_cost = float("inf")

        for cand in candidates:
            # Validate candidate
            report = self.safety_val.validate_trajectory(
                trajectory=cand,
                obstacles=obstacles,
                other_agent_trajectories=other_agent_trajectories
            )

            # Repair if invalid
            if not report.is_safe and self.repair is not None:
                repaired_cand = self.repair.repair_trajectory(
                    trajectory=cand,
                    obstacles=obstacles,
                    other_trajectories=other_agent_trajectories,
                    report=report
                )
                if repaired_cand is not None:
                    repaired_report = self.safety_val.validate_trajectory(
                        trajectory=repaired_cand,
                        obstacles=obstacles,
                        other_agent_trajectories=other_agent_trajectories
                    )
                    if repaired_report.is_safe:
                        cand = repaired_cand

            # Evaluate fitness
            comm_quality = 1.0 if len(neighbor_states) > 0 else 0.0
            breakdown = self.fitness_eval.evaluate(
                trajectory=cand,
                target_position=target_pos,
                obstacles=obstacles,
                other_agent_trajectories=other_agent_trajectories,
                has_task_conflict=has_task_conflict,
                communication_quality=comm_quality,
                collision_weight_multiplier=self.adaptation_state.collision_weight_scale
            )

            if breakdown.total_fitness < min_cost:
                min_cost = breakdown.total_fitness
                best_candidate = cand
                best_breakdown = breakdown

        if best_candidate is None:
            best_candidate = direct_traj
            best_breakdown = self.fitness_eval.evaluate(
                best_candidate, target_pos, obstacles, other_agent_trajectories
            )

        # Update local best
        if min_cost < self.local_best_fitness:
            self.local_best_fitness = min_cost
            self.local_best_trajectory = [p.copy() for p in best_candidate]

        return best_candidate, best_breakdown
