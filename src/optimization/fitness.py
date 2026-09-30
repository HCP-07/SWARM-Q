"""Configurable multi-objective fitness evaluation with normalized metrics."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from configs.config import FitnessWeightsConfig, ArenaConfig, AgentPhysicalConfig
from src.environment.obstacles import Obstacle


@dataclass
class FitnessBreakdown:
    """Detailed scores for each normalized objective component."""
    total_fitness: float
    distance_cost: float
    energy_cost: float
    collision_risk: float
    obstacle_violation: float
    task_conflict: float
    completion_delay: float
    coordination_cost: float


class FitnessEvaluator:
    """Computes multi-objective trajectory fitness with explicit normalized metrics."""

    def __init__(
        self,
        weights: FitnessWeightsConfig,
        arena_cfg: ArenaConfig,
        agent_cfg: AgentPhysicalConfig
    ):
        self.weights = weights
        self.arena_cfg = arena_cfg
        self.agent_cfg = agent_cfg
        self.arena_diagonal = np.hypot(arena_cfg.width, arena_cfg.height)

    def evaluate(
        self,
        trajectory: List[np.ndarray],
        target_position: Optional[np.ndarray],
        obstacles: List[Obstacle],
        other_agent_trajectories: Dict[int, List[np.ndarray]],
        has_task_conflict: bool | float = False,
        neighbor_count: int = 0,
        communication_quality: float = 1.0,
        collision_weight_multiplier: float = 1.0,
        **kwargs
    ) -> FitnessBreakdown:
        """Evaluates total cost (lower is superior) across all normalized dimensions."""
        if not trajectory:
            return FitnessBreakdown(float("inf"), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        dt = self.arena_cfg.time_step
        max_possible_dist = self.agent_cfg.max_speed * max(1, len(trajectory) - 1) * dt

        # 1. Normalized distance to target and progress shortfall
        if target_position is not None:
            init_dist = float(np.linalg.norm(trajectory[0] - target_position))
            final_dist = float(np.linalg.norm(trajectory[-1] - target_position))
            progress = init_dist - final_dist
            f_dist = np.clip(final_dist / (init_dist + 1.0), 0.0, 1.0)
            f_delay = np.clip(1.0 - (progress / max(max_possible_dist, 1e-3)), 0.0, 1.0)
        else:
            f_dist = 0.5
            f_delay = 0.5

        # 2. Normalized maneuver energy consumption (penalizes acceleration jitter)
        acc_sum = 0.0
        for i in range(len(trajectory) - 2):
            v1 = (trajectory[i + 1] - trajectory[i]) / dt
            v2 = (trajectory[i + 2] - trajectory[i + 1]) / dt
            accel = (v2 - v1) / dt
            acc_sum += float(np.linalg.norm(accel))
        max_accel_budget = max(1, len(trajectory) - 2) * self.agent_cfg.max_acceleration
        f_energy = np.clip(acc_sum / max(max_accel_budget, 1e-3), 0.0, 1.0)

        # 3. Collision risk with neighboring agents
        f_collision = 0.0
        min_agent_sep = self.agent_cfg.safety_radius * 2.0
        count_checks = 0

        for other_traj in other_agent_trajectories.values():
            if not other_traj:
                continue
            comp_len = min(len(trajectory), len(other_traj))
            for s in range(comp_len):
                d = float(np.linalg.norm(trajectory[s] - other_traj[s]))
                count_checks += 1
                if d < min_agent_sep * 2.0:
                    f_collision += np.exp(-d / max(min_agent_sep, 1e-2))

        if count_checks > 0:
            f_collision = np.clip(f_collision / max(count_checks, 1), 0.0, 1.0)

        # 4. Obstacle proximity violation risk
        f_obstacle = 0.0
        obs_checks = 0
        min_obs_margin = self.agent_cfg.operational_margin + self.agent_cfg.safety_radius

        for pt in trajectory:
            for obs in obstacles:
                d = obs.distance_to(pt)
                obs_checks += 1
                if d < min_obs_margin * 2.0:
                    safe_d = max(d, 0.0)
                    f_obstacle += np.exp(-safe_d / max(min_obs_margin, 1e-2))

        if obs_checks > 0:
            f_obstacle = np.clip(f_obstacle / max(obs_checks, 1), 0.0, 1.0)

        # 5. Task conflict penalty (smooth sigmoid formulation via expit)
        if isinstance(has_task_conflict, (int, float)) and not isinstance(has_task_conflict, bool):
            f_conflict = float(expit(float(has_task_conflict)))
        elif has_task_conflict:
            f_conflict = float(expit(5.0))  # Smooth near-saturation (~0.993)
        else:
            f_conflict = 0.0

        # 6. Coordination communication cost
        f_comm = np.clip(neighbor_count / 15.0, 0.0, 1.0)

        # Weighted combination with adaptive collision penalty scaling
        w_coll_eff = self.weights.w_collision * float(collision_weight_multiplier)
        total = (
            self.weights.w_distance * f_dist +
            self.weights.w_delay * f_delay +
            self.weights.w_energy * f_energy +
            w_coll_eff * f_collision +
            self.weights.w_obstacle * f_obstacle +
            self.weights.w_conflict * f_conflict +
            self.weights.w_communication * f_comm
        )

        return FitnessBreakdown(
            total_fitness=float(total),
            distance_cost=float(f_dist),
            energy_cost=float(f_energy),
            collision_risk=float(f_collision),
            obstacle_violation=float(f_obstacle),
            task_conflict=float(f_conflict),
            completion_delay=float(f_delay),
            coordination_cost=float(f_comm)
        )
