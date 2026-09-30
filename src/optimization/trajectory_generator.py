"""Candidate trajectory generation incorporating direct, tangential, and stochastic modes."""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import numpy as np
from configs.config import OptimizationParameters, AgentPhysicalConfig, ArenaConfig
from src.environment.obstacles import Obstacle
from src.agents.kinematics import KinematicModel


class TrajectoryGenerator:
    """Synthesizes diverse candidate trajectories for multi-agent swarm evaluation."""

    def __init__(
        self,
        opt_cfg: OptimizationParameters,
        agent_cfg: AgentPhysicalConfig,
        arena_cfg: ArenaConfig,
        kinematics: KinematicModel,
        seed: int = 42
    ):
        self.opt_cfg = opt_cfg
        self.agent_cfg = agent_cfg
        self.arena_cfg = arena_cfg
        self.kinematics = kinematics
        self.rng = np.random.default_rng(seed)

    def generate_direct(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        target_pos: Optional[np.ndarray]
    ) -> List[np.ndarray]:
        """Generates a direct trajectory heading straight for the target."""
        if target_pos is None:
            # Cruise with current velocity or slight dampening
            target_vel = start_vel * 0.95
            waypoints = [start_pos + target_vel * self.arena_cfg.time_step * (k + 1) for k in range(self.opt_cfg.horizon_steps)]
            return self.kinematics.generate_kinematic_trajectory(start_pos, start_vel, waypoints)

        diff = target_pos - start_pos
        dist = np.linalg.norm(diff)
        direction = diff / dist if dist > 1e-4 else np.zeros(2)

        waypoints = []
        for k in range(1, self.opt_cfg.horizon_steps + 1):
            travel = min(dist, self.agent_cfg.max_speed * self.arena_cfg.time_step * k)
            wp = start_pos + direction * travel
            waypoints.append(wp)

        return self.kinematics.generate_kinematic_trajectory(start_pos, start_vel, waypoints)

    def generate_detours(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        target_pos: Optional[np.ndarray],
        obstacles: List[Obstacle]
    ) -> List[List[np.ndarray]]:
        """Generates left and right tangential detours around the closest obstacle."""
        if not obstacles or target_pos is None:
            return []

        # Find closest obstacle within sensor horizon
        closest_obs = None
        min_dist = float("inf")
        for obs in obstacles:
            d = obs.distance_to(start_pos)
            if d < min_dist and d < self.agent_cfg.sensor_radius:
                min_dist = d
                closest_obs = obs

        if closest_obs is None:
            return []

        # Target vector
        to_target = target_pos - start_pos
        dist_target = np.linalg.norm(to_target)
        if dist_target < 1e-4:
            return []
        to_target_norm = to_target / dist_target

        # Perpendicular normal vectors: [-dy, dx] and [dy, -dx]
        tangent_left = np.array([-to_target_norm[1], to_target_norm[0]], dtype=np.float64)
        tangent_right = -tangent_left

        detour_trajs = []
        detour_offset = self.agent_cfg.safety_radius + self.agent_cfg.operational_margin + 2.0

        for tangent in [tangent_left, tangent_right]:
            waypoints = []
            for k in range(1, self.opt_cfg.horizon_steps + 1):
                # Blend forward progress with tangential deflection
                progress = min(dist_target, self.agent_cfg.max_speed * self.arena_cfg.time_step * k)
                # Deflection peaks midway through horizon
                deflection_weight = np.sin((k / self.opt_cfg.horizon_steps) * np.pi)
                wp = start_pos + to_target_norm * progress + tangent * (detour_offset * deflection_weight)
                waypoints.append(wp)

            detour_trajs.append(self.kinematics.generate_kinematic_trajectory(start_pos, start_vel, waypoints))

        return detour_trajs

    def generate_perturbed_candidates(
        self,
        base_trajectory: List[np.ndarray],
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        count: int,
        perturbation_scale: float
    ) -> List[List[np.ndarray]]:
        """Generates stochastic variations of a base trajectory."""
        candidates = []
        if len(base_trajectory) < 2:
            return candidates

        for _ in range(count):
            waypoints = []
            for idx, pt in enumerate(base_trajectory[1:], start=1):
                noise = self.rng.normal(0.0, perturbation_scale, size=2)
                waypoints.append(pt + noise)
            candidates.append(self.kinematics.generate_kinematic_trajectory(start_pos, start_vel, waypoints))

        return candidates
