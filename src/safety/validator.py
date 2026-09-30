"""Dedicated hard safety validation module for multi-agent trajectories."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from src.environment.obstacles import Obstacle
from configs.config import SafetyConstraintsConfig, AgentPhysicalConfig, ArenaConfig


@dataclass
class SafetyValidationResult:
    """Result of hard safety constraint verification."""
    is_valid: bool
    violation_type: Optional[str] = None      # "boundary", "obstacle", "agent_agent", "kinematics"
    violation_step: Optional[int] = None      # Waypoint index where violation occurred
    min_observed_distance: float = float("inf")
    violating_entity_id: Optional[str] = None
    details: str = "Safe"

    @property
    def is_safe(self) -> bool:
        return self.is_valid


class SafetyValidator:
    """Validates trajectories against hard physical, environmental, and inter-agent boundaries."""

    def __init__(
        self,
        safety_config: SafetyConstraintsConfig,
        agent_config: AgentPhysicalConfig,
        arena_config: ArenaConfig
    ):
        self.safety_cfg = safety_config
        self.agent_cfg = agent_config
        self.arena_cfg = arena_config

    def validate_boundary(self, trajectory: List[np.ndarray]) -> SafetyValidationResult:
        """Validates that all points in trajectory stay within arena boundaries."""
        margin = self.agent_cfg.safety_radius
        for idx, pt in enumerate(trajectory):
            x, y = pt[0], pt[1]
            if (x < self.arena_cfg.x_min + margin or
                x > self.arena_cfg.x_max - margin or
                y < self.arena_cfg.y_min + margin or
                y > self.arena_cfg.y_max - margin):
                return SafetyValidationResult(
                    is_valid=False,
                    violation_type="boundary",
                    violation_step=idx,
                    details=f"Point {pt} violated arena bounds at step {idx}"
                )
        return SafetyValidationResult(is_valid=True)

    def validate_kinematics(self, trajectory: List[np.ndarray], dt: float) -> SafetyValidationResult:
        """Validates that velocity and acceleration do not exceed physical limits."""
        if len(trajectory) < 2:
            return SafetyValidationResult(is_valid=True)

        velocities = []
        for i in range(len(trajectory) - 1):
            vel = (trajectory[i + 1] - trajectory[i]) / dt
            speed = np.linalg.norm(vel)
            if speed > self.agent_cfg.max_speed * 1.05:  # 5% tolerance for numerical rounding
                return SafetyValidationResult(
                    is_valid=False,
                    violation_type="kinematics",
                    violation_step=i,
                    details=f"Speed {speed:.2f} m/s exceeded limit {self.agent_cfg.max_speed:.2f} at step {i}"
                )
            velocities.append(vel)

        for i in range(len(velocities) - 1):
            accel = (velocities[i + 1] - velocities[i]) / dt
            accel_mag = np.linalg.norm(accel)
            if accel_mag > self.agent_cfg.max_acceleration * 1.05:
                return SafetyValidationResult(
                    is_valid=False,
                    violation_type="kinematics",
                    violation_step=i,
                    details=f"Acceleration {accel_mag:.2f} m/s^2 exceeded limit at step {i}"
                )

        return SafetyValidationResult(is_valid=True)

    def validate_obstacle_clearance(
        self,
        trajectory: List[np.ndarray],
        obstacles: List[Obstacle]
    ) -> SafetyValidationResult:
        """Ensures trajectory maintains minimum safe operational margin from obstacles."""
        required_clearance = self.safety_cfg.min_obstacle_distance
        min_dist = float("inf")

        for idx, pt in enumerate(trajectory):
            for obs in obstacles:
                dist = obs.distance_to(pt)
                if dist < min_dist:
                    min_dist = dist
                if dist < required_clearance:
                    obs_id = getattr(obs, "obs_id", "obstacle")
                    return SafetyValidationResult(
                        is_valid=False,
                        violation_type="obstacle",
                        violation_step=idx,
                        min_observed_distance=dist,
                        violating_entity_id=obs_id,
                        details=f"Distance {dist:.2f}m to {obs_id} below safe margin {required_clearance:.2f}m at step {idx}"
                    )

        return SafetyValidationResult(is_valid=True, min_observed_distance=min_dist)

    def validate_agent_separation(
        self,
        agent_id: int,
        trajectory: List[np.ndarray],
        other_agent_trajectories: Dict[int, List[np.ndarray]]
    ) -> SafetyValidationResult:
        """Validates that candidate trajectory maintains minimum inter-agent separation with buffer."""
        min_separation = self.safety_cfg.min_agent_separation + (self.agent_cfg.operational_margin * 0.5)
        min_dist = float("inf")

        for other_id, other_traj in other_agent_trajectories.items():
            if other_id == agent_id or not other_traj:
                continue

            for step in range(len(trajectory)):
                # If other agent trajectory is shorter, project last known position
                if step < len(other_traj):
                    other_pt = other_traj[step]
                else:
                    other_pt = other_traj[-1]

                dist = float(np.linalg.norm(trajectory[step] - other_pt))
                if dist < min_dist:
                    min_dist = dist
                if dist < min_separation:
                    return SafetyValidationResult(
                        is_valid=False,
                        violation_type="agent_agent",
                        violation_step=step,
                        min_observed_distance=dist,
                        violating_entity_id=str(other_id),
                        details=f"Inter-agent distance {dist:.2f}m with Agent {other_id} below separation threshold {min_separation:.2f}m at step {step}"
                    )

        return SafetyValidationResult(is_valid=True, min_observed_distance=min_dist)

    def validate_full(
        self,
        agent_id: int,
        trajectory: List[np.ndarray],
        obstacles: List[Obstacle],
        other_agent_trajectories: Dict[int, List[np.ndarray]],
        dt: float
    ) -> SafetyValidationResult:
        """Sequential complete safety audit: boundary -> kinematics -> obstacles -> agents."""
        # 1. Boundary check
        res_bound = self.validate_boundary(trajectory)
        if not res_bound.is_valid:
            return res_bound

        # 2. Kinematic check
        res_kin = self.validate_kinematics(trajectory, dt)
        if not res_kin.is_valid:
            return res_kin

        # 3. Obstacle clearance check
        res_obs = self.validate_obstacle_clearance(trajectory, obstacles)
        if not res_obs.is_valid:
            return res_obs

        # 4. Inter-agent separation check
        res_agents = self.validate_agent_separation(agent_id, trajectory, other_agent_trajectories)
        if not res_agents.is_valid:
            return res_agents

        min_obs_dist = min(res_obs.min_observed_distance, res_agents.min_observed_distance)
        return SafetyValidationResult(is_valid=True, min_observed_distance=min_obs_dist)

    def validate_trajectory(
        self,
        trajectory: List[np.ndarray],
        obstacles: List[Obstacle],
        other_agent_trajectories: Optional[Dict[int, List[np.ndarray]]] = None,
        agent_id: int = -1,
        dt: float = 0.5
    ) -> SafetyValidationResult:
        """Convenience validation entrypoint checking all safety invariants."""
        return self.validate_full(
            agent_id=agent_id,
            trajectory=trajectory,
            obstacles=obstacles,
            other_agent_trajectories=other_agent_trajectories or {},
            dt=dt
        )
