"""Active trajectory repair and emergency fail-safe synthesis."""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
from src.safety.validator import SafetyValidator, SafetyValidationResult
from src.environment.obstacles import Obstacle, CircularObstacle, RectangularObstacle
from configs.config import SafetyConstraintsConfig, AgentPhysicalConfig, ArenaConfig


class TrajectoryRepair:
    """Repairs infeasible candidate trajectories using tangential detours, APF fields, or failsafe braking."""

    def __init__(
        self,
        validator: SafetyValidator,
        safety_cfg: SafetyConstraintsConfig,
        agent_cfg: AgentPhysicalConfig,
        arena_cfg: ArenaConfig
    ):
        self.validator = validator
        self.safety_cfg = safety_cfg
        self.agent_cfg = agent_cfg
        self.arena_cfg = arena_cfg

    def repair(
        self,
        agent_id: int,
        trajectory: List[np.ndarray],
        obstacles: List[Obstacle],
        other_agent_trajectories: Dict[int, List[np.ndarray]],
        dt: float
    ) -> Tuple[List[np.ndarray], bool]:
        """Attempts active tangential detour repair of trajectory. Returns (repaired_trajectory, success_flag)."""
        if not trajectory:
            return [], False

        # First check if already safe
        initial_check = self.validator.validate_full(
            agent_id, trajectory, obstacles, other_agent_trajectories, dt
        )
        if initial_check.is_valid:
            return trajectory, True

        start_pt = trajectory[0].copy()
        if len(trajectory) < 2:
            return trajectory, False

        # Compute net intended heading
        net_diff = trajectory[-1] - trajectory[0]
        net_dist = np.linalg.norm(net_diff)
        if net_dist > 1e-4:
            net_heading = net_diff / net_dist
        else:
            net_heading = np.array([1.0, 0.0], dtype=np.float64)

        tangent_base = np.array([-net_heading[1], net_heading[0]], dtype=np.float64)
        required_clearance = self.safety_cfg.min_obstacle_distance + self.agent_cfg.operational_margin + 0.5
        min_sep = self.safety_cfg.min_agent_separation + self.agent_cfg.operational_margin

        # Try both clockwise and counter-clockwise tangential bypass curves
        for side in [1.0, -1.0]:
            tangent = tangent_base * side
            curve = [start_pt.copy()]
            curr_p = start_pt.copy()
            curr_v = np.zeros(2, dtype=np.float64)

            for step in range(1, len(trajectory)):
                # Check closest obstacle
                closest_obs = None
                min_d = float("inf")
                for obs in obstacles:
                    d = obs.distance_to(curr_p)
                    if d < min_d:
                        min_d = d
                        closest_obs = obs

                # Steering direction synthesis
                if closest_obs is not None and min_d < required_clearance:
                    n_out = closest_obs.repulsive_vector(curr_p, required_clearance)
                    if np.linalg.norm(n_out) < 1e-4:
                        n_out = tangent
                    desired_dir = 0.25 * net_heading + 0.85 * tangent + 0.40 * n_out
                else:
                    desired_dir = net_heading.copy()

                # Add neighbor agent repulsive evasion
                for other_id, other_traj in other_agent_trajectories.items():
                    if other_id == agent_id or not other_traj or step >= len(other_traj):
                        continue
                    ag_diff = curr_p - other_traj[step]
                    ag_d = float(np.linalg.norm(ag_diff))
                    if ag_d < min_sep:
                        rep_n = ag_diff / max(ag_d, 1e-3)
                        desired_dir += 0.5 * rep_n

                dir_norm = np.linalg.norm(desired_dir)
                if dir_norm > 1e-4:
                    desired_dir = desired_dir / dir_norm
                else:
                    desired_dir = tangent

                desired_vel = desired_dir * self.agent_cfg.max_speed

                # Bounded acceleration
                accel = (desired_vel - curr_v) / dt
                a_mag = np.linalg.norm(accel)
                if a_mag > self.agent_cfg.max_acceleration:
                    accel = (accel / a_mag) * self.agent_cfg.max_acceleration

                curr_v = curr_v + accel * dt
                sp = np.linalg.norm(curr_v)
                if sp > self.agent_cfg.max_speed:
                    curr_v = (curr_v / sp) * self.agent_cfg.max_speed

                curr_p = curr_p + curr_v * dt

                # Arena bounds clipping
                margin = self.agent_cfg.safety_radius + 0.1
                curr_p[0] = np.clip(curr_p[0], self.arena_cfg.x_min + margin, self.arena_cfg.x_max - margin)
                curr_p[1] = np.clip(curr_p[1], self.arena_cfg.y_min + margin, self.arena_cfg.y_max - margin)

                curve.append(curr_p.copy())

            check = self.validator.validate_full(agent_id, curve, obstacles, other_agent_trajectories, dt)
            if check.is_valid:
                return curve, True

        # If bypass failed, generate emergency deceleration / stop trajectory
        emergency_traj = self.generate_emergency_stop(agent_id, trajectory, obstacles, dt)
        check_emergency = self.validator.validate_full(
            agent_id, emergency_traj, obstacles, other_agent_trajectories, dt
        )
        if check_emergency.is_valid:
            return emergency_traj, True

        # Last resort: hold current position
        hold_pos_traj = [start_pt.copy() for _ in range(len(trajectory))]
        return hold_pos_traj, False

    def generate_emergency_stop(
        self,
        agent_id: int,
        trajectory: List[np.ndarray],
        obstacles: List[Obstacle],
        dt: float
    ) -> List[np.ndarray]:
        """Generates a smooth emergency braking trajectory stopping the agent safely."""
        start_pt = trajectory[0].copy()
        if len(trajectory) < 2:
            return [start_pt.copy() for _ in range(len(trajectory))]

        initial_vel = (trajectory[1] - trajectory[0]) / dt
        current_speed = np.linalg.norm(initial_vel)
        if current_speed > self.agent_cfg.max_speed:
            current_speed = self.agent_cfg.max_speed
            initial_vel = (initial_vel / np.linalg.norm(initial_vel)) * current_speed

        direction = initial_vel / current_speed if current_speed > 1e-4 else np.zeros(2)

        stop_traj = [start_pt.copy()]
        curr_p = start_pt.copy()
        curr_speed = current_speed
        decel = self.safety_cfg.emergency_brake_decel

        for _ in range(1, len(trajectory)):
            curr_speed = max(0.0, curr_speed - decel * dt)
            curr_p = curr_p + direction * curr_speed * dt
            # Arena clipping
            margin = self.agent_cfg.safety_radius + 0.1
            curr_p[0] = np.clip(curr_p[0], self.arena_cfg.x_min + margin, self.arena_cfg.x_max - margin)
            curr_p[1] = np.clip(curr_p[1], self.arena_cfg.y_min + margin, self.arena_cfg.y_max - margin)
            stop_traj.append(curr_p.copy())

        return stop_traj

    def repair_trajectory(
        self,
        trajectory: List[np.ndarray],
        obstacles: List[Obstacle],
        other_trajectories: Optional[Dict[int, List[np.ndarray]]] = None,
        report: Optional[SafetyValidationResult] = None,
        agent_id: int = -1,
        dt: float = 0.5
    ) -> Optional[List[np.ndarray]]:
        """Convenience adapter for candidate trajectory repair. Returns repaired trajectory or None."""
        if not trajectory:
            return None
        others = other_trajectories or {}
        repaired, ok = self.repair(
            agent_id=agent_id,
            trajectory=trajectory,
            obstacles=obstacles,
            other_agent_trajectories=others,
            dt=dt
        )
        return repaired if ok else None
