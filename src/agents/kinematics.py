"""Kinematic models and continuous motion integration."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from configs.config import AgentPhysicalConfig


class KinematicModel:
    """Bounded acceleration and velocity integrator."""

    def __init__(self, config: AgentPhysicalConfig, dt: float = 0.1):
        self.config = config
        self.dt = dt

    def clamp_velocity(self, velocity: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(velocity)
        if speed > self.config.max_speed:
            return (velocity / speed) * self.config.max_speed
        return velocity

    def compute_acceleration(self, current_velocity: np.ndarray, target_velocity: np.ndarray) -> np.ndarray:
        desired_accel = (target_velocity - current_velocity) / self.dt
        accel_mag = np.linalg.norm(desired_accel)
        if accel_mag > self.config.max_acceleration:
            return (desired_accel / accel_mag) * self.config.max_acceleration
        return desired_accel

    def step(self, position: np.ndarray, velocity: np.ndarray, desired_velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Integrates motion over dt, clamping limits, and returns (new_pos, new_vel, energy_spent)."""
        accel = self.compute_acceleration(velocity, desired_velocity)
        new_velocity = self.clamp_velocity(velocity + accel * self.dt)
        new_position = position + new_velocity * self.dt

        # Energy consumption: idle + velocity^2 + acceleration^2
        speed_sq = float(np.sum(new_velocity ** 2))
        accel_sq = float(np.sum(accel ** 2))
        energy_spent = (
            self.config.idle_energy_rate +
            self.config.motion_energy_rate * speed_sq * self.dt +
            0.05 * accel_sq * self.dt
        )
        return new_position, new_velocity, energy_spent

    def generate_kinematic_trajectory(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        waypoints: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Projects a physically realizable trajectory sequence through waypoints."""
        trajectory: List[np.ndarray] = [start_pos.copy()]
        curr_p = start_pos.copy()
        curr_v = start_vel.copy()

        for wp in waypoints:
            diff = wp - curr_p
            dist = np.linalg.norm(diff)
            if dist > 1e-4:
                desired_v = (diff / dist) * min(self.config.max_speed, dist / self.dt)
            else:
                desired_v = np.zeros(2, dtype=np.float64)

            curr_p, curr_v, _ = self.step(curr_p, curr_v, desired_v)
            trajectory.append(curr_p.copy())

        return trajectory
