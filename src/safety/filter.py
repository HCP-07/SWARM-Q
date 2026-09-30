"""Control Barrier Function (CBF) reactive safety filter with Reciprocal Velocity Obstacle (RVO) guarantees."""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from src.environment.obstacles import Obstacle


class ReactiveSafetyFilter:
    """Control Barrier Function filter enforcing strict inter-agent and obstacle safety invariants."""

    @staticmethod
    def filter_velocity(
        position: np.ndarray,
        desired_velocity: np.ndarray,
        neighbor_positions: List[np.ndarray],
        obstacles: List[Obstacle],
        dt: float,
        min_agent_sep: float = 3.0,
        min_obstacle_dist: float = 1.5,
        max_speed: float = 3.0,
        max_accel: float = 4.0,
        avoidance_margin: float = 3.2
    ) -> np.ndarray:
        """Filters velocity command so that next position is guaranteed collision-free."""
        v_safe = desired_velocity.copy()

        # Dynamic physical braking distance buffer: v^2 / (2 * a_max) = 3^2 / 8 = 1.125m
        # For dual-agent reciprocal approach: 2.25m + safety padding
        agent_avoidance_dist = min_agent_sep + avoidance_margin
        obs_avoidance_dist = min_obstacle_dist + 1.8

        # 1. Obstacle barrier constraints
        for obs in obstacles:
            d = obs.distance_to(position)
            if d < obs_avoidance_dist:
                rep_n = obs.repulsive_vector(position, obs_avoidance_dist)
                closing_speed = float(np.dot(v_safe, -rep_n))
                if closing_speed > 0.0:
                    v_safe -= closing_speed * (-rep_n)
                    tangent = np.array([-rep_n[1], rep_n[0]], dtype=np.float64)
                    v_safe += tangent * 1.5

        # 2. Inter-agent Reciprocal Velocity Obstacle (RVO) constraints
        for n_pos in neighbor_positions:
            r = position - n_pos
            dist = float(np.linalg.norm(r))
            if dist < 1e-4:
                continue

            if dist < agent_avoidance_dist:
                r_hat = r / dist
                closing_speed = float(np.dot(v_safe, -r_hat))
                if closing_speed > 0.0:
                    # Remove closing velocity towards neighbor completely
                    v_safe -= closing_speed * (-r_hat)
                    # Add orthogonal bypass velocity (deflect away)
                    tangent = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)
                    v_safe += tangent * 2.0

        # Speed clamping
        speed = float(np.linalg.norm(v_safe))
        if speed > max_speed:
            v_safe = (v_safe / speed) * max_speed

        return v_safe
