"""Geometric and dynamic obstacles with analytical distance fields and safety checks."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np


class Obstacle(ABC):
    """Abstract base class for static and dynamic obstacles."""

    @abstractmethod
    def distance_to(self, point: np.ndarray) -> float:
        """Returns signed distance to surface: positive outside, negative inside."""
        pass

    @abstractmethod
    def closest_point(self, point: np.ndarray) -> np.ndarray:
        """Returns the point on or inside the obstacle closest to query point."""
        pass

    def repulsive_vector(self, point: np.ndarray, margin: float) -> np.ndarray:
        """Calculates normalized repulsive steering vector pointing away from obstacle."""
        dist = self.distance_to(point)
        if dist >= margin:
            return np.zeros(2, dtype=np.float64)

        closest = self.closest_point(point)
        diff = point - closest
        norm = np.linalg.norm(diff)
        if norm < 1e-6:
            # Point is on or inside boundary; fallback to center-based or default outward normal
            if hasattr(self, "center"):
                c_diff = point - getattr(self, "center")
                c_norm = np.linalg.norm(c_diff)
                if c_norm > 1e-6:
                    return c_diff / c_norm
            return np.array([0.0, 1.0], dtype=np.float64)
        return diff / norm


class CircularObstacle(Obstacle):
    """Static circular obstacle with center (x, y) and radius r."""

    def __init__(self, center: np.ndarray, radius: float, obs_id: str = "circ_obs"):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.obs_id = obs_id

    def distance_to(self, point: np.ndarray) -> float:
        center_dist = np.linalg.norm(point - self.center)
        return float(center_dist - self.radius)

    def closest_point(self, point: np.ndarray) -> np.ndarray:
        diff = point - self.center
        norm = np.linalg.norm(diff)
        if norm < 1e-6:
            return self.center + np.array([self.radius, 0.0])
        # Closest point on the obstacle boundary
        return self.center + (diff / norm) * self.radius

    def repulsive_vector(self, point: np.ndarray, margin: float) -> np.ndarray:
        dist = self.distance_to(point)
        if dist >= margin:
            return np.zeros(2, dtype=np.float64)
        diff = point - self.center
        norm = np.linalg.norm(diff)
        if norm < 1e-6:
            return np.array([0.0, 1.0], dtype=np.float64)
        return diff / norm

    def __repr__(self) -> str:
        return f"CircularObstacle(id={self.obs_id}, center={self.center}, r={self.radius})"


class RectangularObstacle(Obstacle):
    """Static axis-aligned rectangular obstacle with bounds [x_min, y_min, x_max, y_max]."""

    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float, obs_id: str = "rect_obs"):
        self.x_min = min(x_min, x_max)
        self.x_max = max(x_min, x_max)
        self.y_min = min(y_min, y_max)
        self.y_max = max(y_min, y_max)
        self.obs_id = obs_id

    def closest_point(self, point: np.ndarray) -> np.ndarray:
        cx = np.clip(point[0], self.x_min, self.x_max)
        cy = np.clip(point[1], self.y_min, self.y_max)
        return np.array([cx, cy], dtype=np.float64)

    def distance_to(self, point: np.ndarray) -> float:
        cx = np.clip(point[0], self.x_min, self.x_max)
        cy = np.clip(point[1], self.y_min, self.y_max)
        dx = point[0] - cx
        dy = point[1] - cy
        outside_dist = np.hypot(dx, dy)
        if outside_dist > 0.0:
            return float(outside_dist)
        # Point is inside the rectangle; return negative distance to nearest boundary
        dist_to_edges = [
            point[0] - self.x_min,
            self.x_max - point[0],
            point[1] - self.y_min,
            self.y_max - point[1]
        ]
        return -float(min(dist_to_edges))

    def __repr__(self) -> str:
        return f"RectangularObstacle(id={self.obs_id}, x=[{self.x_min}, {self.x_max}], y=[{self.y_min}, {self.y_max}])"


class DynamicObstacle(Obstacle):
    """Dynamic moving circular obstacle with velocity, bounding box bounce or waypoint track."""

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        velocity: np.ndarray,
        bounds: Tuple[float, float, float, float],
        obs_id: str = "dyn_obs"
    ):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.bounds = bounds  # (x_min, x_max, y_min, y_max)
        self.obs_id = obs_id
        self.trajectory_history: List[np.ndarray] = [self.center.copy()]

    def step(self, dt: float) -> None:
        """Advance obstacle position and bounce off arena boundaries."""
        self.center += self.velocity * dt
        x_min, x_max, y_min, y_max = self.bounds

        # Bounce x
        if self.center[0] - self.radius < x_min:
            self.center[0] = x_min + self.radius
            self.velocity[0] *= -1.0
        elif self.center[0] + self.radius > x_max:
            self.center[0] = x_max - self.radius
            self.velocity[0] *= -1.0

        # Bounce y
        if self.center[1] - self.radius < y_min:
            self.center[1] = y_min + self.radius
            self.velocity[1] *= -1.0
        elif self.center[1] + self.radius > y_max:
            self.center[1] = y_max - self.radius
            self.velocity[1] *= -1.0

        self.trajectory_history.append(self.center.copy())

    def distance_to(self, point: np.ndarray) -> float:
        center_dist = np.linalg.norm(point - self.center)
        return float(center_dist - self.radius)

    def closest_point(self, point: np.ndarray) -> np.ndarray:
        diff = point - self.center
        norm = np.linalg.norm(diff)
        if norm < 1e-6:
            return self.center + np.array([self.radius, 0.0])
        return self.center + (diff / norm) * min(norm, self.radius)

    def predict_position(self, future_dt: float) -> np.ndarray:
        """Linear projection for predictive collision checks."""
        return self.center + self.velocity * future_dt

    def __repr__(self) -> str:
        return f"DynamicObstacle(id={self.obs_id}, center={self.center}, vel={self.velocity}, r={self.radius})"


class HazardZone:
    """Environmental hazard zone imposing risk penalty or speed reduction."""

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        risk_multiplier: float = 2.0,
        max_speed_limit: float = 1.0,
        zone_id: str = "hazard"
    ):
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        self.risk_multiplier = float(risk_multiplier)
        self.max_speed_limit = float(max_speed_limit)
        self.zone_id = zone_id

    def contains(self, point: np.ndarray) -> bool:
        return bool(np.linalg.norm(point - self.center) <= self.radius)

    def distance_to(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(point - self.center) - self.radius)
