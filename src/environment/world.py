"""Continuous 2D environment orchestrating static/dynamic obstacles, hazards, and tasks."""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from src.environment.obstacles import Obstacle, CircularObstacle, RectangularObstacle, DynamicObstacle, HazardZone
from src.environment.targets import TaskTarget
from src.environment.perturbations import PerturbationEngine, PerturbationEvent
from configs.config import ArenaConfig


class Environment:
    """Continuous 2D simulation world."""

    def __init__(self, arena_config: ArenaConfig, seed: int = 42):
        self.arena = arena_config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.obstacles: List[Obstacle] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []
        self.hazard_zones: List[HazardZone] = []
        self.targets: List[TaskTarget] = []
        self.perturbation_engine = PerturbationEngine(seed=seed)
        self.current_tick: int = 0

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)
        if isinstance(obstacle, DynamicObstacle):
            self.dynamic_obstacles.append(obstacle)

    def add_hazard_zone(self, hazard: HazardZone) -> None:
        self.hazard_zones.append(hazard)

    def add_target(self, target: TaskTarget) -> None:
        self.targets.append(target)

    def step(self, agents_map: Dict[int, Any]) -> List[PerturbationEvent]:
        """Advance time step dt, move dynamic obstacles, evaluate perturbations."""
        # 1. Advance dynamic obstacles
        for dyn_obs in self.dynamic_obstacles:
            dyn_obs.step(self.arena.time_step)

        # 2. Advance perturbation engine
        triggered_events = self.perturbation_engine.step(
            self.current_tick, agents_map, self.obstacles
        )

        # Sync any newly spawned dynamic obstacles from perturbations
        for obs in self.obstacles:
            if isinstance(obs, DynamicObstacle) and obs not in self.dynamic_obstacles:
                self.dynamic_obstacles.append(obs)

        self.current_tick += 1
        return triggered_events

    def get_active_targets(self) -> List[TaskTarget]:
        return [t for t in self.targets if not t.completed]

    def get_all_obstacles(self) -> List[Obstacle]:
        return self.obstacles

    def is_inside_arena(self, point: np.ndarray, margin: float = 0.0) -> bool:
        x, y = point[0], point[1]
        return bool(
            (self.arena.x_min + margin <= x <= self.arena.x_max - margin) and
            (self.arena.y_min + margin <= y <= self.arena.y_max - margin)
        )

    def reset(self) -> None:
        """Reset world to initial state."""
        self.current_tick = 0
        self.dynamic_obstacles.clear()
        self.obstacles.clear()
        self.hazard_zones.clear()
        self.targets.clear()
        self.rng = np.random.default_rng(self.seed)
        self.perturbation_engine = PerturbationEngine(seed=self.seed)
