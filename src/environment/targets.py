"""Task targets and delivery objectives for multi-agent allocation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TaskTarget:
    """Represents a discrete spatial target or task that agents must service."""
    task_id: int
    position: np.ndarray
    priority: float = 1.0           # 1.0 (standard) to 5.0 (critical priority)
    required_energy: float = 5.0    # Energy required to service task
    completed: bool = False
    completed_at_tick: Optional[int] = None
    completed_by_agent: Optional[int] = None
    assigned_agent_id: Optional[int] = None
    satisfaction_radius: float = 2.0  # Distance threshold to mark task fulfilled

    def distance_to(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(point - self.position))

    def is_reached(self, point: np.ndarray) -> bool:
        return bool(self.distance_to(point) <= self.satisfaction_radius)

    def mark_completed(self, agent_id: int, tick: int) -> None:
        self.completed = True
        self.completed_by_agent = agent_id
        self.completed_at_tick = tick

    def reset(self) -> None:
        self.completed = False
        self.completed_at_tick = None
        self.completed_by_agent = None
        self.assigned_agent_id = None
