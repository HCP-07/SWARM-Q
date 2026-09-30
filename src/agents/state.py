"""Agent state representation as typed dataclasses."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class AgentState:
    """Explicit state representation of an autonomous swarm agent."""
    agent_id: int
    position: np.ndarray                             # 2D coordinates [x, y]
    velocity: np.ndarray                             # 2D velocity vector [vx, vy]
    target_id: Optional[int] = None                  # Current assigned target ID
    target_position: Optional[np.ndarray] = None     # Target coordinates [tx, ty]
    energy: float = 100.0                            # Remaining energy capacity
    current_trajectory: List[np.ndarray] = field(default_factory=list) # Waypoints
    local_best: Optional[np.ndarray] = None          # Best trajectory found by agent
    local_best_fitness: float = float("inf")         # Fitness of local best
    neighborhood_best: Optional[np.ndarray] = None   # Best trajectory among neighbors
    communication_status: bool = True                # True = connected, False = dropout
    safety_radius: float = 1.5                       # Minimum separation threshold
    sensor_radius: float = 18.0                      # Environmental awareness radius
    communication_radius: float = 25.0               # Local neighborhood radius
    active: bool = True                              # True = operating, False = failed
    failed_at_tick: Optional[int] = None             # Tick index when agent crashed
    history_path: List[np.ndarray] = field(default_factory=list) # Full trajectory history

    def copy(self) -> AgentState:
        """Deep copy of agent state for isolated neighborhood exchanges."""
        return AgentState(
            agent_id=self.agent_id,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            target_id=self.target_id,
            target_position=self.target_position.copy() if self.target_position is not None else None,
            energy=self.energy,
            current_trajectory=[p.copy() for p in self.current_trajectory],
            local_best=self.local_best.copy() if self.local_best is not None else None,
            local_best_fitness=self.local_best_fitness,
            neighborhood_best=self.neighborhood_best.copy() if self.neighborhood_best is not None else None,
            communication_status=self.communication_status,
            safety_radius=self.safety_radius,
            sensor_radius=self.sensor_radius,
            communication_radius=self.communication_radius,
            active=self.active,
            failed_at_tick=self.failed_at_tick,
            history_path=[p.copy() for p in self.history_path]
        )
