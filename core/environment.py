"""2D grid world environment, spatial indexing, dynamic obstacle simulation, and perturbation injector."""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List, Dict, Set, Optional, Any
from config import EnvironmentConfig


class PerturbationType(str, Enum):
    """Dynamic perturbation event types."""
    OBSTACLE_APPEAR = "obstacle_appear"
    OBSTACLE_DISAPPEAR = "obstacle_disappear"
    COMMUNICATION_DROPOUT = "communication_dropout"
    AGENT_FAILURE = "agent_failure"


@dataclass
class PerturbationEvent:
    """Scheduled environmental or hardware perturbation."""
    trigger_tick: int
    event_type: PerturbationType
    payload: Dict[str, Any]
    applied: bool = False


@dataclass
class DynamicObstacle:
    """Obstacle moving through the 2D grid space."""
    obs_id: str
    position: Tuple[int, int]
    velocity: Tuple[int, int]

    def step(self, env_cfg: EnvironmentConfig) -> None:
        """Updates obstacle position with reflective boundary bouncing."""
        new_x = self.position[0] + self.velocity[0]
        new_y = self.position[1] + self.velocity[1]
        vx, vy = self.velocity

        if new_x < 0 or new_x >= env_cfg.width:
            vx = -vx
            new_x = max(0, min(env_cfg.width - 1, self.position[0] + vx))

        if new_y < 0 or new_y >= env_cfg.height:
            vy = -vy
            new_y = max(0, min(env_cfg.height - 1, self.position[1] + vy))

        self.position = (new_x, new_y)
        self.velocity = (vx, vy)


class Environment:
    """Manages spatial entities, spatial hashing, and perturbation event injection."""

    def __init__(self, config: Optional[EnvironmentConfig] = None, seed: int = 42):
        self.config = config or EnvironmentConfig()
        self.rng = random.Random(seed)
        self.static_obstacles: Set[Tuple[int, int]] = set()
        self.dynamic_obstacles: Dict[str, DynamicObstacle] = {}
        self.targets: Dict[int, Tuple[int, int]] = {}
        self.perturbations: List[PerturbationEvent] = []
        self.active_dropouts: Dict[int, int] = {}  # agent_id -> remaining dropout ticks
        self.bucket_size = 5
        self.spatial_buckets: Dict[Tuple[int, int], List[int]] = {}

    def add_static_obstacle(self, pos: Tuple[int, int]) -> None:
        """Adds an immovable obstacle cell."""
        if 0 <= pos[0] < self.config.width and 0 <= pos[1] < self.config.height:
            self.static_obstacles.add(pos)

    def add_dynamic_obstacle(self, obs_id: str, start_pos: Tuple[int, int], velocity: Tuple[int, int]) -> None:
        """Adds a moving obstacle."""
        self.dynamic_obstacles[obs_id] = DynamicObstacle(obs_id, start_pos, velocity)

    def add_perturbation(self, event: PerturbationEvent) -> None:
        """Enqueues a perturbation to fire at a specified tick."""
        self.perturbations.append(event)

    def get_dynamic_obstacle_positions(self) -> Set[Tuple[int, int]]:
        """Returns the current set of dynamic obstacle coordinates."""
        return {obs.position for obs in self.dynamic_obstacles.values()}

    def step_environment(self, tick: int, agents_map: Dict[int, Any]) -> Set[int]:
        """Advances dynamic obstacles and applies scheduled perturbations.
        
        Returns:
            Set[int]: agent IDs directly affected by dynamic events on this tick.
        """
        affected_agents: Set[int] = set()

        # 1. Advance moving dynamic obstacles
        for obs in self.dynamic_obstacles.values():
            obs.step(self.config)

        # 2. Check and apply scheduled perturbations
        for event in self.perturbations:
            if event.trigger_tick == tick and not event.applied:
                event.applied = True

                if event.event_type == PerturbationType.OBSTACLE_APPEAR:
                    for pos in event.payload.get("positions", []):
                        self.static_obstacles.add(pos)
                        # Notify nearby agents within 8 units
                        for aid, ag in agents_map.items():
                            if math.hypot(ag.state.position[0] - pos[0], ag.state.position[1] - pos[1]) <= 8.0:
                                affected_agents.add(aid)

                elif event.event_type == PerturbationType.OBSTACLE_DISAPPEAR:
                    for pos in event.payload.get("positions", []):
                        self.static_obstacles.discard(pos)
                        for aid, ag in agents_map.items():
                            if math.hypot(ag.state.position[0] - pos[0], ag.state.position[1] - pos[1]) <= 8.0:
                                affected_agents.add(aid)

                elif event.event_type == PerturbationType.COMMUNICATION_DROPOUT:
                    target_ids = event.payload.get("agent_ids", [])
                    duration = event.payload.get("duration", 20)
                    for aid in target_ids:
                        self.active_dropouts[aid] = duration
                        affected_agents.add(aid)

                elif event.event_type == PerturbationType.AGENT_FAILURE:
                    fail_id = event.payload.get("agent_id")
                    if fail_id in agents_map:
                        failed_ag = agents_map[fail_id]
                        failed_ag.state.active = False
                        affected_agents.add(fail_id)
                        # Orphan target reassignment: notify nearest active agents
                        orphan_target = failed_ag.state.target
                        for aid, ag in agents_map.items():
                            if ag.state.active and aid != fail_id:
                                if math.hypot(ag.state.position[0] - orphan_target[0], ag.state.position[1] - orphan_target[1]) <= 12.0:
                                    affected_agents.add(aid)

        # 3. Decrement active dropouts
        expired = []
        for aid in self.active_dropouts:
            self.active_dropouts[aid] -= 1
            if self.active_dropouts[aid] <= 0:
                expired.append(aid)
        for aid in expired:
            del self.active_dropouts[aid]

        return affected_agents

    def update_spatial_index(self, agents_map: Dict[int, Any]) -> None:
        """Constructs an O(N) 2D spatial grid bucket index for rapid local neighbor retrieval."""
        self.spatial_buckets.clear()
        for aid, ag in agents_map.items():
            if not ag.state.active:
                continue
            bx = ag.state.position[0] // self.bucket_size
            by = ag.state.position[1] // self.bucket_size
            self.spatial_buckets.setdefault((bx, by), []).append(aid)

    def get_nearby_agent_ids(self, position: Tuple[int, int], radius: float, agents_map: Dict[int, Any]) -> List[int]:
        """Fast spatial hash lookup for agents within circular Euclidean radius."""
        min_bx = int((position[0] - radius) // self.bucket_size)
        max_bx = int((position[0] + radius) // self.bucket_size)
        min_by = int((position[1] - radius) // self.bucket_size)
        max_by = int((position[1] + radius) // self.bucket_size)

        nearby = []
        r_sq = radius * radius
        for bx in range(min_bx, max_bx + 1):
            for by in range(min_by, max_by + 1):
                for aid in self.spatial_buckets.get((bx, by), []):
                    ag = agents_map[aid]
                    dx = ag.state.position[0] - position[0]
                    dy = ag.state.position[1] - position[1]
                    if dx * dx + dy * dy <= r_sq:
                        nearby.append(aid)
        return nearby
