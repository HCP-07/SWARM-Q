"""Perturbation engine for injecting dynamic obstacles, comm dropouts, and agent failures."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import numpy as np
from src.environment.obstacles import DynamicObstacle, Obstacle


class PerturbationType(str, Enum):
    DYNAMIC_OBSTACLE_SPAWN = "dynamic_obstacle_spawn"
    COMMUNICATION_DROPOUT = "communication_dropout"
    AGENT_FAILURE = "agent_failure"
    HAZARD_ACTIVATION = "hazard_activation"


@dataclass
class PerturbationEvent:
    """A scheduled environmental or operational disturbance."""
    trigger_tick: int
    event_type: PerturbationType
    target_ids: List[int] = field(default_factory=list)  # Affected agent IDs or obstacle IDs
    duration_ticks: int = 1                              # For transient dropouts
    payload: Dict[str, Any] = field(default_factory=dict)
    applied: bool = False
    resolved: bool = False


class PerturbationEngine:
    """Manages perturbation schedule and triggers state modifications during simulation."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.schedule: List[PerturbationEvent] = []
        self.active_events: List[PerturbationEvent] = []
        self.history: List[Dict[str, Any]] = []

    def add_event(self, event: PerturbationEvent) -> None:
        self.schedule.append(event)
        self.schedule.sort(key=lambda e: e.trigger_tick)

    def step(self, current_tick: int, agents_map: Dict[int, Any], obstacles: List[Obstacle]) -> List[PerturbationEvent]:
        """Evaluates triggers at current_tick, applies mutations, and manages active windows."""
        triggered_now: List[PerturbationEvent] = []

        # 1. Trigger pending events
        for event in self.schedule:
            if event.trigger_tick == current_tick and not event.applied:
                event.applied = True
                self.active_events.append(event)
                triggered_now.append(event)
                self._apply_event(event, agents_map, obstacles, current_tick)
                self.history.append({
                    "tick": current_tick,
                    "type": event.event_type.value,
                    "target_ids": event.target_ids,
                    "status": "triggered",
                    "payload": str(event.payload)
                })

        # 2. Check resolution for transient events (e.g. comm dropouts)
        remaining_active: List[PerturbationEvent] = []
        for event in self.active_events:
            if current_tick >= event.trigger_tick + event.duration_ticks:
                event.resolved = True
                self._resolve_event(event, agents_map, obstacles, current_tick)
                self.history.append({
                    "tick": current_tick,
                    "type": event.event_type.value,
                    "target_ids": event.target_ids,
                    "status": "resolved"
                })
            else:
                remaining_active.append(event)
        self.active_events = remaining_active

        return triggered_now

    def _apply_event(
        self,
        event: PerturbationEvent,
        agents_map: Dict[int, Any],
        obstacles: List[Obstacle],
        tick: int
    ) -> None:
        if event.event_type == PerturbationType.COMMUNICATION_DROPOUT:
            for aid in event.target_ids:
                if aid in agents_map:
                    agents_map[aid].state.communication_status = False

        elif event.event_type == PerturbationType.AGENT_FAILURE:
            for aid in event.target_ids:
                if aid in agents_map:
                    agent = agents_map[aid]
                    agent.state.active = False
                    agent.state.failed_at_tick = tick
                    agent.state.velocity = np.zeros(2, dtype=np.float64)

        elif event.event_type == PerturbationType.DYNAMIC_OBSTACLE_SPAWN:
            dyn_obs = event.payload.get("obstacle")
            if dyn_obs is not None and isinstance(dyn_obs, DynamicObstacle):
                obstacles.append(dyn_obs)

    def _resolve_event(
        self,
        event: PerturbationEvent,
        agents_map: Dict[int, Any],
        obstacles: List[Obstacle],
        tick: int
    ) -> None:
        if event.event_type == PerturbationType.COMMUNICATION_DROPOUT:
            for aid in event.target_ids:
                if aid in agents_map and agents_map[aid].state.active:
                    agents_map[aid].state.communication_status = True
