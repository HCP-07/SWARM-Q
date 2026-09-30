"""Event detection, selective affected-agent scoping, and event logs."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
import numpy as np
from src.environment.obstacles import DynamicObstacle, Obstacle
from src.environment.perturbations import PerturbationEvent


@dataclass
class SwarmEventRecord:
    """Historical telemetry for an event-triggered disturbance."""
    event_id: str
    tick_occurred: int
    event_type: str
    affected_agent_ids: List[int]
    detection_time_ms: float
    replanning_time_ms: float = 0.0
    recovery_ticks: int = 0
    resolved: bool = False


class EventTriggerManager:
    """Identifies environmental disturbances and pinpoints affected agents and 1-hop neighbors."""

    def __init__(self):
        self.event_records: List[SwarmEventRecord] = []

    def identify_affected_agents(
        self,
        agents_map: Dict[int, Any],
        obstacles: List[Obstacle],
        dynamic_obstacles: List[DynamicObstacle],
        perturbations: List[PerturbationEvent],
        tick: int
    ) -> Tuple[Set[int], List[SwarmEventRecord]]:
        """Determines which agents must replan based on localized disturbances."""
        affected_agents: Set[int] = set()
        new_records: List[SwarmEventRecord] = []

        # 1. Perturbations directly targeting agents (e.g. comm dropout, failure)
        for pert in perturbations:
            if pert.trigger_tick == tick:
                for aid in pert.target_ids:
                    if aid in agents_map:
                        affected_agents.add(aid)
                        # Add immediate physical neighbors
                        pos_a = agents_map[aid].state.position
                        for other_id, other_ag in agents_map.items():
                            if other_id != aid and other_ag.state.active:
                                if float(np.linalg.norm(other_ag.state.position - pos_a)) <= other_ag.state.communication_radius:
                                    affected_agents.add(other_id)

                rec = SwarmEventRecord(
                    event_id=f"pert_{tick}_{pert.event_type.value}",
                    tick_occurred=tick,
                    event_type=pert.event_type.value,
                    affected_agent_ids=sorted(list(affected_agents)),
                    detection_time_ms=0.05
                )
                self.event_records.append(rec)
                new_records.append(rec)

        # 2. Dynamic obstacles intruding on agent paths
        for dyn_obs in dynamic_obstacles:
            for aid, ag in agents_map.items():
                if not ag.state.active:
                    continue

                # Check if dynamic obstacle is inside agent's sensor radius
                dist_to_obs = dyn_obs.distance_to(ag.state.position)
                if dist_to_obs <= ag.state.sensor_radius:
                    # Check if obstacle crosses current trajectory
                    traj = ag.state.current_trajectory
                    intersects = False
                    for pt in traj:
                        if dyn_obs.distance_to(pt) <= dyn_obs.radius + ag.state.safety_radius + 1.0:
                            intersects = True
                            break

                    if intersects:
                        affected_agents.add(aid)

        return affected_agents, new_records
