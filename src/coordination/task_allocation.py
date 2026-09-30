"""Decentralized consensus-based task allocation and orphan task recovery."""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from src.environment.targets import TaskTarget
from src.agents.state import AgentState


class DecentralizedTaskAllocator:
    """Distributed auction heuristic ensuring conflict-free local task assignments."""

    @staticmethod
    def compute_bid(
        agent_pos: np.ndarray,
        agent_energy: float,
        target: TaskTarget,
        current_target_id: Optional[int] = None
    ) -> float:
        """Calculates multi-criteria bid for a specific task target."""
        dist = float(np.linalg.norm(agent_pos - target.position))
        # Prioritize tasks: higher priority, higher energy, lower distance
        # Add stickiness bonus if currently servicing target to prevent rapid oscillation
        stickiness = 1.25 if target.task_id == current_target_id else 1.0
        normalized_energy = max(0.1, agent_energy / 100.0)

        # Bid formula
        bid = (target.priority * 10.0 * normalized_energy * stickiness) / (dist + 2.0)
        return float(bid)

    def allocate_task(
        self,
        agent_id: int,
        agent_pos: np.ndarray,
        agent_energy: float,
        current_target_id: Optional[int],
        active_targets: List[TaskTarget],
        neighbor_states: List[AgentState]
    ) -> Tuple[Optional[TaskTarget], float]:
        """Runs local consensus auction against immediate communicating neighbors.

        Returns (assigned_target, winning_bid).
        """
        if not active_targets:
            return None, 0.0

        # 1. Identify tasks already claimed or completed by known neighbors
        claimed_by_neighbors: Dict[int, Tuple[int, float]] = {}  # target_id -> (neighbor_id, neighbor_bid)
        for ns in neighbor_states:
            if not ns.active or not ns.communication_status:
                continue
            if ns.target_id is not None:
                # Estimate neighbor's bid
                matching_t = next((t for t in active_targets if t.task_id == ns.target_id), None)
                if matching_t:
                    n_bid = self.compute_bid(ns.position, ns.energy, matching_t, ns.target_id)
                    claimed_by_neighbors[ns.target_id] = (ns.agent_id, n_bid)

        # 2. Score all active targets for this agent
        scored_targets: List[Tuple[TaskTarget, float]] = []
        for target in active_targets:
            if target.completed:
                continue

            my_bid = self.compute_bid(agent_pos, agent_energy, target, current_target_id)

            # Check if neighbor also claimed this target
            if target.task_id in claimed_by_neighbors:
                n_id, n_bid = claimed_by_neighbors[target.task_id]
                # If neighbor's bid is higher (or tie broken by agent_id), concede
                if n_bid > my_bid or (np.isclose(n_bid, my_bid) and n_id < agent_id):
                    continue  # Concede this target to neighbor

            scored_targets.append((target, my_bid))

        if not scored_targets:
            # If all preferred targets are contested, pick target with highest bid anyway or idle
            fallback = max(active_targets, key=lambda t: self.compute_bid(agent_pos, agent_energy, t, current_target_id))
            fallback_bid = self.compute_bid(agent_pos, agent_energy, fallback, current_target_id)
            return fallback, fallback_bid

        # Pick the highest viable bid
        best_target, best_bid = max(scored_targets, key=lambda pair: pair[1])
        return best_target, best_bid
