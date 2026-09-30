"""Decentralized conflict resolution and right-of-way negotiation."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.agents.state import AgentState


class ConflictNegotiator:
    """Resolves pairwise path conflicts and determines right-of-way."""

    @staticmethod
    def resolve_trajectory_conflict(
        agent_id: int,
        agent_traj: List[np.ndarray],
        neighbor_id: int,
        neighbor_traj: List[np.ndarray],
        min_separation: float
    ) -> bool:
        """Determines if agent_id should yield to neighbor_id.

        Tie-breaking rule: The agent with higher remaining energy or lower agent ID
        has right-of-way; the yielding agent decelerates or detours.
        Returns True if agent_id must yield, False if agent_id has right-of-way.
        """
        if not agent_traj or not neighbor_traj:
            return False

        comp_len = min(len(agent_traj), len(neighbor_traj))
        for step in range(comp_len):
            dist = float(np.linalg.norm(agent_traj[step] - neighbor_traj[step]))
            if dist < min_separation:
                # Conflict detected at step
                # Deterministic right-of-way: higher ID yields
                return agent_id > neighbor_id

        return False
