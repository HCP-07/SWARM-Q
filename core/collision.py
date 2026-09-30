"""Collision avoidance, boundary validation, and hard safety enforcement."""

from __future__ import annotations
from typing import Tuple, Dict, Set, Optional, List
from config import Action, EnvironmentConfig


class CollisionChecker:
    """Enforces zero-tolerance safety constraints including cell collisions and head-on swaps."""

    @staticmethod
    def is_within_bounds(pos: Tuple[int, int], env_cfg: EnvironmentConfig) -> bool:
        """Verifies if coordinate (x, y) resides inside arena limits."""
        x, y = pos
        return 0 <= x < env_cfg.width and 0 <= y < env_cfg.height

    @staticmethod
    def is_obstacle_free(
        pos: Tuple[int, int],
        static_obstacles: Set[Tuple[int, int]],
        dynamic_obstacles: Set[Tuple[int, int]]
    ) -> bool:
        """Verifies candidate cell is not obstructed by static or moving obstacles."""
        if pos in static_obstacles:
            return False
        if pos in dynamic_obstacles:
            return False
        return True

    @staticmethod
    def detect_swap_conflict(
        agent_id: int,
        current_pos: Tuple[int, int],
        candidate_pos: Tuple[int, int],
        other_current_positions: Dict[int, Tuple[int, int]],
        other_intended_positions: Dict[int, Tuple[int, int]]
    ) -> bool:
        """Detects illegal head-on swap maneuvers (Agent A: X->Y while Agent B: Y->X)."""
        if candidate_pos == current_pos:
            return False  # Waiting causes no swap

        for other_id, other_curr in other_current_positions.items():
            if other_id == agent_id:
                continue
            # If the candidate target is currently held by other_id
            if candidate_pos == other_curr:
                # Check if other_id is simultaneously moving into current_pos
                other_next = other_intended_positions.get(other_id)
                if other_next == current_pos:
                    return True
        return False

    @staticmethod
    def is_cell_occupied(
        agent_id: int,
        candidate_pos: Tuple[int, int],
        other_intended_positions: Dict[int, Tuple[int, int]],
        other_current_positions: Dict[int, Tuple[int, int]],
        priorities: Optional[Dict[int, float]] = None
    ) -> bool:
        """Checks if another agent has higher priority or already reserved the candidate cell."""
        my_prio = priorities.get(agent_id, 0.0) if priorities else 0.0

        for other_id, other_next in other_intended_positions.items():
            if other_id == agent_id:
                continue
            if other_next == candidate_pos:
                other_prio = priorities.get(other_id, 0.0) if priorities else 0.0
                # If other agent has strictly higher priority, we cannot take this cell
                if other_prio > my_prio or (other_prio == my_prio and other_id < agent_id):
                    return True

        return False

    @classmethod
    def validate_action(
        cls,
        agent_id: int,
        current_pos: Tuple[int, int],
        action: Action,
        env_cfg: EnvironmentConfig,
        static_obstacles: Set[Tuple[int, int]],
        dynamic_obstacles: Set[Tuple[int, int]],
        other_current_positions: Dict[int, Tuple[int, int]],
        other_intended_positions: Dict[int, Tuple[int, int]],
        priorities: Optional[Dict[int, float]] = None
    ) -> Tuple[bool, str]:
        """Performs complete hard safety validation of candidate action.
        
        Returns:
            Tuple[bool, str]: (is_safe, failure_reason)
        """
        next_pos = (current_pos[0] + action.dx, current_pos[1] + action.dy)

        # 1. Boundary check
        if not cls.is_within_bounds(next_pos, env_cfg):
            return False, "boundary_violation"

        # 2. Obstacle check
        if not cls.is_obstacle_free(next_pos, static_obstacles, dynamic_obstacles):
            return False, "obstacle_collision"

        # 3. Swap collision check
        if cls.detect_swap_conflict(
            agent_id, current_pos, next_pos,
            other_current_positions, other_intended_positions
        ):
            return False, "swap_collision"

        # 4. Destination conflict / priority occupancy
        if cls.is_cell_occupied(
            agent_id, next_pos,
            other_intended_positions, other_current_positions,
            priorities
        ):
            return False, "cell_occupied"

        return True, "valid"
