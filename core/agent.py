"""Autonomous agent representation, local perception, candidate scoring, and state transitions."""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Set, Optional, Any
from config import Action, AgentConfig, AdaptiveWeightsConfig, EnvironmentConfig
from core.collision import CollisionChecker
from core.adaptive_policy import AdaptivePolicy, LocalPerceptionState


@dataclass
class SwarmMessage:
    """Compact heartbeat message broadcast strictly within local communication radius."""
    agent_id: int
    position: Tuple[int, int]
    intended_next_position: Tuple[int, int]
    target: Tuple[int, int]
    priority: float
    local_risk: float
    active: bool


@dataclass
class AgentState:
    """Local, private internal state of an autonomous agent."""
    agent_id: int
    position: Tuple[int, int]
    target: Tuple[int, int]
    energy: float
    active: bool = True
    intended_position: Tuple[int, int] = (0, 0)
    last_action: Action = Action.WAIT
    stagnation_ticks: int = 0
    recent_conflicts: int = 0
    total_path_length: float = 0.0
    total_energy_expended: float = 0.0
    completed_target: bool = False
    deadlocks_encountered: int = 0
    deadlocks_resolved: int = 0
    history: List[Tuple[int, int]] = field(default_factory=list)


class AutonomousAgent:
    """Decentralized autonomous agent executing local perception, adaptation, and navigation."""

    def __init__(
        self,
        agent_id: int,
        initial_position: Tuple[int, int],
        target: Tuple[int, int],
        config: Optional[AgentConfig] = None,
        base_weights: Optional[AdaptiveWeightsConfig] = None,
        enable_adaptation: bool = True
    ):
        self.config = config or AgentConfig()
        self.enable_adaptation = enable_adaptation
        self.policy = AdaptivePolicy(base_weights)
        self.state = AgentState(
            agent_id=agent_id,
            position=initial_position,
            target=target,
            energy=self.config.initial_energy,
            intended_position=initial_position,
            history=[initial_position]
        )
        self.last_weights = base_weights or AdaptiveWeightsConfig()
        self._ranked_candidates: List[Tuple[Action, float, Tuple[int, int]]] = []

    def get_priority(self) -> float:
        """Computes dynamic priority based on target proximity and remaining energy."""
        dist = self._dist(self.state.position, self.state.target)
        proximity_urgency = 10.0 / (max(0.5, dist) + 0.1)
        energy_ratio = self.state.energy / max(1.0, self.config.initial_energy)
        stagnation_boost = 2.0 if self.state.stagnation_ticks >= self.config.stagnation_threshold else 0.0
        return proximity_urgency + energy_ratio + stagnation_boost

    def create_broadcast_message(self) -> SwarmMessage:
        """Generates compact local status message for neighboring agents."""
        return SwarmMessage(
            agent_id=self.state.agent_id,
            position=self.state.position,
            intended_next_position=self.state.intended_position,
            target=self.state.target,
            priority=self.get_priority(),
            local_risk=1.0 if self.state.stagnation_ticks > 2 else 0.0,
            active=self.state.active
        )

    def plan_candidates(
        self,
        neighbor_messages: List[SwarmMessage],
        static_obstacles: Set[Tuple[int, int]],
        dynamic_obstacles: Set[Tuple[int, int]],
        env_cfg: EnvironmentConfig,
        communication_quality: float = 1.0,
        rng: Optional[random.Random] = None
    ) -> Tuple[int, int]:
        """Phase 1: Local perception, adaptive heuristic weighting, and candidate ranking.
        
        Returns:
            Tuple[int, int]: Preferred intended next cell.
        """
        if not self.state.active or self.state.completed_target:
            self.state.intended_position = self.state.position
            self._ranked_candidates = [(Action.WAIT, 0.0, self.state.position)]
            return self.state.position

        local_rng = rng or random.Random(self.state.agent_id)

        # 1. Analyze local neighborhood
        other_curr = {m.agent_id: m.position for m in neighbor_messages if m.active}
        min_nbr_d = min([self._dist(self.state.position, p) for p in other_curr.values()], default=float("inf"))
        min_obs_d = float("inf")
        for obs in static_obstacles | dynamic_obstacles:
            d = self._dist(self.state.position, obs)
            if d < min_obs_d:
                min_obs_d = d

        curr_target_dist = self._dist(self.state.position, self.state.target)

        # 2. Local perception state & adaptive weights
        perception = LocalPerceptionState(
            nearby_agent_count=len(neighbor_messages),
            min_neighbor_dist=min_nbr_d,
            min_obstacle_dist=min_obs_d,
            stagnation_ticks=self.state.stagnation_ticks,
            recent_conflicts=self.state.recent_conflicts,
            communication_quality=communication_quality,
            target_distance=curr_target_dist
        )

        if self.enable_adaptation:
            weights = self.policy.adapt_weights(perception)
        else:
            weights = self.policy.base_weights

        self.last_weights = weights

        # 3. Evaluate candidate actions (all 9 discrete choices)
        candidates: List[Tuple[Action, float, Tuple[int, int]]] = []

        for act in Action:
            cand_pos = (self.state.position[0] + act.dx, self.state.position[1] + act.dy)

            # Hard boundary check
            if not CollisionChecker.is_within_bounds(cand_pos, env_cfg):
                continue

            # Hard obstacle check
            if not CollisionChecker.is_obstacle_free(cand_pos, static_obstacles, dynamic_obstacles):
                continue

            # Score candidate action
            score = self._score_action(
                action=act,
                cand_pos=cand_pos,
                weights=weights,
                other_curr=other_curr,
                static_obs=static_obstacles,
                dynamic_obs=dynamic_obstacles,
                rng=local_rng
            )
            candidates.append((act, score, cand_pos))

        # Always include WAIT as guaranteed safe fallback
        if not any(c[0] == Action.WAIT for c in candidates):
            candidates.append((Action.WAIT, -100.0, self.state.position))

        # Sort descending by score
        candidates.sort(key=lambda item: item[1], reverse=True)
        self._ranked_candidates = candidates

        # Set intended position to top candidate
        top_pos = candidates[0][2]
        self.state.intended_position = top_pos
        return top_pos

    def resolve_and_select_action(
        self,
        neighbor_messages: List[SwarmMessage],
        priorities: Dict[int, float]
    ) -> Action:
        """Phase 2: Local conflict resolution and yielding logic.
        
        Inspects neighbor intended positions. If another agent with higher priority
        claims the same destination cell or creates a swap conflict, selects
        the highest-scoring non-conflicting candidate or WAIT.
        """
        if not self.state.active or self.state.completed_target:
            return Action.WAIT

        my_id = self.state.agent_id
        my_prio = priorities.get(my_id, self.get_priority())

        # Collect higher-priority intended destinations and current positions of neighbors
        higher_priority_destinations: Set[Tuple[int, int]] = set()
        higher_priority_current: Dict[int, Tuple[int, int]] = {}
        all_neighbor_current: Dict[int, Tuple[int, int]] = {}

        for m in neighbor_messages:
            if not m.active or m.agent_id == my_id:
                continue
            all_neighbor_current[m.agent_id] = m.position
            other_prio = priorities.get(m.agent_id, m.priority)

            # Strict priority order: higher prio wins; break tie with lower agent_id
            if other_prio > my_prio or (other_prio == my_prio and m.agent_id < my_id):
                higher_priority_destinations.add(m.intended_next_position)
                higher_priority_current[m.agent_id] = m.position

        # Find best candidate that has NO conflict with higher-priority neighbors
        chosen_action = Action.WAIT
        chosen_pos = self.state.position

        for act, score, cand_pos in self._ranked_candidates:
            # 1. Destination cell conflict: higher-priority neighbor already claiming this cell
            if cand_pos in higher_priority_destinations:
                continue

            # 2. Swap conflict: higher-priority agent is at cand_pos and moving into our current pos
            is_swap = False
            for other_id, other_pos in higher_priority_current.items():
                if cand_pos == other_pos:
                    # check if other agent is heading into our cell
                    for m in neighbor_messages:
                        if m.agent_id == other_id and m.intended_next_position == self.state.position:
                            is_swap = True
                            break
                if is_swap:
                    break

            if is_swap:
                continue

            # Found valid safe candidate
            chosen_action = act
            chosen_pos = cand_pos
            break

        if chosen_action == Action.WAIT and self._ranked_candidates and self._ranked_candidates[0][0] != Action.WAIT:
            # Yielded due to conflict
            self.state.recent_conflicts += 1

        self.state.intended_position = chosen_pos
        return chosen_action

    def perceive_and_plan(
        self,
        neighbor_messages: List[SwarmMessage],
        static_obstacles: Set[Tuple[int, int]],
        dynamic_obstacles: Set[Tuple[int, int]],
        env_cfg: EnvironmentConfig,
        communication_quality: float = 1.0,
        rng: Optional[random.Random] = None
    ) -> Action:
        """Single-step convenience wrapper executing both candidate ranking and selection."""
        self.plan_candidates(
            neighbor_messages=neighbor_messages,
            static_obstacles=static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
            env_cfg=env_cfg,
            communication_quality=communication_quality,
            rng=rng
        )
        priorities = {m.agent_id: m.priority for m in neighbor_messages if m.active}
        priorities[self.state.agent_id] = self.get_priority()
        return self.resolve_and_select_action(neighbor_messages, priorities)

    def execute_action(self, action: Action) -> None:
        """Commits the verified action, updates position, tracks energy and deadlocks."""
        if not self.state.active or self.state.completed_target:
            return

        old_pos = self.state.position
        next_pos = (old_pos[0] + action.dx, old_pos[1] + action.dy)

        # Update movement & energy
        step_len = math.hypot(action.dx, action.dy)
        self.state.position = next_pos
        self.state.last_action = action
        self.state.total_path_length += step_len

        cost = self.config.energy_move_cost * action.step_cost if action != Action.WAIT else self.config.energy_wait_cost
        self.state.energy = max(0.0, self.state.energy - cost)
        self.state.total_energy_expended += cost
        self.state.history.append(next_pos)

        # Check target reach
        if next_pos == self.state.target:
            self.state.completed_target = True

        # Stagnation & deadlock tracking
        old_dist = self._dist(old_pos, self.state.target)
        new_dist = self._dist(next_pos, self.state.target)

        if new_dist < old_dist - 0.1:
            # Positive progress
            if self.state.stagnation_ticks >= self.config.stagnation_threshold:
                self.state.deadlocks_resolved += 1
            self.state.stagnation_ticks = 0
            self.state.recent_conflicts = max(0, self.state.recent_conflicts - 1)
        else:
            # Stagnant or waiting
            self.state.stagnation_ticks += 1
            if self.state.stagnation_ticks == self.config.stagnation_threshold:
                self.state.deadlocks_encountered += 1

    def _score_action(
        self,
        action: Action,
        cand_pos: Tuple[int, int],
        weights: AdaptiveWeightsConfig,
        other_curr: Dict[int, Tuple[int, int]],
        static_obs: Set[Tuple[int, int]],
        dynamic_obs: Set[Tuple[int, int]],
        rng: random.Random
    ) -> float:
        """Evaluates heuristic score for a candidate action."""
        curr_dist = self._dist(self.state.position, self.state.target)
        new_dist = self._dist(cand_pos, self.state.target)

        # 1. Goal progress component (positive for closer, negative for further)
        goal_progress = curr_dist - new_dist

        # 2. Proximity risk to currently occupied neighbor positions
        coll_risk = 0.0
        for other_p in other_curr.values():
            d = self._dist(cand_pos, other_p)
            if d < 2.5:
                coll_risk += 1.0 / (d + 0.1)

        # 3. Obstacle proximity risk
        obs_risk = 0.0
        for obs in static_obs | dynamic_obs:
            d = self._dist(cand_pos, obs)
            if d < 2.0:
                obs_risk += 1.0 / (d + 0.1)

        # 4. Movement energy cost
        energy_cost = action.step_cost

        # 5. Momentum / directional persistence
        if action == self.state.last_action and action != Action.WAIT:
            momentum = 1.0
        elif action.dx == self.state.last_action.dx and action.dy == self.state.last_action.dy:
            momentum = 0.5
        else:
            momentum = 0.0

        # 6. Exploration bonus (jitter to break symmetrical local deadlocks)
        exploration_bonus = rng.uniform(0.0, 1.0)

        # Weighted combination
        score = (
            weights.w_goal * goal_progress
            - weights.w_collision * coll_risk
            - weights.w_obstacle * obs_risk
            - weights.w_energy * energy_cost
            + weights.w_momentum * momentum
            + weights.w_exploration * exploration_bonus
        )
        return score

    @staticmethod
    def _dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
