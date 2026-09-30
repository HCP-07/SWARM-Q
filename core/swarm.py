"""Decentralized Swarm Coordinator orchestrating local agent decision ticks and telemetry."""

from __future__ import annotations
import time
import math
import random
from typing import Dict, List, Set, Tuple, Optional, Any
from config import SimulationConfig, Action
from core.agent import AutonomousAgent, SwarmMessage
from core.environment import Environment, PerturbationType
from core.metrics import MetricsCollector, SwarmMetrics


class SwarmCoordinator:
    """Manages the headless multi-agent decentralized simulation lifecycle."""

    def __init__(self, config: SimulationConfig, env: Optional[Environment] = None):
        self.config = config
        self.env = env or Environment(config.env, seed=config.seed)
        self.rng = random.Random(config.seed)
        self.agents: Dict[int, AutonomousAgent] = {}
        self.metrics_collector = MetricsCollector(config.real_time_budget_ms)
        self.hard_collisions = 0
        self.dropout_agent_ids: Set[int] = set()
        self.current_tick = 0

    def add_agent(self, agent: AutonomousAgent) -> None:
        """Enrolls an agent into the simulation world."""
        self.agents[agent.state.agent_id] = agent

    def step(self) -> float:
        """Executes a single decentralized simulation tick.
        
        Returns:
            float: Decision latency of the tick in milliseconds.
        """
        t_start = time.perf_counter()

        # 1. Environment updates (dynamic obstacles & perturbations)
        self.env.step_environment(self.current_tick, self.agents)

        # 2. Re-index spatial buckets for fast local neighbor sensing
        self.env.update_spatial_index(self.agents)

        # 3. Decentralized Phase 1: Local Sensing & Preferred Candidate Planning
        initial_broadcast: Dict[int, SwarmMessage] = {
            aid: ag.create_broadcast_message()
            for aid, ag in self.agents.items()
            if ag.state.active
        }

        static_obs = self.env.static_obstacles
        dynamic_obs = self.env.get_dynamic_obstacle_positions()

        for aid, agent in self.agents.items():
            if not agent.state.active or agent.state.completed_target:
                continue

            # Check communication dropout
            is_dropped = aid in self.env.active_dropouts
            if is_dropped:
                self.dropout_agent_ids.add(aid)
                comm_quality = 0.1  # Severe link degradation
            else:
                comm_quality = 1.0

            # Gather neighbor messages within local communication radius
            nearby_ids = self.env.get_nearby_agent_ids(
                agent.state.position,
                agent.config.communication_radius,
                self.agents
            )

            local_msgs = [
                initial_broadcast[nid]
                for nid in nearby_ids
                if nid != aid and nid in initial_broadcast
            ]

            # Rank candidate actions locally
            agent.plan_candidates(
                neighbor_messages=local_msgs,
                static_obstacles=static_obs,
                dynamic_obstacles=dynamic_obs,
                env_cfg=self.env.config,
                communication_quality=comm_quality,
                rng=self.rng
            )

        # 4. Phase 2: Intent Broadcast & Local Conflict Resolution
        updated_broadcast: Dict[int, SwarmMessage] = {
            aid: ag.create_broadcast_message()
            for aid, ag in self.agents.items()
            if ag.state.active
        }
        all_priorities = {aid: ag.get_priority() for aid, ag in self.agents.items() if ag.state.active}

        planned_actions: Dict[int, Action] = {}
        for aid, agent in self.agents.items():
            if not agent.state.active or agent.state.completed_target:
                planned_actions[aid] = Action.WAIT
                continue

            nearby_ids = self.env.get_nearby_agent_ids(
                agent.state.position,
                agent.config.communication_radius,
                self.agents
            )
            local_msgs = [
                updated_broadcast[nid]
                for nid in nearby_ids
                if nid != aid and nid in updated_broadcast
            ]

            # Resolve conflicts locally via priority negotiation
            action = agent.resolve_and_select_action(local_msgs, all_priorities)
            planned_actions[aid] = action

        # 5. Safety Invariant Enforcement (guards against dropout blindness)
        # Verify no two agents target the same cell or perform head-on swaps
        intended_destinations: Dict[Tuple[int, int], int] = {}
        final_actions: Dict[int, Action] = {}

        # Process in descending order of dynamic priority
        sorted_agent_ids = sorted(
            [aid for aid in self.agents if self.agents[aid].state.active and not self.agents[aid].state.completed_target],
            key=lambda aid: (self.agents[aid].get_priority(), -aid),
            reverse=True
        )

        for aid in sorted_agent_ids:
            act = planned_actions.get(aid, Action.WAIT)
            ag = self.agents[aid]
            dest = (ag.state.position[0] + act.dx, ag.state.position[1] + act.dy)

            # Check destination collision
            conflict = False
            if dest in intended_destinations and dest != ag.state.position:
                conflict = True

            # Check swap collision
            if not conflict and act != Action.WAIT:
                for other_dest, other_id in intended_destinations.items():
                    if dest == self.agents[other_id].state.position and other_dest == ag.state.position:
                        conflict = True
                        break

            # Check obstacle intrusion
            if dest in static_obs or dest in dynamic_obs:
                conflict = True

            if conflict:
                # Lower priority agent yields safely
                final_actions[aid] = Action.WAIT
                intended_destinations[ag.state.position] = aid
            else:
                final_actions[aid] = act
                intended_destinations[dest] = aid

        # 6. Execute Simultaneous Movement & Verify Invariants
        occupied_next: Dict[Tuple[int, int], int] = {}
        for aid, ag in self.agents.items():
            if not ag.state.active:
                continue

            act = final_actions.get(aid, Action.WAIT)
            ag.execute_action(act)
            next_pos = ag.state.position
            self.metrics_collector.record_cell_visit(next_pos)

            # Check hard cell collision (invariant audit)
            if next_pos in occupied_next:
                self.hard_collisions += 1
            else:
                occupied_next[next_pos] = aid

            # Check obstacle intrusion (invariant audit)
            if next_pos in static_obs or next_pos in dynamic_obs:
                self.hard_collisions += 1

            # Check completion tick
            if ag.state.completed_target and ag.state.position == ag.state.target:
                if self.current_tick not in self.metrics_collector.completed_ticks:
                    self.metrics_collector.completed_ticks.append(self.current_tick)

        t_end = time.perf_counter()
        tick_latency_ms = (t_end - t_start) * 1000.0

        # Record telemetry
        active_count = sum(1 for ag in self.agents.values() if ag.state.active and not ag.state.completed_target)
        fitness_proxy = sum(
            1.0 / (1.0 + math.hypot(ag.state.position[0] - ag.state.target[0], ag.state.position[1] - ag.state.target[1]))
            for ag in self.agents.values() if ag.state.active
        )
        self.metrics_collector.record_tick(tick_latency_ms, active_count, fitness_proxy)

        self.current_tick += 1
        return tick_latency_ms

    def run(self, max_ticks: Optional[int] = None) -> SwarmMetrics:
        """Executes simulation until max_ticks or all agents complete targets."""
        ticks_to_run = max_ticks or self.config.max_ticks

        for _ in range(ticks_to_run):
            self.step()
            # Early completion condition
            if all(not ag.state.active or ag.state.completed_target for ag in self.agents.values()):
                break

        return self.metrics_collector.finalize(
            self.agents,
            self.current_tick,
            self.hard_collisions,
            self.dropout_agent_ids
        )
