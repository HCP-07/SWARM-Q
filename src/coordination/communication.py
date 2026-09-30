"""Local wireless mesh communication model with range limits and packet dropout."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import numpy as np
from src.agents.state import AgentState


@dataclass
class NetworkPacket:
    """Standardized message packet exchanged over the wireless mesh."""
    sender_id: int
    timestamp_tick: int
    position: np.ndarray
    velocity: np.ndarray
    claimed_target_id: Optional[int]
    task_bid: float
    current_trajectory: List[np.ndarray]
    local_best_fitness: float
    local_best_trajectory: Optional[List[np.ndarray]]


class LocalCommunicationMesh:
    """Decentralized local mesh model with distance-limited connectivity and dropouts."""

    def __init__(self, default_comm_radius: float = 25.0, packet_loss_prob: float = 0.0, seed: int = 42):
        self.default_comm_radius = default_comm_radius
        self.packet_loss_prob = packet_loss_prob
        self.rng = np.random.default_rng(seed)
        self.active_links: List[Tuple[int, int]] = []
        self.total_packets_sent: int = 0
        self.total_packets_dropped: int = 0

    def compute_topology(self, agents_map: Dict[int, Any]) -> Dict[int, List[int]]:
        """Identifies active neighborhood graph based on physical proximity and comm status."""
        neighbors_map: Dict[int, List[int]] = {aid: [] for aid in agents_map.keys()}
        self.active_links.clear()

        agent_ids = list(agents_map.keys())
        n = len(agent_ids)

        for i in range(n):
            id_i = agent_ids[i]
            ag_i = agents_map[id_i]
            if not ag_i.state.active or not ag_i.state.communication_status:
                continue

            r_i = getattr(ag_i.state, "communication_radius", self.default_comm_radius)
            pos_i = ag_i.state.position

            for j in range(i + 1, n):
                id_j = agent_ids[j]
                ag_j = agents_map[id_j]
                if not ag_j.state.active or not ag_j.state.communication_status:
                    continue

                r_j = getattr(ag_j.state, "communication_radius", self.default_comm_radius)
                effective_radius = min(r_i, r_j)
                dist = float(np.linalg.norm(pos_i - ag_j.state.position))

                if dist <= effective_radius:
                    neighbors_map[id_i].append(id_j)
                    neighbors_map[id_j].append(id_i)
                    self.active_links.append((id_i, id_j))

        return neighbors_map

    def exchange_states(
        self,
        agents_map: Dict[int, Any],
        current_tick: int
    ) -> Dict[int, List[AgentState]]:
        """Transmits state copies across local communication links with simulated loss."""
        neighbors_map = self.compute_topology(agents_map)
        received_states: Dict[int, List[AgentState]] = {aid: [] for aid in agents_map.keys()}

        for sender_id, neighbor_ids in neighbors_map.items():
            sender_agent = agents_map[sender_id]
            if not sender_agent.state.active or not sender_agent.state.communication_status:
                continue

            # Package state
            state_snapshot = sender_agent.state.copy()

            for recv_id in neighbor_ids:
                self.total_packets_sent += 1
                if self.packet_loss_prob > 0.0 and self.rng.uniform(0.0, 1.0) < self.packet_loss_prob:
                    self.total_packets_dropped += 1
                    continue
                received_states[recv_id].append(state_snapshot)

        return received_states
