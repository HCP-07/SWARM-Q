"""Unit tests for wireless mesh communication and decentralized consensus allocation."""

import numpy as np
import pytest
from src.agents.state import AgentState
from src.environment.targets import TaskTarget
from src.coordination.communication import LocalCommunicationMesh
from src.coordination.task_allocation import DecentralizedTaskAllocator


def test_communication_mesh_topology():
    mesh = LocalCommunicationMesh(default_comm_radius=20.0, seed=42)

    # Agent 0 at (10, 10), Agent 1 at (20, 10) [dist 10m -> connected]
    # Agent 2 at (60, 10) [dist 50m -> out of range]
    class MockAgent:
        def __init__(self, aid, pos, comm_active=True):
            self.state = AgentState(
                agent_id=aid,
                position=pos,
                velocity=np.zeros(2),
                communication_radius=20.0,
                communication_status=comm_active
            )

    agents = {
        0: MockAgent(0, np.array([10.0, 10.0])),
        1: MockAgent(1, np.array([20.0, 10.0])),
        2: MockAgent(2, np.array([60.0, 10.0]))
    }

    topology = mesh.compute_topology(agents)
    assert 1 in topology[0]
    assert 0 in topology[1]
    assert 2 not in topology[0]

    # Test dropout
    agents[1].state.communication_status = False
    topology_dropout = mesh.compute_topology(agents)
    assert 1 not in topology_dropout[0]
    assert 0 not in topology_dropout[1]


def test_decentralized_task_allocation_conflict_resolution():
    allocator = DecentralizedTaskAllocator()

    target_a = TaskTarget(task_id=1, position=np.array([20.0, 20.0]), priority=2.0)
    target_b = TaskTarget(task_id=2, position=np.array([40.0, 40.0]), priority=1.0)
    active_targets = [target_a, target_b]

    # Agent 0 is close to target_a: pos (19, 19)
    # Agent 1 is further from target_a: pos (10, 10)
    agent_0_pos = np.array([19.0, 19.0])
    agent_1_pos = np.array([10.0, 10.0])

    state_1 = AgentState(
        agent_id=1,
        position=agent_1_pos,
        velocity=np.zeros(2),
        target_id=1, # Agent 1 desires target 1
        energy=100.0
    )

    # When Agent 0 evaluates targets while listening to Agent 1's claim:
    # Since Agent 0 is much closer (higher bid), Agent 0 wins target 1
    won_target, bid = allocator.allocate_task(
        agent_id=0,
        agent_pos=agent_0_pos,
        agent_energy=100.0,
        current_target_id=None,
        active_targets=active_targets,
        neighbor_states=[state_1]
    )
    assert won_target is not None
    assert won_target.task_id == 1

    # When Agent 1 evaluates with Agent 0 claiming target 1:
    state_0 = AgentState(
        agent_id=0,
        position=agent_0_pos,
        velocity=np.zeros(2),
        target_id=1,
        energy=100.0
    )
    # Agent 1 concedes target 1 and claims target 2!
    conceded_target, _ = allocator.allocate_task(
        agent_id=1,
        agent_pos=agent_1_pos,
        agent_energy=100.0,
        current_target_id=None,
        active_targets=active_targets,
        neighbor_states=[state_0]
    )
    assert conceded_target is not None
    assert conceded_target.task_id == 2
