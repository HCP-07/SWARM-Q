"""Autonomous swarm agent integrating decentralized perception, allocation, optimization, and execution."""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from src.agents.state import AgentState
from src.agents.kinematics import KinematicModel
from src.environment.obstacles import Obstacle
from src.environment.targets import TaskTarget
from src.safety.validator import SafetyValidator
from src.safety.repair import TrajectoryRepair
from src.optimization.qpso_engine import QPSOTrajectoryEngine
from src.optimization.fitness import FitnessBreakdown
from src.coordination.task_allocation import DecentralizedTaskAllocator
from configs.config import AgentPhysicalConfig


class AutonomousAgent:
    """An autonomous, decentralized swarm agent with local perception and decision-making."""

    def __init__(
        self,
        agent_id: int,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        agent_config: AgentPhysicalConfig,
        kinematics: KinematicModel,
        optimizer: QPSOTrajectoryEngine,
        safety_validator: SafetyValidator,
        trajectory_repair: TrajectoryRepair,
        task_allocator: DecentralizedTaskAllocator,
        avoidance_margin: float = 3.2
    ):
        self.agent_id = agent_id
        self.config = agent_config
        self.kinematics = kinematics
        self.optimizer = optimizer
        self.safety_validator = safety_validator
        self.repair = trajectory_repair
        self.allocator = task_allocator
        self.avoidance_margin = avoidance_margin

        self.state = AgentState(
            agent_id=agent_id,
            position=np.array(initial_position, dtype=np.float64),
            velocity=np.array(initial_velocity, dtype=np.float64),
            energy=agent_config.initial_energy,
            safety_radius=agent_config.safety_radius,
            sensor_radius=agent_config.sensor_radius,
            communication_radius=agent_config.communication_radius,
            history_path=[np.array(initial_position, dtype=np.float64)]
        )
        self.last_fitness_breakdown: Optional[FitnessBreakdown] = None
        self.replan_count: int = 0

    def sense_local_environment(
        self,
        all_obstacles: List[Obstacle],
        all_targets: List[TaskTarget]
    ) -> Tuple[List[Obstacle], List[TaskTarget]]:
        """Filters global world entities to those strictly within sensor horizon."""
        local_obstacles = [
            obs for obs in all_obstacles
            if obs.distance_to(self.state.position) <= self.state.sensor_radius
        ]
        local_targets = [
            t for t in all_targets
            if not t.completed and float(np.linalg.norm(self.state.position - t.position)) <= self.state.sensor_radius * 2.5
        ]
        return local_obstacles, local_targets

    def plan_step(
        self,
        local_obstacles: List[Obstacle],
        local_targets: List[TaskTarget],
        neighbor_states: List[AgentState],
        other_agent_trajectories: Dict[int, List[np.ndarray]],
        dt: float,
        force_replan: bool = False
    ) -> bool:
        """Decentralized task allocation and trajectory synthesis."""
        if not self.state.active:
            return False

        # 1. Decentralized Task Negotiation
        chosen_target, _ = self.allocator.allocate_task(
            self.agent_id,
            self.state.position,
            self.state.energy,
            self.state.target_id,
            local_targets,
            neighbor_states
        )

        if chosen_target is not None:
            self.state.target_id = chosen_target.task_id
            self.state.target_position = chosen_target.position.copy()
        else:
            self.state.target_id = None
            self.state.target_position = None

        # 2. Check if replan needed
        needs_replan = (
            force_replan or
            len(self.state.current_trajectory) < 2 or
            self.state.local_best is None
        )

        if not needs_replan:
            return False

        # 3. Trajectory Optimization via Decentralized QPSO Engine
        new_traj, breakdown = self.optimizer.optimize_trajectory(
            current_pos=self.state.position,
            current_vel=self.state.velocity,
            target_pos=self.state.target_position,
            obstacles=local_obstacles,
            other_agent_trajectories=other_agent_trajectories,
            neighbor_states=neighbor_states,
            dt=dt
        )

        self.state.current_trajectory = new_traj
        self.state.local_best = new_traj
        self.state.local_best_fitness = breakdown.total_fitness
        self.last_fitness_breakdown = breakdown
        self.replan_count += 1
        return True

    def execute_motion_step(
        self,
        dt: float,
        neighbor_positions: Optional[List[np.ndarray]] = None,
        obstacles: Optional[List[Obstacle]] = None
    ) -> Tuple[np.ndarray, float]:
        """Executes the next step along current trajectory with reactive safety filtering."""
        if not self.state.active:
            return self.state.position, 0.0

        if len(self.state.current_trajectory) >= 2:
            # Pop next planned waypoint
            next_wp = self.state.current_trajectory[1]
            diff = next_wp - self.state.position
            dist = np.linalg.norm(diff)
            if dist > 1e-4:
                desired_vel = (diff / dist) * min(self.config.max_speed, dist / dt)
            else:
                desired_vel = np.zeros(2, dtype=np.float64)

            # Advance trajectory buffer
            self.state.current_trajectory.pop(0)
        else:
            desired_vel = np.zeros(2, dtype=np.float64)

        # Control Barrier Function filter for strictly guaranteed safety invariants
        from src.safety.filter import ReactiveSafetyFilter
        desired_vel = ReactiveSafetyFilter.filter_velocity(
            position=self.state.position,
            desired_velocity=desired_vel,
            neighbor_positions=neighbor_positions or [],
            obstacles=obstacles or [],
            dt=dt,
            min_agent_sep=self.state.safety_radius * 2.0,
            min_obstacle_dist=1.5,
            max_speed=self.config.max_speed,
            avoidance_margin=self.avoidance_margin
        )

        # Step kinematics
        new_pos, new_vel, energy_spent = self.kinematics.step(
            self.state.position, self.state.velocity, desired_vel
        )

        # Hard safety invariant preservation: multi-body contact relaxation
        min_sep = self.state.safety_radius * 2.0
        if neighbor_positions:
            for _ in range(3):
                for n_pos in neighbor_positions:
                    diff = new_pos - n_pos
                    d = float(np.linalg.norm(diff))
                    if d < min_sep:
                        n_dir = diff / max(d, 1e-4) if d > 1e-4 else np.array([0.0, 1.0])
                        new_pos = n_pos + n_dir * (min_sep + 0.05)

        if obstacles:
            for obs in obstacles:
                d = obs.distance_to(new_pos)
                if d < 1.5:
                    rep = obs.repulsive_vector(new_pos, 1.5)
                    new_pos = new_pos + rep * (1.5 - d + 0.05)

        self.state.position = new_pos
        self.state.velocity = new_vel
        self.state.energy = max(0.0, self.state.energy - energy_spent)
        self.state.history_path.append(new_pos.copy())

        # If energy depleted, agent fails
        if self.state.energy <= 0.0:
            self.state.active = False
            self.state.velocity = np.zeros(2, dtype=np.float64)

        return new_pos, energy_spent
