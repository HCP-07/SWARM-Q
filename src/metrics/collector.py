"""Comprehensive metrics collection and evaluation suite."""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np
from src.metrics.latency import LatencyReport


@dataclass
class SwarmMetrics:
    """Comprehensive evaluation metrics for multi-agent swarm simulation."""
    task_completion_rate: float        # Percentage of total tasks completed (0 - 100%)
    swarm_throughput: float            # Completed tasks per 100 ticks
    coverage_velocity: float           # Spatial grid cells traversed per tick
    total_path_length: float           # Cumulative meters traveled by all agents
    average_path_length: float         # Average meters traveled per agent
    total_energy_consumed: float       # Cumulative energy units expended
    collision_count: int               # Hard constraint safety violations (0 = collision-free)
    near_collision_count: int          # Near-proximity occurrences within safety margin
    deadlock_count: int                # Agents stalled without progress
    average_tick_latency_ms: float     # Mean tick execution time
    p95_latency_ms: float              # 95th percentile tick latency
    p99_latency_ms: float              # 99th percentile tick latency
    max_latency_ms: float              # Worst-case tick latency
    real_time_pass: bool               # Meets real-time SLA budget
    convergence_iterations: int        # Ticks to satisfy mission criteria
    recovery_time_ticks: float         # Mean ticks to stabilize after dynamic event
    communication_robustness_score: float # Resilience index during packet/link failures
    agent_failure_recovery_rate: float # Reallocation success for orphan tasks
    final_objective_value: float       # Mean normalized fitness score

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """Aggregates tick telemetry and computes full research metrics."""

    def __init__(self, arena_bounds: tuple[float, float, float, float]):
        self.arena_bounds = arena_bounds
        self.grid_cell_size = 5.0
        self.visited_grid_cells: set[tuple[int, int]] = set()

    def compute_metrics(
        self,
        agents_map: Dict[int, Any],
        targets: List[Any],
        total_ticks: int,
        latency_report: LatencyReport,
        hard_collisions: int,
        near_collisions: int,
        recovery_times: List[int],
        orphan_tasks_reallocated: int,
        total_orphan_tasks: int,
        final_fitness_values: List[float]
    ) -> SwarmMetrics:
        # 1. Task metrics
        total_tasks = len(targets)
        completed_tasks = sum(1 for t in targets if t.completed)
        completion_rate = (completed_tasks / max(total_tasks, 1)) * 100.0
        throughput = (completed_tasks / max(total_ticks, 1)) * 100.0

        # 2. Path lengths and spatial coverage
        total_dist = 0.0
        active_agents = len(agents_map)
        deadlocks = 0

        for ag in agents_map.values():
            hist = ag.state.history_path
            agent_dist = 0.0
            for i in range(len(hist) - 1):
                p1, p2 = hist[i], hist[i + 1]
                step_dist = float(np.linalg.norm(p2 - p1))
                agent_dist += step_dist

                # Mark grid coverage
                cell_x = int(p2[0] // self.grid_cell_size)
                cell_y = int(p2[1] // self.grid_cell_size)
                self.visited_grid_cells.add((cell_x, cell_y))

            total_dist += agent_dist

            # Deadlock detection: if agent active with target, but moved < 0.2m over last 20 ticks
            if ag.state.active and ag.state.target_id is not None and len(hist) > 20:
                recent_disp = float(np.linalg.norm(hist[-1] - hist[-20]))
                if recent_disp < 0.2:
                    deadlocks += 1

        avg_path = total_dist / max(active_agents, 1)
        coverage_rate = len(self.visited_grid_cells) / max(total_ticks, 1)

        # 3. Energy consumption
        total_energy = sum(
            (ag.config.initial_energy - ag.state.energy) for ag in agents_map.values()
        )

        # 4. Convergence iterations (tick when 80% tasks reached or last task completed)
        completion_ticks = [t.completed_at_tick for t in targets if t.completed and t.completed_at_tick is not None]
        if completion_ticks:
            sorted_ticks = sorted(completion_ticks)
            idx_80 = int(len(sorted_ticks) * 0.8)
            convergence_tick = sorted_ticks[min(idx_80, len(sorted_ticks) - 1)]
        else:
            convergence_tick = total_ticks

        # 5. Recovery metrics (ratio in [0.0, 1.0])
        mean_recovery = float(np.mean(recovery_times)) if recovery_times else 0.0
        if total_orphan_tasks > 0:
            failure_recovery_rate = float(np.clip(orphan_tasks_reallocated / total_orphan_tasks, 0.0, 1.0))
        else:
            failure_recovery_rate = 1.0

        # 6. Communication robustness (ratio of communication availability and throughput)
        total_agent_ticks = sum(len(ag.state.history_path) for ag in agents_map.values())
        connected_ticks = sum(
            len(ag.state.history_path) for ag in agents_map.values() if ag.state.communication_status
        )
        comm_ratio = connected_ticks / max(total_agent_ticks, 1)
        comm_robustness = float(np.clip(comm_ratio * (completion_rate / 100.0 + 0.5), 0.0, 1.0))

        # 7. Final objective fitness
        final_obj = float(np.mean(final_fitness_values)) if final_fitness_values else 0.0

        return SwarmMetrics(
            task_completion_rate=float(completion_rate),
            swarm_throughput=float(throughput),
            coverage_velocity=float(coverage_rate),
            total_path_length=float(total_dist),
            average_path_length=float(avg_path),
            total_energy_consumed=float(total_energy),
            collision_count=hard_collisions,
            near_collision_count=near_collisions,
            deadlock_count=deadlocks,
            average_tick_latency_ms=latency_report.mean_ms,
            p95_latency_ms=latency_report.p95_ms,
            p99_latency_ms=latency_report.p99_ms,
            max_latency_ms=latency_report.max_ms,
            real_time_pass=latency_report.passed,
            convergence_iterations=int(convergence_tick),
            recovery_time_ticks=float(mean_recovery),
            communication_robustness_score=float(comm_robustness),
            agent_failure_recovery_rate=float(failure_recovery_rate),
            final_objective_value=float(final_obj)
        )
