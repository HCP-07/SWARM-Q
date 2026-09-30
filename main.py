"""Command-line interface and execution harness for Decentralized Adaptive Swarm Navigation.

Intelligent Systems & Autonomous Computing: Swarm Intelligence + Multi-Agent Heuristics
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from typing import Dict, Any
from config import SimulationConfig, ScenarioType
from scenarios import get_scenario
from algorithms.adaptive_swarm import AdaptiveSwarmAgent
from experiments.benchmark import run_benchmark
from experiments.ablation import run_ablation
from experiments.statistics import run_statistics
from experiments.visualize import generate_all_plots, plot_trajectories, plot_convergence, plot_latency


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive Decentralized Swarm Intelligence for Real-Time Multi-Agent Navigation"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="obstacles",
        choices=["open", "obstacles", "dense", "dynamic", "dropout", "failure", "combined"],
        help="Benchmark scenario to execute (default: obstacles)"
    )
    parser.add_argument("--agents", type=int, default=15, help="Number of autonomous agents in swarm")
    parser.add_argument("--ticks", type=int, default=80, help="Maximum simulation ticks to execute")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--budget", type=float, default=100.0, help="Real-time SLA decision latency budget in ms")
    parser.add_argument("--benchmark", action="store_true", help="Execute comparative multi-baseline evaluation across scenarios")
    parser.add_argument("--ablation", action="store_true", help="Execute 5-stage component ablation study")
    parser.add_argument("--stats", action="store_true", help="Execute multi-seed statistical significance evaluation")
    parser.add_argument("--runs", type=int, default=30, help="Number of repeated runs for statistical evaluation")
    parser.add_argument("--visualize", action="store_true", help="Generate publication-quality diagnostic plots")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory where artifacts are saved")

    return parser.parse_args()


def print_banner(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("DECENTRALIZED ADAPTIVE SWARM NAVIGATION ENGINE")
    print("Intelligent Systems & Autonomous Computing | Swarm Intelligence + Multi-Agent Heuristics")
    print("=" * 80)
    print(f"Scenario:             {args.scenario}")
    print(f"Swarm Size:           {args.agents} agents")
    print(f"Max Ticks:            {args.ticks}")
    print(f"Deterministic Seed:   {args.seed}")
    print(f"Real-Time Budget:     {args.budget:.1f} ms")
    print(f"Output Directory:     {args.output_dir}")
    print("=" * 80)


def main() -> int:
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if args.benchmark:
        run_benchmark(seed=args.seed, agent_count=args.agents, max_ticks=args.ticks, output_dir=args.output_dir)
        return 0

    if args.ablation:
        run_ablation(seed=args.seed, agent_count=args.agents, max_ticks=args.ticks, output_dir=args.output_dir)
        return 0

    if args.stats:
        run_statistics(runs=args.runs, agent_count=args.agents, max_ticks=args.ticks, scenario=ScenarioType(args.scenario), output_dir=args.output_dir)
        return 0

    if args.visualize:
        print("[VISUALIZATION] Generating diagnostic plots into", plots_dir)
        generate_all_plots(output_dir=plots_dir)
        return 0

    # Default: Run the selected scenario simulation
    print_banner(args)
    cfg = SimulationConfig(
        seed=args.seed,
        agent_count=args.agents,
        max_ticks=args.ticks,
        real_time_budget_ms=args.budget,
        scenario=ScenarioType(args.scenario),
        enable_adaptation=True
    )

    print("\n[1/3] Initializing decentralized swarm simulation...")
    coordinator = get_scenario(args.scenario, config=cfg, agent_cls=AdaptiveSwarmAgent)

    print(f"[2/3] Executing simulation across {args.ticks} ticks...")
    metrics = coordinator.run()

    print("\n[3/3] Simulation Execution Summary:")
    print("-" * 50)
    print(f"Task Completion Rate:       {metrics.task_completion_rate:>6.2f}%")
    print(f"Hard Collisions:             {metrics.collision_count:>6} (Zero-Tolerance: {'PASS' if metrics.collision_count == 0 else 'FAIL'})")
    print(f"Average Decision Latency:    {metrics.average_decision_latency_ms:>6.3f} ms")
    print(f"P95 Decision Latency:        {metrics.p95_decision_latency_ms:>6.3f} ms (Budget: {args.budget:.1f} ms)")
    print(f"Real-Time SLA Status:        {'PASS' if metrics.real_time_pass else 'FAIL'}")
    print(f"Deadlocks Encountered:       {metrics.deadlock_count:>6}")
    print(f"Deadlock Resolution Rate:    {metrics.deadlock_resolution_rate:>6.2f}%")
    print(f"Average Path Length:         {metrics.average_path_length:>6.2f} units")
    print(f"Total Energy Expended:       {metrics.total_energy_cost:>6.2f} units")
    print(f"Collective Swarm Objective:  {metrics.final_objective_value:>6.4f}")
    print("-" * 50)

    # Save summary artifacts
    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"Metrics saved to {summary_path}")

    # Generate plots for this run
    traj_path = os.path.join(plots_dir, "trajectories.png")
    conv_path = os.path.join(plots_dir, "convergence.png")
    lat_path = os.path.join(plots_dir, "latency.png")

    plot_trajectories(coordinator, traj_path)
    plot_convergence(coordinator.metrics_collector.fitness_history, conv_path)
    plot_latency(coordinator.metrics_collector.tick_latencies_ms, args.budget, lat_path)

    ok = bool(metrics.real_time_pass and metrics.collision_count == 0)
    print("\n" + "=" * 80)
    print("STATUS:", "ALL SAFETY CONSTRAINTS AND REAL-TIME SLA CHECKS PASSED" if ok else "CONSTRAINTS VIOLATED")
    print("=" * 80)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
