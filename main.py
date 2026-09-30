"""Command-line interface and execution harness for Adaptive Decentralized QPSO (AD-QPSO)."""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from configs.config import SimulationConfig, LatencyBudgetConfig, AgentPhysicalConfig
from scenarios.scenario_definitions import ScenarioBuilder, ScenarioType
from src.simulation.engine import SimulationEngine
from src.benchmarking.suite import BenchmarkSuite
from src.benchmarking.ablation import AblationStudy
from src.benchmarking.baselines import BaselineRunner
from src.visualization.renderer import SwarmVisualizer
from src.visualization.charts import ChartGenerator


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(os.getenv(name, default))
    except (ValueError, TypeError):
        return default
    return v if lo <= v <= hi else default


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(os.getenv(name, default))
    except (ValueError, TypeError):
        return default
    return v if lo <= v <= hi else default


LATENCY_BUDGET_MS = _env_float("ADSO_DEFAULT_LATENCY_BUDGET_MS", 25.0, 1.0, 5000.0)
DEFAULT_SEED = _env_int("ADSO_DEFAULT_SEED", 42, 0, 2**31 - 1)
LOG_LEVEL = os.getenv("ADSO_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("AD-QPSO")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive Decentralized QPSO (AD-QPSO) for Multi-Agent Autonomous Coordination"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="stress",
        choices=[s.value for s in ScenarioType],
        help="Perturbation scenario to execute (default: stress)"
    )
    parser.add_argument("--agents", type=int, default=15, help="Number of autonomous agents in swarm")
    parser.add_argument("--iterations", type=int, default=120, help="Total simulation ticks to execute")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic random seed")
    parser.add_argument("--latency-budget", type=float, default=LATENCY_BUDGET_MS, help="Real-time latency budget in ms")
    parser.add_argument("--communication-radius", type=float, default=25.0, help="Wireless mesh range in meters")
    parser.add_argument("--benchmark", action="store_true", help="Execute comparative multi-baseline evaluation")
    parser.add_argument("--ablation", action="store_true", help="Execute 6-stage component ablation study")
    parser.add_argument("--visualize", action="store_true", default=True, help="Render 2D maps and analytical charts")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory where artifacts are saved")

    return parser.parse_args()


def export_trajectory_csv(engine: SimulationEngine, file_path: str) -> None:
    """Exports agent spatial traces at every tick to CSV."""
    rows = []
    for aid, ag in engine.agents.items():
        for tick, pos in enumerate(ag.state.history_path):
            rows.append({
                "agent_id": aid,
                "tick": tick,
                "pos_x": round(float(pos[0]), 3),
                "pos_y": round(float(pos[1]), 3),
                "active": ag.state.active
            })
    pd.DataFrame(rows).to_csv(file_path, index=False)


def export_convergence_csv(convergence_history: List[float], file_path: str) -> None:
    """Exports objective fitness progression to CSV."""
    df = pd.DataFrame({
        "tick": list(range(len(convergence_history))),
        "fitness": [round(f, 4) for f in convergence_history]
    })
    df.to_csv(file_path, index=False)


def export_latency_csv(latencies: List[float], file_path: str) -> None:
    """Exports tick-by-tick latency recordings to CSV."""
    df = pd.DataFrame({
        "tick": list(range(len(latencies))),
        "latency_ms": [round(l, 4) for l in latencies]
    })
    df.to_csv(file_path, index=False)


def main() -> int:
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("ADAPTIVE DECENTRALIZED QPSO (AD-QPSO) SWARM ENGINE")
    print("=" * 70)
    print(f"Scenario:             {args.scenario}")
    print(f"Agent Population:     {args.agents}")
    print(f"Simulation Ticks:     {args.iterations}")
    print(f"Deterministic Seed:   {args.seed}")
    print(f"Latency Budget:       {args.latency_budget} ms")
    print(f"Communication Radius: {args.communication_radius} m")
    print(f"Output Directory:     {args.output_dir}")
    print("=" * 70)

    # Master configuration
    config = SimulationConfig(
        seed=args.seed,
        agent_count=args.agents,
        agent_params=AgentPhysicalConfig(
            communication_radius=args.communication_radius
        ),
        latency_budget=LatencyBudgetConfig(
            budget_ms=args.latency_budget
        )
    )
    config.arena.max_ticks = args.iterations

    # 1. Main Simulation Execution (Proposed ADSO)
    print("\n[1/4] Initializing and running proposed swarm optimization engine...")
    engine = ScenarioBuilder.build(args.scenario, config)
    metrics = engine.run(args.iterations)

    print(f"  -> Task Completion Rate: {metrics.task_completion_rate:.1f}%")
    print(f"  -> Total Path Length:    {metrics.total_path_length:.2f} m")
    print(f"  -> Energy Consumed:      {metrics.total_energy_consumed:.2f} units")
    print(f"  -> Hard Collisions:      {metrics.collision_count}")
    print(f"  -> Near Collisions:      {metrics.near_collision_count}")
    print(f"  -> Mean Tick Latency:    {metrics.average_tick_latency_ms:.3f} ms")
    print(f"  -> P95 Tick Latency:     {metrics.p95_latency_ms:.3f} ms (Budget: {args.latency_budget} ms)")
    print(f"  -> Real-Time Status:     {'PASS' if metrics.real_time_pass else 'FAIL'}")

    # 2. Benchmark Multi-Baseline Comparison
    benchmark_file = os.path.join(args.output_dir, "benchmark.csv")
    print(f"\n[2/4] Executing comparative benchmarks across all baselines...")
    bench_df = BenchmarkSuite.run_comparison(args.scenario, config)
    bench_df.to_csv(benchmark_file, index=False)
    print(f"  -> Saved benchmark table: {benchmark_file}")
    print(bench_df[["Solver", "task_completion_rate", "collision_count", "p95_latency_ms", "real_time_pass"]].to_string(index=False))

    # 3. Ablation Study
    if args.ablation:
        print(f"\n[3/4] Running 6-stage component ablation study...")
        ablation_file = os.path.join(args.output_dir, "ablation.csv")
        ablation_df = AblationStudy.run_all(config, args.scenario)
        ablation_df.to_csv(ablation_file, index=False)
        print(f"  -> Saved ablation study: {ablation_file}")

    # 4. Save Core Data Artifacts
    metrics_json_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    summary_json_path = os.path.join(args.output_dir, "summary.json")
    summary_data = {
        "scenario": args.scenario,
        "agents": args.agents,
        "ticks": args.iterations,
        "seed": args.seed,
        "real_time_sla_met": bool(metrics.real_time_pass),
        "hard_safety_preserved": bool(metrics.collision_count == 0),
        "task_completion_pct": round(metrics.task_completion_rate, 2),
        "mean_latency_ms": round(metrics.average_tick_latency_ms, 3),
        "p95_latency_ms": round(metrics.p95_latency_ms, 3),
        "p99_latency_ms": round(metrics.p99_latency_ms, 3),
        "max_latency_ms": round(metrics.max_latency_ms, 3),
        "total_energy": round(metrics.total_energy_consumed, 2)
    }
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    convergence_csv_path = os.path.join(args.output_dir, "convergence.csv")
    export_convergence_csv(engine.convergence_fitness_history, convergence_csv_path)

    latency_csv_path = os.path.join(args.output_dir, "latency.csv")
    export_latency_csv(engine.latency_tracker.tick_latencies_ms, latency_csv_path)

    trajectory_csv_path = os.path.join(args.output_dir, "trajectory.csv")
    export_trajectory_csv(engine, trajectory_csv_path)

    # 5. Visualizations & Analytical Charts
    if args.visualize:
        print("\n[4/4] Rendering 2D spatial map and publication-grade analytical charts...")
        vis = SwarmVisualizer()
        map_path = os.path.join(args.output_dir, "trajectory_map.png")
        vis.render_map(engine, map_path)
        print(f"  -> Generated: {map_path}")

        chart_gen = ChartGenerator()

        # Convergence chart
        conv_series = {"Proposed AD-QPSO": engine.convergence_fitness_history}
        # Run brief baselines for convergence curve comparison
        _, eng_pso = BaselineRunner.run_classical_pso(args.scenario, config)
        conv_series["Classical PSO"] = eng_pso.convergence_fitness_history
        _, eng_greedy = BaselineRunner.run_greedy_nearest(args.scenario, config)
        conv_series["Greedy Nearest"] = eng_greedy.convergence_fitness_history
        _, eng_rand = BaselineRunner.run_random_local_search(args.scenario, config)
        conv_series["Random Local Search"] = eng_rand.convergence_fitness_history

        chart_gen.generate_convergence_chart(conv_series, os.path.join(args.output_dir, "convergence_graph.png"))

        # Latency chart
        chart_gen.generate_latency_chart(
            engine.latency_tracker.tick_latencies_ms,
            budget_ms=args.latency_budget,
            output_path=os.path.join(args.output_dir, "latency_graph.png")
        )

        # Scalability chart (benchmarking 10, 25, 50, 100 agents)
        scale_agents = [10, 25, 50, 100]
        scale_lats = []
        scale_thrus = []
        print("  -> Benchmarking scalability profile across [10, 25, 50, 100] agents...")
        for pop in scale_agents:
            s_cfg = SimulationConfig(seed=args.seed, agent_count=pop)
            s_cfg.arena.max_ticks = 40
            s_engine = ScenarioBuilder.build("static", s_cfg)
            s_m = s_engine.run(40)
            scale_lats.append(s_m.p95_latency_ms)
            scale_thrus.append(s_m.swarm_throughput)

        chart_gen.generate_scalability_chart(
            {"agents": scale_agents, "p95_latency": scale_lats, "throughput": scale_thrus},
            output_path=os.path.join(args.output_dir, "scalability_graph.png")
        )

        # Collision safety chart
        collision_stats = {}
        for row in bench_df.to_dict(orient="records"):
            collision_stats[row["Solver"]] = {
                "hard": int(row["collision_count"]),
                "near": int(row["near_collision_count"])
            }
        chart_gen.generate_collision_safety_chart(
            collision_stats,
            output_path=os.path.join(args.output_dir, "collision_safety_graph.png")
        )

        # Throughput chart
        thru_series = {
            "Proposed ADSO": [int(i * (metrics.task_completion_rate / 100.0 * len(engine.env.targets) / args.iterations)) for i in range(args.iterations)]
        }
        for row in bench_df.to_dict(orient="records"):
            rate = row["task_completion_rate"]
            thru_series[row["Solver"]] = [int(i * (rate / 100.0 * len(engine.env.targets) / args.iterations)) for i in range(args.iterations)]
        chart_gen.generate_throughput_chart(
            thru_series,
            output_path=os.path.join(args.output_dir, "throughput_graph.png")
        )

        # Robustness chart across all scenarios
        print("  -> Evaluating robustness profile across perturbation scenarios...")
        rob_data = {}
        for sc in ScenarioType:
            r_cfg = SimulationConfig(seed=args.seed, agent_count=min(args.agents, 12))
            r_cfg.arena.max_ticks = 60
            r_eng = ScenarioBuilder.build(sc.value, r_cfg)
            r_m = r_eng.run(60)
            rob_data[sc.value] = r_m.task_completion_rate

        chart_gen.generate_robustness_chart(
            rob_data,
            output_path=os.path.join(args.output_dir, "robustness_graph.png")
        )
        print("  -> All charts successfully generated in results/ directory.")

    ok = bool(metrics.real_time_pass and metrics.collision_count == 0)
    print("\n" + "=" * 70)
    if ok:
        print("ALL CONSTRAINTS AND REAL-TIME SLA CHECKS PASSED")
    else:
        print(f"CHECKS EVALUATED: real_time_sla={'PASS' if metrics.real_time_pass else 'FAIL'} (p95={metrics.p95_latency_ms:.1f}ms, budget={args.latency_budget:.1f}ms) hard_collisions={metrics.collision_count}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
