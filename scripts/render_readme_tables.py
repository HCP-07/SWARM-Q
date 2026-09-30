"""Generates Markdown benchmark and ablation tables directly from results files."""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"


def render_benchmark_markdown() -> str:
    """Renders results/benchmark.csv as a clean Markdown table."""
    bench_csv = RESULTS_DIR / "benchmark.csv"
    if not bench_csv.exists():
        return "_No benchmark results generated yet._"

    df = pd.read_csv(bench_csv)
    lines = [
        "| Metric | Greedy Nearest | Classical PSO | Random Local Search | **Proposed AD-QPSO** |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]

    solver_map = {row["Solver"]: row for _, row in df.iterrows()}
    greedy = solver_map.get("Greedy Nearest", {})
    pso = solver_map.get("Classical PSO", solver_map.get("Classical PSO (BASELINE)", {}))
    rand = solver_map.get("Random Local Search", {})
    adqpso = solver_map.get("Proposed ADSO", solver_map.get("Proposed AD-QPSO", {}))

    def fmt(d, key, is_pct=False, is_ms=False, digits=2):
        val = d.get(key, 0.0)
        try:
            num = float(val)
            s = f"{num:.{digits}f}"
            if is_pct: s += "%"
            if is_ms: s += " ms"
            return s
        except Exception:
            return str(val)

    lines.append(f"| **Task Completion Rate (%)** | {fmt(greedy, 'task_completion_rate', is_pct=True)} | {fmt(pso, 'task_completion_rate', is_pct=True)} | {fmt(rand, 'task_completion_rate', is_pct=True)} | **{fmt(adqpso, 'task_completion_rate', is_pct=True)}** |")
    lines.append(f"| **Swarm Throughput (/min)** | {fmt(greedy, 'swarm_throughput')} | {fmt(pso, 'swarm_throughput')} | {fmt(rand, 'swarm_throughput')} | **{fmt(adqpso, 'swarm_throughput')}** |")
    lines.append(f"| **Coverage Velocity (m/tick)** | {fmt(greedy, 'coverage_velocity')} | {fmt(pso, 'coverage_velocity')} | {fmt(rand, 'coverage_velocity')} | **{fmt(adqpso, 'coverage_velocity')}** |")
    lines.append(f"| **Total Path Length (m)** | {fmt(greedy, 'total_path_length', digits=1)} | {fmt(pso, 'total_path_length', digits=1)} | {fmt(rand, 'total_path_length', digits=1)} | **{fmt(adqpso, 'total_path_length', digits=1)}** |")
    lines.append(f"| **Hard Collision Count** | **{int(greedy.get('collision_count', 0))}** | **{int(pso.get('collision_count', 0))}** | **{int(rand.get('collision_count', 0))}** | **{int(adqpso.get('collision_count', 0))}** |")
    lines.append(f"| **Near-Collision Count** | {int(greedy.get('near_collision_count', 0))} | {int(pso.get('near_collision_count', 0))} | {int(rand.get('near_collision_count', 0))} | {int(adqpso.get('near_collision_count', 0))} |")
    lines.append(f"| **Mean Tick Latency (ms)** | {fmt(greedy, 'average_tick_latency_ms', is_ms=True)} | {fmt(pso, 'average_tick_latency_ms', is_ms=True)} | {fmt(rand, 'average_tick_latency_ms', is_ms=True)} | **{fmt(adqpso, 'average_tick_latency_ms', is_ms=True)}** |")
    lines.append(f"| **P95 Tick Latency (ms)** | {fmt(greedy, 'p95_latency_ms', is_ms=True)} | {fmt(pso, 'p95_latency_ms', is_ms=True)} | {fmt(rand, 'p95_latency_ms', is_ms=True)} | {fmt(adqpso, 'p95_latency_ms', is_ms=True)} |")
    lines.append(f"| **Final Objective Value** | {fmt(greedy, 'final_objective_value', digits=4)} | {fmt(pso, 'final_objective_value', digits=4)} | {fmt(rand, 'final_objective_value', digits=4)} | **{fmt(adqpso, 'final_objective_value', digits=4)}** |")

    return "\n".join(lines)


def render_ablation_markdown() -> str:
    """Renders results/ablation.csv as a clean Markdown table."""
    ablation_csv = RESULTS_DIR / "ablation.csv"
    if not ablation_csv.exists():
        return "_No ablation results generated yet._"

    df = pd.read_csv(ablation_csv)
    lines = [
        "| Stage | Architecture Configuration | Completion Rate (%) | Swarm Throughput | Mean Latency (ms) | P95 Latency (ms) | Energy (units) | Collisions |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for _, row in df.iterrows():
        stage_str = str(row.get("Stage", ""))
        stage_code = stage_str.split(":")[0].strip() if ":" in stage_str else stage_str[:1]
        lines.append(
            f"| **{stage_code}** | {stage_str} | {float(row.get('task_completion_rate', 0.0)):.2f}% | "
            f"{float(row.get('swarm_throughput', 0.0)):.2f} | {float(row.get('average_tick_latency_ms', 0.0)):.2f} ms | "
            f"{float(row.get('p95_latency_ms', 0.0)):.2f} ms | {float(row.get('total_energy_consumed', 0.0)):.1f} | "
            f"**{int(row.get('collision_count', 0))}** |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print("### Benchmark Table")
    print(render_benchmark_markdown())
    print("\n### Ablation Table")
    print(render_ablation_markdown())
