"""Generates deterministic cProfile call-tree statistics for results/profile.txt."""

import cProfile
import pstats
from io import StringIO
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from configs.config import SimulationConfig
from scenarios.scenario_definitions import ScenarioBuilder, ScenarioType


def run_profiler():
    pr = cProfile.Profile()
    pr.enable()

    cfg = SimulationConfig(seed=42, agent_count=10)
    cfg.arena.max_ticks = 25
    engine = ScenarioBuilder.build(ScenarioType.STATIC, cfg)
    engine.run(25)

    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(35)

    out_file = root / "results" / "profile.txt"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("AD-QPSO MULTI-AGENT SWARM SIMULATION PROFILING REPORT (tottime)\n")
        f.write("Scenario: static, Agents: 10, Ticks: 25, Seed: 42\n")
        f.write("=" * 70 + "\n\n")
        f.write(s.getvalue())

    print(f"Profile report generated at {out_file}")


if __name__ == "__main__":
    run_profiler()
