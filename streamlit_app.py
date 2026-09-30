"""Streamlit front-end for the AD-QPSO swarm project (Streamlit Community Cloud entrypoint).

Shows the committed benchmark results and lets visitors run a small live simulation.
Live runs are capped so they finish quickly on free hosting.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from configs.config import LatencyBudgetConfig, SimulationConfig
from scenarios.scenario_definitions import ScenarioBuilder

RESULTS = Path(__file__).resolve().parent / "results"
SCENARIOS = [
    "static",
    "dynamic_obstacle",
    "multiple_moving_obstacles",
    "communication_dropout",
    "agent_failure",
    "stress",
]
CHARTS = [
    "trajectory_map.png",
    "convergence_graph.png",
    "latency_graph.png",
    "scalability_graph.png",
    "collision_safety_graph.png",
    "throughput_graph.png",
    "robustness_graph.png",
]
MAX_AGENTS, MAX_TICKS = 30, 120

st.set_page_config(page_title="AD-QPSO Swarm Mission Control", layout="wide")
st.title("AD-QPSO: Adaptive Decentralized Swarm Coordination")
st.caption("Benchmark results come from committed files in results/; the simulator tab runs live.")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    """Loads a results CSV if it exists."""
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner="Running swarm simulation...")
def run_sim(scenario: str, agents: int, ticks: int, seed: int, budget_ms: float) -> dict:
    """Runs one capped simulation and returns metrics plus per-agent paths."""
    cfg = SimulationConfig(
        seed=seed,
        agent_count=agents,
        latency_budget=LatencyBudgetConfig(budget_ms=budget_ms),
    )
    cfg.arena.max_ticks = ticks
    engine = ScenarioBuilder.build(scenario, cfg)
    metrics = engine.run(ticks).to_dict()
    rows = [
        {"agent": int(aid), "tick": i, "x": float(p[0]), "y": float(p[1])}
        for aid, agent in engine.agents.items()
        for i, p in enumerate(agent.state.history_path)
    ]
    return {"metrics": metrics, "paths": rows, "arena": (cfg.arena.width, cfg.arena.height)}


tab_results, tab_live = st.tabs(["Benchmark results", "Live simulator"])

with tab_results:
    summary_path = RESULTS / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        cols = st.columns(4)
        cols[0].metric("Task completion", f"{s.get('task_completion_pct', 0):.1f}%")
        cols[1].metric("Mean latency", f"{s.get('mean_latency_ms', 0):.1f} ms")
        cols[2].metric("p95 latency", f"{s.get('p95_latency_ms', 0):.1f} ms")
        cols[3].metric("Hard safety kept", "Yes" if s.get("hard_safety_preserved") else "No")
        if not s.get("real_time_sla_met", False):
            st.warning("The configured real-time latency budget was not met in this run.")
    for title, fname in [("Solver comparison", "benchmark.csv"), ("Ablation study", "ablation.csv")]:
        df = load_csv(fname)
        if df is not None:
            st.subheader(title)
            st.dataframe(df, width="stretch")
    st.subheader("Charts")
    available = [c for c in CHARTS if (RESULTS / c).exists()]
    for i in range(0, len(available), 2):
        for col, name in zip(st.columns(2), available[i : i + 2]):
            col.image(str(RESULTS / name), caption=name.replace("_", " ").removesuffix(".png"))

with tab_live:
    c1, c2, c3, c4 = st.columns(4)
    scenario = c1.selectbox("Scenario", SCENARIOS, index=len(SCENARIOS) - 1)
    agents = c2.slider("Agents", 2, MAX_AGENTS, 10)
    ticks = c3.slider("Ticks", 10, MAX_TICKS, 60)
    seed = int(c4.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42))
    budget = st.number_input("Latency budget (ms)", min_value=1.0, max_value=5000.0, value=25.0)
    if st.button("Run simulation", type="primary"):
        try:
            out = run_sim(scenario, agents, ticks, seed, float(budget))
        except Exception as exc:  # show a friendly error instead of a stack trace
            st.error(f"Simulation failed: {exc}")
        else:
            m = out["metrics"]
            st.json({k: m[k] for k in list(m)[:12]}, expanded=False)
            w, h = out["arena"]
            chart = (
                alt.Chart(pd.DataFrame(out["paths"]))
                .mark_line(point=False)
                .encode(
                    x=alt.X("x:Q", scale=alt.Scale(domain=[0, w])),
                    y=alt.Y("y:Q", scale=alt.Scale(domain=[0, h])),
                    color=alt.Color("agent:N", legend=None),
                    order="tick:Q",
                )
                .properties(height=520)
            )
            st.altair_chart(chart, width="stretch")
