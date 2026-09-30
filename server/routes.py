"""API route handlers and endpoint controllers for Mission Control."""

from __future__ import annotations
import os
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from configs.config import SimulationConfig, LatencyBudgetConfig
from scenarios.scenario_definitions import ScenarioBuilder
from src.simulation.engine import SimulationEngine
from src.environment.obstacles import CircularObstacle, RectangularObstacle, DynamicObstacle, HazardZone
from server.schemas import SimRequest, require_key
from server.dashboard import get_dashboard_html

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
ZIP_PATH = BASE_DIR / "adaptive_swarm_intelligence_solution.zip"

ALLOWED_CHARTS = {
    "trajectory_map.png",
    "convergence_graph.png",
    "latency_graph.png",
    "scalability_graph.png",
    "collision_safety_graph.png",
    "throughput_graph.png",
    "robustness_graph.png"
}


def create_solution_zip() -> Path:
    """Creates a clean zip archive of the entire solution, excluding caches and node_modules."""
    exclude_dirs = {"venv", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git", ".idea", ".vscode"}
    exclude_exts = {".pyc", ".pyo", ".pyd", ".DS_Store"}

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(BASE_DIR):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if any(file.endswith(ext) for ext in exclude_exts) or file.startswith("._") or file == ".env":
                    continue
                if file.endswith(".zip"):
                    continue
                full_path = Path(root) / file
                rel_path = full_path.relative_to(BASE_DIR)
                zipf.write(full_path, rel_path)
    return ZIP_PATH


def extract_arena_entities(engine: SimulationEngine) -> Dict[str, Any]:
    """Helper extracting obstacle, hazard zone, and target geometries for canvas rendering."""
    obstacles = []
    for obs in engine.env.obstacles:
        if isinstance(obs, CircularObstacle):
            obstacles.append({
                "type": "circle",
                "center": [float(obs.center[0]), float(obs.center[1])],
                "radius": float(obs.radius)
            })
        elif isinstance(obs, RectangularObstacle):
            obstacles.append({
                "type": "rect",
                "x_min": float(obs.x_min),
                "x_max": float(obs.x_max),
                "y_min": float(obs.y_min),
                "y_max": float(obs.y_max),
                "width": float(obs.x_max - obs.x_min),
                "height": float(obs.y_max - obs.y_min),
                "center": [float((obs.x_min + obs.x_max) / 2.0), float((obs.y_min + obs.y_max) / 2.0)]
            })

    hazard_zones = []
    for hz in engine.env.hazard_zones:
        hazard_zones.append({
            "center": [float(hz.center[0]), float(hz.center[1])],
            "radius": float(hz.radius),
            "risk_multiplier": float(hz.risk_multiplier)
        })

    targets = []
    for t in engine.env.targets:
        targets.append({
            "id": int(t.task_id),
            "position": [float(t.position[0]), float(t.position[1])],
            "priority": float(t.priority),
            "radius": float(t.satisfaction_radius),
            "completed": bool(t.completed)
        })

    return {
        "obstacles": obstacles,
        "hazard_zones": hazard_zones,
        "targets": targets
    }


@router.get("/", response_class=HTMLResponse)
@router.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the interactive Mission Control dashboard."""
    return get_dashboard_html()


@router.get("/health")
@router.get("/api/health")
def get_health():
    """Health status and solver version metadata."""
    return {"status": "ok", "version": "1.0.0", "solver": "AD-QPSO"}


@router.get("/api/summary")
def get_summary():
    """Returns high-level simulation summary metrics."""
    summary_path = RESULTS_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            return json.load(f)
    return {"status": "no summary generated yet"}


@router.get("/api/metrics")
def get_metrics():
    """Returns full 16-metric telemetry breakdown."""
    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {"status": "no metrics generated yet"}


@router.get("/api/benchmark")
def get_benchmark():
    """Returns parsed multi-baseline comparison records."""
    benchmark_path = RESULTS_DIR / "benchmark.csv"
    if benchmark_path.exists():
        df = pd.read_csv(benchmark_path)
        return df.to_dict(orient="records")
    return []


@router.get("/api/ablation")
def get_ablation():
    """Returns parsed 6-stage component ablation study records."""
    ablation_path = RESULTS_DIR / "ablation.csv"
    if ablation_path.exists():
        df = pd.read_csv(ablation_path)
        return df.to_dict(orient="records")
    return []


@router.get("/api/charts/{filename}")
def get_chart(filename: str):
    """Safely serves generated analytical graphs using an explicit allow-list."""
    safe_name = os.path.basename(filename)
    if safe_name not in ALLOWED_CHARTS:
        raise HTTPException(status_code=403, detail="Requested chart file is not in the allowed chart registry")

    file_path = RESULTS_DIR / safe_name
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Chart not found")


@router.get("/api/download/solution.zip")
@router.head("/api/download/solution.zip")
def download_solution():
    """Serves the verified solution zip archive."""
    if not ZIP_PATH.exists():
        create_solution_zip()
    return FileResponse(
        ZIP_PATH,
        media_type="application/zip",
        filename="adaptive_swarm_intelligence_solution.zip"
    )


@router.get("/api/replay")
def get_replay_data():
    """Returns static arena entities and trajectory traces for canvas animation playback."""
    cfg = SimulationConfig(seed=42)
    engine = ScenarioBuilder.build("stress", cfg)
    entities = extract_arena_entities(engine)

    traj_path = RESULTS_DIR / "trajectory.csv"
    agent_paths = {}
    if traj_path.exists():
        df = pd.read_csv(traj_path)
        for _, row in df.iterrows():
            aid = int(row["agent_id"])
            if aid not in agent_paths:
                agent_paths[aid] = []
            agent_paths[aid].append({
                "tick": int(row["tick"]),
                "x": float(row["pos_x"]),
                "y": float(row["pos_y"]),
                "active": bool(row["active"])
            })

    max_ticks = max([len(p) for p in agent_paths.values()]) if agent_paths else 120
    return {
        "arena": {"width": cfg.arena.width, "height": cfg.arena.height},
        "obstacles": entities["obstacles"],
        "hazard_zones": entities["hazard_zones"],
        "targets": entities["targets"],
        "agent_paths": agent_paths,
        "max_ticks": max_ticks,
        "communication_radius": cfg.agent_params.communication_radius
    }


@router.post("/api/simulate", dependencies=[Depends(require_key)])
def run_simulation_api(req: SimRequest):
    """Executes a scenario dynamically with Pydantic validation and error containment."""
    try:
        cfg = SimulationConfig(
            seed=req.seed,
            agent_count=req.agents,
            latency_budget=LatencyBudgetConfig(budget_ms=req.latency_budget)
        )
        cfg.arena.max_ticks = req.ticks

        engine = ScenarioBuilder.build(req.scenario, cfg)
        metrics = engine.run(req.ticks)
        entities = extract_arena_entities(engine)

        agent_paths = {}
        for aid, agent in engine.agents.items():
            agent_paths[aid] = [
                {"tick": idx, "x": round(float(pos[0]), 3), "y": round(float(pos[1]), 3), "active": bool(agent.state.active)}
                for idx, pos in enumerate(agent.state.history_path)
            ]

        dynamic_obs_frames = []
        for do in engine.env.dynamic_obstacles:
            dynamic_obs_frames.append({
                "id": str(do.obs_id),
                "center": [float(do.center[0]), float(do.center[1])],
                "radius": float(do.radius),
                "velocity": [float(do.velocity[0]), float(do.velocity[1])]
            })

        return {
            "status": "success",
            "arena": {"width": cfg.arena.width, "height": cfg.arena.height},
            "obstacles": entities["obstacles"],
            "hazard_zones": entities["hazard_zones"],
            "targets": entities["targets"],
            "metrics": metrics.to_dict(),
            "agent_paths": agent_paths,
            "dynamic_obstacles": dynamic_obs_frames,
            "total_ticks": req.ticks,
            "max_ticks": req.ticks,
            "communication_radius": cfg.agent_params.communication_radius
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation execution failed: {type(exc).__name__}")
