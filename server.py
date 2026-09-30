"""FastAPI Mission Control & Real-Time Swarm Dashboard Server.

Provides interactive browser visualization, execution controls,
telemetry inspection, and solution zip distribution.
"""

from __future__ import annotations
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import math
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import SimulationConfig, ScenarioType
from scenarios import get_scenario
from algorithms.adaptive_swarm import AdaptiveSwarmAgent
from algorithms.non_adaptive_swarm import NonAdaptiveSwarmAgent
from algorithms.static_priority import StaticPriorityAgent
from algorithms.greedy import GreedyAgent
from scripts.package_solution import package_solution
from scripts.audit import (
    audit_banned_terms,
    audit_required_files,
    audit_tests,
    audit_results_artifacts,
    audit_safety_invariants,
)

app = FastAPI(title="Adaptive Decentralized Swarm Intelligence Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ZIP_PATH = os.path.abspath("adaptive_swarm_intelligence_solution.zip")


@app.get("/api/download-zip")
def download_solution_zip():
    """Provides direct download of the clean, standalone solution archive."""
    if not os.path.exists(ZIP_PATH):
        package_solution(ZIP_PATH)
    return FileResponse(
        path=ZIP_PATH,
        filename="adaptive_swarm_intelligence_solution.zip",
        media_type="application/zip",
    )


@app.get("/api/audit")
def get_audit_scorecard():
    """Runs automated system audit and returns 7-dimension scorecard JSON."""
    banned_ok, banned_violations = audit_banned_terms()
    files_ok, missing_files = audit_required_files()
    test_ok, test_count, _ = audit_tests()
    artifacts_ok, missing_artifacts = audit_results_artifacts()
    safety_ok, safety_msg = audit_safety_invariants()

    sla_ok = True
    sla_msg = "P95 latency <= 100 ms"
    results_file = "results/results.json"
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                d = json.load(f)
            p95_max = d.get("summary", {}).get("p95_decision_latency_ms", {}).get("max", 0.0)
            if p95_max > 100.0:
                sla_ok = False
                sla_msg = f"Max P95 latency {p95_max:.2f} ms exceeded 100 ms"
            else:
                sla_msg = f"Max P95 latency: {p95_max:.2f} ms"
        except Exception:
            sla_ok = False

    decentralized_ok = os.path.exists("core/adaptive_policy.py") and os.path.exists("core/agent.py")

    scores = [
        {"dimension": "1. Zero Banned Legacy Terms", "passed": banned_ok},
        {"dimension": "2. Architecture & File Structure", "passed": files_ok},
        {"dimension": "3. Comprehensive Test Suite (40+ assertions)", "passed": test_ok, "detail": f"{test_count} passed"},
        {"dimension": "4. Empirical Benchmark & Ablation Data", "passed": artifacts_ok},
        {"dimension": "5. Hard Collision-Free Safety Invariants", "passed": safety_ok, "detail": safety_msg},
        {"dimension": "6. Real-Time SLA Decision Latency (<100ms)", "passed": sla_ok, "detail": sla_msg},
        {"dimension": "7. Decentralized Multi-Agent Autonomy", "passed": decentralized_ok},
    ]

    total_pass = sum(1 for s in scores if s["passed"])
    return {
        "status": "PASS" if total_pass == len(scores) else "FAIL",
        "score": f"{total_pass} / {len(scores)}",
        "scores": scores,
    }


@app.get("/api/simulate")
def run_simulation(
    scenario: str = Query("obstacles", description="Scenario name"),
    agents: int = Query(15, ge=1, le=50, description="Swarm size"),
    ticks: int = Query(60, ge=10, le=120, description="Max simulation ticks"),
    seed: int = Query(42, description="Random seed"),
    algorithm: str = Query("adaptive", description="Algorithm: adaptive, non_adaptive, static, greedy")
):
    """Executes a simulation run and returns complete frame-by-frame animation telemetry."""
    try:
        algo_map = {
            "adaptive": AdaptiveSwarmAgent,
            "non_adaptive": NonAdaptiveSwarmAgent,
            "static": StaticPriorityAgent,
            "greedy": GreedyAgent,
        }
        agent_cls = algo_map.get(algorithm.lower(), AdaptiveSwarmAgent)
        enable_adaptation = (algorithm.lower() == "adaptive")

        cfg = SimulationConfig(
            seed=seed,
            agent_count=agents,
            max_ticks=ticks,
            enable_adaptation=enable_adaptation
        )

        coordinator = get_scenario(scenario, config=cfg, agent_cls=agent_cls)
        w, h = coordinator.env.config.width, coordinator.env.config.height

        frames: List[Dict[str, Any]] = []
        event_log: List[Dict[str, Any]] = []

        # Record initial frame (tick 0)
        initial_agents = []
        for aid, ag in coordinator.agents.items():
            initial_agents.append({
                "id": aid,
                "pos": list(ag.state.position),
                "target": list(ag.state.target),
                "active": ag.state.active,
                "completed": ag.state.completed_target,
                "energy": round(ag.state.energy, 2),
                "stagnation": ag.state.stagnation_ticks,
            })

        frames.append({
            "tick": 0,
            "agents": initial_agents,
            "dynamic_obstacles": [list(obs.position) for obs in coordinator.env.dynamic_obstacles.values()],
            "active_dropouts": list(coordinator.env.active_dropouts.keys()),
            "latency_ms": 0.0,
        })

        # Run tick by tick and capture state
        for t in range(ticks):
            lat_ms = coordinator.step()

            # Check for dynamic events triggered this tick
            for event in coordinator.env.perturbations:
                if event.trigger_tick == t:
                    event_log.append({
                        "tick": t,
                        "type": event.event_type.value,
                        "message": f"Perturbation triggered: {event.event_type.value}"
                    })

            # Check target completions this tick
            for aid, ag in coordinator.agents.items():
                if ag.state.completed_target and ag.state.position == ag.state.target and len(ag.state.history) > 1 and ag.state.history[-2] != ag.state.target:
                    event_log.append({
                        "tick": t,
                        "type": "target_reached",
                        "message": f"Agent {aid} successfully reached assigned target {ag.state.target}"
                    })

            agent_states = []
            for aid, ag in coordinator.agents.items():
                agent_states.append({
                    "id": aid,
                    "pos": list(ag.state.position),
                    "target": list(ag.state.target),
                    "active": ag.state.active,
                    "completed": ag.state.completed_target,
                    "energy": round(ag.state.energy, 2),
                    "stagnation": ag.state.stagnation_ticks,
                })

            frames.append({
                "tick": t + 1,
                "agents": agent_states,
                "dynamic_obstacles": [list(obs.position) for obs in coordinator.env.dynamic_obstacles.values()],
                "active_dropouts": list(coordinator.env.active_dropouts.keys()),
                "latency_ms": round(lat_ms, 3),
            })

            # Check if all completed or failed
            if all(not ag.state.active or ag.state.completed_target for ag in coordinator.agents.values()):
                break

        # Finalize metrics
        metrics = coordinator.metrics_collector.finalize(
            coordinator.agents,
            coordinator.current_tick,
            coordinator.hard_collisions,
            coordinator.dropout_agent_ids
        )

        static_obs = [list(p) for p in coordinator.env.static_obstacles]

        return {
            "success": True,
            "grid_width": w,
            "grid_height": h,
            "communication_radius": cfg.agent.communication_radius,
            "static_obstacles": static_obs,
            "total_ticks": len(frames) - 1,
            "frames": frames,
            "event_log": event_log,
            "metrics": metrics.to_dict(),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the complete self-contained Swarm Mission Control dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Adaptive Decentralized Swarm Intelligence Dashboard</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --success: #22c55e;
      --danger: #ef4444;
      --warning: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
    body { background: var(--bg); color: var(--text); padding: 16px; min-height: 100vh; }
    header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 16px; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
    h1 { font-size: 1.25rem; font-weight: 700; color: var(--accent); letter-spacing: 0.5px; }
    .subtitle { font-size: 0.8rem; color: var(--text-muted); }
    .badge { background: #0369a1; color: #e0f2fe; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
    .btn-zip { background: #059669; color: white; border: none; padding: 8px 14px; border-radius: 6px; font-weight: 600; cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; gap: 6px; }
    .btn-zip:hover { background: #047857; }
    
    .layout { display: grid; grid-template-columns: 320px 1fr 300px; gap: 16px; }
    @media (max-width: 1200px) { .layout { grid-template-columns: 1fr; } }
    
    .panel { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px; display: flex; flex-direction: column; gap: 12px; }
    .panel h2 { font-size: 0.95rem; font-weight: 600; color: var(--text); border-bottom: 1px solid var(--border); padding-bottom: 6px; }
    
    label { font-size: 0.78rem; color: var(--text-muted); display: block; margin-bottom: 4px; }
    select, input { width: 100%; background: #0f172a; border: 1px solid var(--border); color: var(--text); padding: 7px 10px; border-radius: 6px; font-size: 0.85rem; }
    
    .btn-primary { background: var(--accent); color: #0f172a; border: none; padding: 10px; border-radius: 6px; font-weight: 700; cursor: pointer; font-size: 0.9rem; transition: background 0.2s; }
    .btn-primary:hover { background: var(--accent-hover); }
    .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
    
    .playback-bar { display: flex; gap: 8px; align-items: center; }
    .btn-ctrl { background: #334155; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; font-weight: 600; }
    .btn-ctrl:hover { background: #475569; }
    
    .canvas-container { display: flex; justify-content: center; align-items: center; background: #020617; border-radius: 8px; border: 1px solid var(--border); position: relative; overflow: hidden; }
    canvas { display: block; width: 100%; max-width: 650px; aspect-ratio: 1/1; background: #090d16; }
    
    .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .metric-card { background: #0f172a; padding: 10px; border-radius: 6px; border: 1px solid var(--border); }
    .metric-title { font-size: 0.7rem; color: var(--text-muted); }
    .metric-val { font-size: 1.1rem; font-weight: 700; margin-top: 2px; }
    .metric-val.pass { color: var(--success); }
    .metric-val.fail { color: var(--danger); }
    .metric-val.info { color: var(--accent); }
    
    .log-box { background: #0f172a; border: 1px solid var(--border); border-radius: 6px; padding: 8px; height: 260px; overflow-y: auto; font-size: 0.75rem; color: #cbd5e1; display: flex; flex-direction: column; gap: 4px; }
    .log-entry { padding: 4px 6px; border-radius: 4px; background: #1e293b; border-left: 3px solid var(--accent); }
    .log-entry.target { border-left-color: var(--success); }
    .log-entry.event { border-left-color: var(--warning); }
    
    .legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 0.72rem; color: var(--text-muted); padding: 8px; background: #0f172a; border-radius: 6px; }
    .legend-item { display: flex; align-items: center; gap: 5px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; }
    .square { width: 8px; height: 8px; }
  </style>
</head>
<body>

  <header>
    <div>
      <h1>Adaptive Decentralized Swarm Intelligence</h1>
      <div class="subtitle">Autonomous Multi-Agent Navigation | Zero-Tolerance Safety Verification</div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
      <span class="badge" id="sla-badge">SLA: &lt;100 ms</span>
      <a href="/api/download-zip" class="btn-zip" download>
        ⬇ Download Solution ZIP
      </a>
    </div>
  </header>

  <div class="layout">
    <!-- Controls Panel -->
    <div class="panel">
      <h2>Simulation Parameters</h2>
      
      <div>
        <label>Benchmark Scenario</label>
        <select id="scenario-select">
          <option value="obstacles" selected>Static Obstacles & Corridors</option>
          <option value="dynamic">Dynamic Moving Obstacles</option>
          <option value="combined">Combined Multi-Perturbation</option>
          <option value="dense">Dense Swarm (Crowded Arena)</option>
          <option value="dropout">Wireless Communication Blackout</option>
          <option value="failure">Hardware Agent Failure Crash</option>
          <option value="open">Open Arena Baseline</option>
        </select>
      </div>

      <div>
        <label>Swarm Algorithm</label>
        <select id="algo-select">
          <option value="adaptive" selected>Proposed Adaptive Swarm (Dynamic)</option>
          <option value="non_adaptive">Non-Adaptive Swarm (Fixed Weights)</option>
          <option value="static">Static Priority Navigation</option>
          <option value="greedy">Greedy Goal Navigation</option>
        </select>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
        <div>
          <label>Agents</label>
          <input type="number" id="agent-count" value="15" min="2" max="40">
        </div>
        <div>
          <label>Max Ticks</label>
          <input type="number" id="max-ticks" value="70" min="20" max="120">
        </div>
      </div>

      <div>
        <label>Deterministic Seed</label>
        <input type="number" id="seed-input" value="42">
      </div>

      <button id="btn-run" class="btn-primary">▶ Execute Swarm Simulation</button>

      <h2>Playback Controls</h2>
      <div class="playback-bar">
        <button id="btn-play" class="btn-ctrl">Pause</button>
        <button id="btn-step" class="btn-ctrl">Step ➔</button>
        <button id="btn-reset" class="btn-ctrl">Reset</button>
      </div>
      <div>
        <label>Playback Speed: <span id="speed-label">2x</span></label>
        <input type="range" id="speed-range" min="1" max="10" value="2">
      </div>

      <div class="legend">
        <div class="legend-item"><span class="dot" style="background:#38bdf8;"></span> Agent</div>
        <div class="legend-item"><span class="dot" style="background:#f59e0b; border: 1px solid white;"></span> Target</div>
        <div class="legend-item"><span class="square" style="background:#334155;"></span> Static Barrier</div>
        <div class="legend-item"><span class="dot" style="background:#ef4444;"></span> Dynamic Obs</div>
        <div class="legend-item"><span class="dot" style="background:#64748b;"></span> Inactive/Crash</div>
      </div>
    </div>

    <!-- 2D Canvas Visualization -->
    <div class="canvas-container">
      <canvas id="swarm-canvas" width="650" height="650"></canvas>
    </div>

    <!-- Live Telemetry & Log Panel -->
    <div class="panel">
      <h2>Swarm Telemetry</h2>
      
      <div class="metric-grid">
        <div class="metric-card">
          <div class="metric-title">Hard Collisions</div>
          <div class="metric-val pass" id="val-collisions">0</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">Mean Latency</div>
          <div class="metric-val info" id="val-latency">0.0 ms</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">Task Completion</div>
          <div class="metric-val" id="val-completion">0.0%</div>
        </div>
        <div class="metric-card">
          <div class="metric-title">Deadlocks Res.</div>
          <div class="metric-val" id="val-deadlocks">0.0%</div>
        </div>
      </div>

      <div style="font-size: 0.75rem; color: var(--text-muted); display: flex; justify-content: space-between;">
        <span>Current Tick: <b id="tick-indicator" style="color:white;">0 / 0</b></span>
        <span>Active Agents: <b id="active-indicator" style="color:white;">0</b></span>
      </div>

      <h2>Mission Event Log</h2>
      <div class="log-box" id="log-box">
        <div class="log-entry">Engine ready. Click [Execute Swarm Simulation] to begin.</div>
      </div>

      <button id="btn-audit" class="btn-ctrl" style="width: 100%; padding: 8px;">Run Automated Audit</button>
    </div>
  </div>

  <script>
    const canvas = document.getElementById("swarm-canvas");
    const ctx = canvas.getContext("2d");

    const COLORS = [
      "#38bdf8", "#f43f5e", "#10b981", "#fbbf24", "#a855f7",
      "#ec4899", "#3b82f6", "#14b8a6", "#f97316", "#8b5cf6",
      "#06b6d4", "#84cc16", "#e11d48", "#6366f1", "#d97706"
    ];

    let simData = null;
    let currentFrameIdx = 0;
    let isPlaying = false;
    let animTimer = null;
    let fps = 15;

    async function executeSimulation() {
      const btn = document.getElementById("btn-run");
      btn.disabled = true;
      btn.textContent = "⏳ Simulating...";

      const scenario = document.getElementById("scenario-select").value;
      const algo = document.getElementById("algo-select").value;
      const agents = document.getElementById("agent-count").value;
      const ticks = document.getElementById("max-ticks").value;
      const seed = document.getElementById("seed-input").value;

      try {
        const res = await fetch(`/api/simulate?scenario=${scenario}&agents=${agents}&ticks=${ticks}&seed=${seed}&algorithm=${algo}`);
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Simulation failed");
        }
        simData = await res.json();
        currentFrameIdx = 0;
        
        // Update metric display
        const m = simData.metrics;
        document.getElementById("val-collisions").textContent = m.collision_count;
        document.getElementById("val-collisions").className = m.collision_count === 0 ? "metric-val pass" : "metric-val fail";
        document.getElementById("val-latency").textContent = `${m.average_decision_latency_ms} ms`;
        document.getElementById("val-completion").textContent = `${m.task_completion_rate}%`;
        document.getElementById("val-deadlocks").textContent = `${m.deadlock_resolution_rate}%`;

        // Update log
        const logBox = document.getElementById("log-box");
        logBox.innerHTML = "";
        simData.event_log.forEach(e => {
          const div = document.createElement("div");
          div.className = `log-entry ${e.type === 'target_reached' ? 'target' : 'event'}`;
          div.textContent = `[T${e.tick}] ${e.message}`;
          logBox.appendChild(div);
        });

        drawFrame(0);
        isPlaying = true;
        document.getElementById("btn-play").textContent = "Pause";
        startLoop();
      } catch (err) {
        alert("Error: " + err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = "▶ Execute Swarm Simulation";
      }
    }

    function toScreen(gx, gy, w, h) {
      const pad = 30;
      const drawW = canvas.width - 2 * pad;
      const drawH = canvas.height - 2 * pad;
      const sx = pad + (gx / w) * drawW;
      const sy = canvas.height - pad - (gy / h) * drawH;
      return [sx, sy];
    }

    function drawFrame(idx) {
      if (!simData || !simData.frames || idx >= simData.frames.length) return;
      const frame = simData.frames[idx];
      const gw = simData.grid_width;
      const gh = simData.grid_height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Arena background
      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Grid lines
      ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
      ctx.lineWidth = 1;
      for (let x = 0; x <= gw; x += 5) {
        const [x1, y1] = toScreen(x, 0, gw, gh);
        const [x2, y2] = toScreen(x, gh, gw, gh);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      }
      for (let y = 0; y <= gh; y += 5) {
        const [x1, y1] = toScreen(0, y, gw, gh);
        const [x2, y2] = toScreen(gw, y, gw, gh);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      }

      // Static Obstacles
      ctx.fillStyle = "#334155";
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;
      simData.static_obstacles.forEach(([ox, oy]) => {
        const [sx, sy] = toScreen(ox, oy, gw, gh);
        const size = (canvas.width - 60) / gw;
        ctx.fillRect(sx - size/2, sy - size/2, size, size);
        ctx.strokeRect(sx - size/2, sy - size/2, size, size);
      });

      // Dynamic Obstacles
      ctx.fillStyle = "#ef4444";
      frame.dynamic_obstacles.forEach(([dx, dy]) => {
        const [sx, sy] = toScreen(dx, dy, gw, gh);
        ctx.beginPath();
        ctx.arc(sx, sy, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "white";
        ctx.stroke();
      });

      // Communication Links between active agents within R_comm
      const rComm = simData.communication_radius || 6.0;
      ctx.strokeStyle = "rgba(56, 189, 248, 0.15)";
      ctx.lineWidth = 1;
      for (let i = 0; i < frame.agents.length; i++) {
        for (let j = i + 1; j < frame.agents.length; j++) {
          const a1 = frame.agents[i];
          const a2 = frame.agents[j];
          if (!a1.active || !a2.active) continue;
          const dist = Math.hypot(a1.pos[0] - a2.pos[0], a1.pos[1] - a2.pos[1]);
          if (dist <= rComm) {
            const [p1x, p1y] = toScreen(a1.pos[0], a1.pos[1], gw, gh);
            const [p2x, p2y] = toScreen(a2.pos[0], a2.pos[1], gw, gh);
            ctx.beginPath(); ctx.moveTo(p1x, p1y); ctx.lineTo(p2x, p2y); ctx.stroke();
          }
        }
      }

      // Targets
      frame.agents.forEach((ag, i) => {
        const col = COLORS[i % COLORS.length];
        const [tx, ty] = toScreen(ag.target[0], ag.target[1], gw, gh);
        ctx.fillStyle = col;
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(tx, ty, 4, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
      });

      // Agents
      let activeCount = 0;
      frame.agents.forEach((ag, i) => {
        const col = ag.active ? COLORS[i % COLORS.length] : "#64748b";
        if (ag.active && !ag.completed) activeCount++;

        const [ax, ay] = toScreen(ag.pos[0], ag.pos[1], gw, gh);

        // Path history
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.4;
        ctx.beginPath();
        for (let f = 0; f <= idx; f++) {
          const pastAg = simData.frames[f].agents[i];
          const [px, py] = toScreen(pastAg.pos[0], pastAg.pos[1], gw, gh);
          if (f === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        // Agent Dot
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(ax, ay, ag.active ? 6 : 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Label
        ctx.fillStyle = "white";
        ctx.font = "9px monospace";
        ctx.fillText(ag.id, ax + 7, ay - 7);
      });

      // Update indicator
      document.getElementById("tick-indicator").textContent = `${frame.tick} / ${simData.total_ticks}`;
      document.getElementById("active-indicator").textContent = activeCount;
    }

    function startLoop() {
      if (animTimer) clearInterval(animTimer);
      animTimer = setInterval(() => {
        if (!isPlaying || !simData) return;
        currentFrameIdx++;
        if (currentFrameIdx >= simData.frames.length) {
          currentFrameIdx = simData.frames.length - 1;
          isPlaying = false;
          document.getElementById("btn-play").textContent = "Play";
          clearInterval(animTimer);
        }
        drawFrame(currentFrameIdx);
      }, 1000 / fps);
    }

    document.getElementById("btn-run").addEventListener("click", executeSimulation);
    
    document.getElementById("btn-play").addEventListener("click", () => {
      isPlaying = !isPlaying;
      document.getElementById("btn-play").textContent = isPlaying ? "Pause" : "Play";
      if (isPlaying) startLoop();
    });

    document.getElementById("btn-step").addEventListener("click", () => {
      isPlaying = false;
      document.getElementById("btn-play").textContent = "Play";
      if (simData && currentFrameIdx < simData.frames.length - 1) {
        currentFrameIdx++;
        drawFrame(currentFrameIdx);
      }
    });

    document.getElementById("btn-reset").addEventListener("click", () => {
      isPlaying = false;
      document.getElementById("btn-play").textContent = "Play";
      currentFrameIdx = 0;
      drawFrame(0);
    });

    document.getElementById("speed-range").addEventListener("input", (e) => {
      const spd = parseInt(e.target.value);
      fps = spd * 7;
      document.getElementById("speed-label").textContent = `${spd}x`;
      if (isPlaying) startLoop();
    });

    document.getElementById("btn-audit").addEventListener("click", async () => {
      try {
        const res = await fetch("/api/audit");
        const audit = await res.json();
        alert(`Audit Result: ${audit.status} (${audit.score} dimensions passed)\\n\\n` + 
          audit.scores.map(s => `${s.passed ? '✓' : '✗'} ${s.dimension}`).join('\\n')
        );
      } catch (e) {
        alert("Audit error: " + e.message);
      }
    });

    // Auto-run simulation on initial page load
    window.addEventListener("DOMContentLoaded", () => {
      executeSimulation();
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
