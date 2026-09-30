"""Publication-quality 2D spatial rendering of swarm trajectories, entities, and events."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.simulation.engine import SimulationEngine
from src.environment.obstacles import CircularObstacle, RectangularObstacle, DynamicObstacle, HazardZone
from src.environment.targets import TaskTarget


class SwarmVisualizer:
    """Renders high-definition spatial map of multi-agent swarm simulation."""

    def __init__(self, width: int = 1200, height: int = 1200):
        self.img_w = width
        self.img_h = height

    def _world_to_pixel(
        self,
        pt: np.ndarray,
        bounds: tuple[float, float, float, float],
        padding: int = 80
    ) -> tuple[int, int]:
        x_min, x_max, y_min, y_max = bounds
        usable_w = self.img_w - 2 * padding
        usable_h = self.img_h - 2 * padding

        px = int(padding + ((pt[0] - x_min) / (x_max - x_min)) * usable_w)
        # Flip Y for standard Cartesian display
        py = int(self.img_h - padding - ((pt[1] - y_min) / (y_max - y_min)) * usable_h)
        return (px, py)

    def render_map(
        self,
        engine: SimulationEngine,
        output_path: str = "results/trajectory_map.png"
    ) -> None:
        """Generates full publication-grade 2D snapshot of the multi-agent mission."""
        bounds = engine.config.arena.bounds
        padding = 90
        img = Image.new("RGB", (self.img_w, self.img_h), color=(248, 250, 252)) # Light slate
        draw = ImageDraw.Draw(img)

        # 1. Grid lines
        for gx in np.linspace(bounds[0], bounds[1], 11):
            p1 = self._world_to_pixel(np.array([gx, bounds[2]]), bounds, padding)
            p2 = self._world_to_pixel(np.array([gx, bounds[3]]), bounds, padding)
            draw.line([p1, p2], fill=(226, 232, 240), width=1)
        for gy in np.linspace(bounds[2], bounds[3], 11):
            p1 = self._world_to_pixel(np.array([bounds[0], gy]), bounds, padding)
            p2 = self._world_to_pixel(np.array([bounds[1], gy]), bounds, padding)
            draw.line([p1, p2], fill=(226, 232, 240), width=1)

        # Arena boundary border
        b_tl = self._world_to_pixel(np.array([bounds[0], bounds[3]]), bounds, padding)
        b_br = self._world_to_pixel(np.array([bounds[1], bounds[2]]), bounds, padding)
        draw.rectangle([b_tl, b_br], outline=(100, 116, 139), width=2)

        # 2. Hazard Zones
        for hz in engine.env.hazard_zones:
            c_px = self._world_to_pixel(hz.center, bounds, padding)
            scale = (self.img_w - 2 * padding) / (bounds[1] - bounds[0])
            r_px = int(hz.radius * scale)
            draw.ellipse(
                [c_px[0] - r_px, c_px[1] - r_px, c_px[0] + r_px, c_px[1] + r_px],
                fill=(254, 243, 199), # Pale amber
                outline=(245, 158, 11),
                width=2
            )

        # 3. Static and Dynamic Obstacles
        for obs in engine.env.obstacles:
            scale = (self.img_w - 2 * padding) / (bounds[1] - bounds[0])
            if isinstance(obs, RectangularObstacle):
                p_min = self._world_to_pixel(np.array([obs.x_min, obs.y_max]), bounds, padding)
                p_max = self._world_to_pixel(np.array([obs.x_max, obs.y_min]), bounds, padding)
                # Obstacle body
                draw.rectangle([p_min, p_max], fill=(71, 85, 105), outline=(30, 41, 59), width=2)

            elif isinstance(obs, CircularObstacle):
                c_px = self._world_to_pixel(obs.center, bounds, padding)
                r_px = int(obs.radius * scale)
                draw.ellipse(
                    [c_px[0] - r_px, c_px[1] - r_px, c_px[0] + r_px, c_px[1] + r_px],
                    fill=(71, 85, 105), outline=(30, 41, 59), width=2
                )

            elif isinstance(obs, DynamicObstacle):
                c_px = self._world_to_pixel(obs.center, bounds, padding)
                r_px = int(obs.radius * scale)
                # Trail
                if len(obs.trajectory_history) > 1:
                    trail_pts = [self._world_to_pixel(p, bounds, padding) for p in obs.trajectory_history]
                    draw.line(trail_pts, fill=(239, 68, 68), width=2)
                draw.ellipse(
                    [c_px[0] - r_px, c_px[1] - r_px, c_px[0] + r_px, c_px[1] + r_px],
                    fill=(220, 38, 38), outline=(153, 27, 27), width=3
                )

        # 4. Communication Links
        for (i, j) in engine.comm_mesh.active_links:
            if i in engine.agents and j in engine.agents:
                p_i = self._world_to_pixel(engine.agents[i].state.position, bounds, padding)
                p_j = self._world_to_pixel(engine.agents[j].state.position, bounds, padding)
                draw.line([p_i, p_j], fill=(147, 197, 253), width=1) # Soft blue

        # 5. Task Targets
        for target in engine.env.targets:
            t_px = self._world_to_pixel(target.position, bounds, padding)
            r = 7
            if target.completed:
                # Green circle with check
                draw.ellipse([t_px[0] - r, t_px[1] - r, t_px[0] + r, t_px[1] + r], fill=(34, 197, 94), outline=(21, 128, 61), width=2)
            else:
                # Gold diamond
                diamond = [(t_px[0], t_px[1] - r - 2), (t_px[0] + r + 2, t_px[1]), (t_px[0], t_px[1] + r + 2), (t_px[0] - r - 2, t_px[1])]
                draw.polygon(diamond, fill=(234, 179, 8), outline=(161, 98, 7))

        # 6. Agent Trajectory Trails and Agents
        agent_colors = [
            (2, 132, 199), (13, 148, 136), (124, 58, 237), (217, 70, 239),
            (234, 88, 12), (5, 150, 105), (37, 99, 235), (192, 38, 211),
            (14, 165, 233), (16, 185, 129), (99, 102, 241), (244, 63, 94)
        ]

        for aid, ag in engine.agents.items():
            color = agent_colors[aid % len(agent_colors)]
            hist = ag.state.history_path
            if len(hist) > 1:
                trail_pixels = [self._world_to_pixel(p, bounds, padding) for p in hist]
                draw.line(trail_pixels, fill=color, width=2)

            curr_px = self._world_to_pixel(ag.state.position, bounds, padding)
            ag_r = 6
            if ag.state.active:
                draw.ellipse(
                    [curr_px[0] - ag_r, curr_px[1] - ag_r, curr_px[0] + ag_r, curr_px[1] + ag_r],
                    fill=color, outline=(15, 23, 42), width=2
                )
            else:
                # Failed agent marker (red cross)
                draw.line([curr_px[0] - ag_r, curr_px[1] - ag_r, curr_px[0] + ag_r, curr_px[1] + ag_r], fill=(220, 38, 38), width=3)
                draw.line([curr_px[0] - ag_r, curr_px[1] + ag_r, curr_px[0] + ag_r, curr_px[1] - ag_r], fill=(220, 38, 38), width=3)

        # 7. Title and Telemetry HUD Header
        title = "Adaptive Decentralized Swarm Intelligence — Spatial Trajectory Map"
        draw.text((padding, 25), title, fill=(15, 23, 42))

        completed_cnt = sum(1 for t in engine.env.targets if t.completed)
        hud_str = f"Agents: {len(engine.agents)} | Completed Tasks: {completed_cnt}/{len(engine.env.targets)} | Hard Collisions: {engine.hard_collisions} | P95 Latency: {engine.latency_tracker.compute_report().p95_ms:.2f} ms"
        draw.text((padding, 50), hud_str, fill=(71, 85, 105))

        img.save(output_path, "PNG")
