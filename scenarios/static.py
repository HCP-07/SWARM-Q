"""Static benchmark scenario without dynamic perturbations."""

from __future__ import annotations
import numpy as np
from configs.config import SimulationConfig
from src.environment.obstacles import CircularObstacle, RectangularObstacle, HazardZone
from src.environment.targets import TaskTarget
from src.simulation.engine import SimulationEngine


def populate_base_world(engine: SimulationEngine, target_count: int = 18) -> None:
    """Adds static obstacles, hazard zones, and spatial targets."""
    rng = np.random.default_rng(engine.config.seed)
    bounds = engine.config.arena

    # 1. Static obstacles (walls and pillars)
    engine.env.add_obstacle(RectangularObstacle(30.0, 20.0, 36.0, 55.0, obs_id="wall_lower"))
    engine.env.add_obstacle(RectangularObstacle(30.0, 65.0, 36.0, 90.0, obs_id="wall_upper"))
    engine.env.add_obstacle(RectangularObstacle(60.0, 10.0, 66.0, 45.0, obs_id="wall_right_lower"))
    engine.env.add_obstacle(RectangularObstacle(60.0, 55.0, 66.0, 85.0, obs_id="wall_right_upper"))

    engine.env.add_obstacle(CircularObstacle(center=np.array([48.0, 50.0]), radius=5.0, obs_id="pillar_center"))
    engine.env.add_obstacle(CircularObstacle(center=np.array([78.0, 30.0]), radius=4.0, obs_id="pillar_east"))
    engine.env.add_obstacle(CircularObstacle(center=np.array([18.0, 75.0]), radius=4.0, obs_id="pillar_northwest"))

    # 2. Hazard zone
    engine.env.add_hazard_zone(HazardZone(center=np.array([48.0, 25.0]), radius=8.0, risk_multiplier=2.5))

    # 3. Targets (distributed progressively: near, mid, and far)
    targets_created = 0
    target_id = 1
    while targets_created < target_count:
        tx = rng.uniform(22.0, bounds.x_max - 5.0)
        ty = rng.uniform(bounds.y_min + 5.0, bounds.y_max - 5.0)
        pt = np.array([tx, ty], dtype=np.float64)

        if any(obs.distance_to(pt) < 3.5 for obs in engine.env.obstacles):
            continue

        priority = float(rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], p=[0.3, 0.3, 0.2, 0.1, 0.1]))
        req_energy = float(rng.uniform(3.0, 8.0))
        engine.env.add_target(TaskTarget(
            task_id=target_id,
            position=pt,
            priority=priority,
            required_energy=req_energy,
            satisfaction_radius=2.5
        ))
        targets_created += 1
        target_id += 1


def build_static_scenario(config: SimulationConfig) -> SimulationEngine:
    """Builds pure static baseline environment."""
    engine = SimulationEngine(config)
    populate_base_world(engine, target_count=max(15, int(config.agent_count * 1.5)))
    engine.initialize_swarm(config.agent_count)
    return engine
