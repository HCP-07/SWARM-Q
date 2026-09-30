"""Adaptive parameter controller based on diversity, collision risk, and stagnation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from configs.config import OptimizationParameters, FitnessWeightsConfig


@dataclass
class SwarmAdaptationState:
    """Live state of adaptation parameters for an agent."""
    beta: float                           # Current QPSO contraction-expansion coefficient
    diversity: float                      # Measured local swarm spatial diversity
    stagnant_ticks: int = 0               # Consecutive ticks without fitness improvement
    mutation_triggered: bool = False      # Whether Cauchy mutation is active for current tick
    risk_escalation: float = 1.0          # Multiplier on safety penalties
    neighbor_weight_discount: float = 1.0 # 0.0 during complete comm blackout, 1.0 normal
    collision_weight_scale: float = 1.0   # Alias for risk_escalation multiplier


class SwarmAdaptationController:
    """Implements explicit, documented adaptation rules for the decentralized swarm."""

    def __init__(self, opt_cfg: OptimizationParameters, base_weights: FitnessWeightsConfig):
        self.opt_cfg = opt_cfg
        self.base_weights = base_weights
        self._current_state = SwarmAdaptationState(
            beta=opt_cfg.beta_initial,
            diversity=opt_cfg.diversity_threshold
        )

    def compute_local_diversity(self, agent_pos: np.ndarray, neighbor_positions: List[np.ndarray]) -> float:
        """Calculates spatial dispersion of the agent and its immediate communicating neighbors."""
        if not neighbor_positions:
            return self.opt_cfg.diversity_threshold * 1.5  # Ample diversity if isolated

        cluster = np.array([agent_pos] + neighbor_positions)
        center = np.mean(cluster, axis=0)
        distances = np.linalg.norm(cluster - center, axis=1)
        return float(np.mean(distances))

    def update_parameters(
        self,
        current_pos: np.ndarray,
        neighbor_states: List[Any],
        obstacles: List[Any],
        current_best_fitness: float
    ) -> SwarmAdaptationState:
        """Convenience adapter evaluating rules and returning updated SwarmAdaptationState."""
        neighbor_pos = [nb.position for nb in neighbor_states if hasattr(nb, "position")]
        min_obs = min([obs.distance_to(current_pos) for obs in obstacles], default=50.0) if obstacles else 50.0
        min_ag = min([float(np.linalg.norm(current_pos - p)) for p in neighbor_pos], default=50.0) if neighbor_pos else 50.0
        comm_on = any(getattr(nb, "communication_status", True) for nb in neighbor_states) if neighbor_states else True

        updated_state, _ = self.update_adaptation(
            current_state=self._current_state,
            agent_pos=current_pos,
            neighbor_positions=neighbor_pos,
            current_fitness=current_best_fitness,
            best_fitness=current_best_fitness,
            min_obstacle_dist=float(min_obs),
            min_agent_dist=float(min_ag),
            comm_active=comm_on
        )
        updated_state.collision_weight_scale = updated_state.risk_escalation
        self._current_state = updated_state
        return updated_state

    def update_adaptation(
        self,
        current_state: SwarmAdaptationState,
        agent_pos: np.ndarray,
        neighbor_positions: List[np.ndarray],
        current_fitness: float,
        best_fitness: float,
        min_obstacle_dist: float,
        min_agent_dist: float,
        comm_active: bool,
        critical_safety_margin: float = 3.0
    ) -> Tuple[SwarmAdaptationState, FitnessWeightsConfig]:
        """Evaluates all 6 adaptation rules and returns updated state and dynamic weights."""
        new_state = SwarmAdaptationState(
            beta=current_state.beta,
            diversity=current_state.diversity,
            stagnant_ticks=current_state.stagnant_ticks,
            mutation_triggered=False,
            risk_escalation=1.0,
            neighbor_weight_discount=1.0
        )

        # -------------------------------------------------------------
        # RULE 1: Swarm Diversity Feedback -> Exploration Control
        # If diversity drops below threshold, agents are clustering dangerously;
        # increase beta to force broader quantum potential search.
        # -------------------------------------------------------------
        diversity = self.compute_local_diversity(agent_pos, neighbor_positions)
        new_state.diversity = diversity

        if diversity < self.opt_cfg.diversity_threshold:
            new_state.beta = min(self.opt_cfg.beta_max, current_state.beta * 1.15)
        else:
            # Anneal back toward exploitation when diversity is healthy
            new_state.beta = max(self.opt_cfg.beta_min, current_state.beta * 0.96)

        # -------------------------------------------------------------
        # RULE 2: Collision Risk Escalation
        # If distance to any neighbor or obstacle approaches safety buffer,
        # escalate collision weights exponentially.
        # -------------------------------------------------------------
        min_dist = min(min_obstacle_dist, min_agent_dist)
        if min_dist < critical_safety_margin:
            deficit = critical_safety_margin - min_dist
            new_state.risk_escalation = float(1.0 + np.exp(deficit / 0.8))
        else:
            new_state.risk_escalation = 1.0

        # -------------------------------------------------------------
        # RULE 3: Stagnation Detection -> Cauchy Mutation
        # If best fitness has not improved for stagnation_limit ticks,
        # trigger heavy-tailed Cauchy mutation to escape local minima.
        # -------------------------------------------------------------
        if current_fitness >= best_fitness - 1e-4:
            new_state.stagnant_ticks = current_state.stagnant_ticks + 1
        else:
            new_state.stagnant_ticks = 0

        if new_state.stagnant_ticks >= self.opt_cfg.stagnation_limit:
            new_state.mutation_triggered = True
            new_state.stagnant_ticks = 0  # Reset counter after triggering mutation

        # -------------------------------------------------------------
        # RULE 5: Communication Dropout Adaptation
        # If communication is lost, discount neighbor influence to 0.0
        # (rely purely on local best and local sensors).
        # -------------------------------------------------------------
        if not comm_active:
            new_state.neighbor_weight_discount = 0.0
        else:
            new_state.neighbor_weight_discount = 1.0

        # Construct adapted fitness weights
        adapted_weights = FitnessWeightsConfig(
            w_distance=self.base_weights.w_distance,
            w_energy=self.base_weights.w_energy,
            w_collision=self.base_weights.w_collision * new_state.risk_escalation,
            w_obstacle=self.base_weights.w_obstacle * new_state.risk_escalation,
            w_conflict=self.base_weights.w_conflict,
            w_delay=self.base_weights.w_delay,
            w_communication=self.base_weights.w_communication * new_state.neighbor_weight_discount
        )
        adapted_weights.validate()

        return new_state, adapted_weights

    def apply_cauchy_mutation(self, waypoints: List[np.ndarray], rng: np.random.Generator) -> List[np.ndarray]:
        """Applies heavy-tailed Cauchy perturbation to waypoints to dislodge stuck particles."""
        mutated: List[np.ndarray] = []
        for wp in waypoints:
            # Standard Cauchy distribution: standard Cauchy = standard normal / standard normal
            cauchy_noise = rng.standard_cauchy(size=2) * self.opt_cfg.cauchy_scale
            # Bound noise to avoid extreme outliers
            cauchy_noise = np.clip(cauchy_noise, -5.0, 5.0)
            mutated.append(wp + cauchy_noise)
        return mutated
