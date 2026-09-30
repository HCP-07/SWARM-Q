# Swarm Intelligence Engineering Progression & Evaluation Log

This document records the empirical progression of the **Adaptive Decentralized Swarm Optimization (ADSO)** framework, tracking architectural modifications, mathematical derivations, and observed phenomena across validation iterations.

---

## Attempt 1: Baseline Architecture & Decentralized Formulation

### What Was Built:
- Initialized core decentralized agent representation (`AgentState`), multi-objective fitness evaluator, continuous 2D environment, and initial QPSO delta potential well candidate trajectory generator.
- Developed the baseline `SafetyValidator` with boundary, kinematic, obstacle, and inter-agent separation verification.

### Empirical Observations:
- **Test Suite Results**: 23 passed, 3 failed.
- **Observed Defect 1 (`test_trajectory_repair_deflects_obstacle`)**: The trajectory repair module failed to deflect around circular obstacles (`is_safe=False`). Inspection revealed that for points inside circular obstacles ($\text{dist} < 0$), `closest_point` returned the interior query point itself, causing `point - closest` to be $\approx 0$ and producing degenerate outward normal vectors $[1.0, 0.0]$.
- **Observed Defect 2 (`test_swarm_scalability[100]`)**: 428 hard collision violations were logged during large swarm initialization. Because initial placement relied on rejection sampling in a restricted spawn box, agents $N > 30$ fell back to an unverified grid that overlapped existing agents at $t=0$.
- **Observed Defect 3 (`latency`)**: Initial tick $t=0$ took $434\text{ ms}$ because all agents simultaneously cold-started trajectory generation. Pairwise inter-agent verification took $\mathcal{O}(N^2)$ scalar Python loops ($4950$ float operations per tick for $N=100$).

---

## Attempt 2: Analytical Distance Fields, Tangential Detours & Vectorized Invariant Checking

### What Changed and Why:
1. **Analytical Obstacle Distance Fields**:
   - Corrected `CircularObstacle.closest_point` to return the true boundary projection $\mathbf{c} + \frac{\mathbf{p} - \mathbf{c}}{\|\mathbf{p} - \mathbf{c}\|} r$.
   - Redesigned `TrajectoryRepair` to synthesize proactive tangential bypass curves ($\mathbf{u} = \alpha \hat{\mathbf{d}} + \beta \hat{\mathbf{t}} + \gamma \hat{\mathbf{n}}_{\text{out}}$) followed by kinematic acceleration and velocity profile smoothing.
2. **Deterministic Collision-Free Spawn Pool**:
   - Replaced uniform rejection sampling with a deterministic geometric grid pool across the western arena corridor with inter-slot spacing $\ge 4.0\text{ m} > d_{\min} = 3.0\text{ m}$.
3. **Vectorized Invariant Verification**:
   - Vectorized pairwise distance calculation using NumPy broadcasted matrix norms:
     $$\mathbf{D}_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|_2, \quad i < j$$
     Reducing verification overhead from $18\text{ ms}$ to $0.08\text{ ms}$ ($225\times$ speedup).

### Empirical Observations:
- `test_trajectory_repair_deflects_obstacle` passed cleanly.
- `test_swarm_scalability[10]` and `[25]` passed with $0$ collisions.
- However, for $N=50$ and $N=100$, small collision counts ($27$ and $134$ counts) occurred during active navigation at ticks $t \ge 7$. Measurements revealed inter-agent separation dipped slightly to $2.965\text{ m} < 3.000\text{ m}$ (a $3.5\text{ cm}$ deficit).

---

## Attempt 3: Reciprocal Velocity Obstacles (RVO) & Physical Braking Distance Envelope

### What Changed and Why:
1. **Root Cause Analysis of $2.965\text{ m}$ Deficit**:
   - When two agents move simultaneously towards each other at $v_{\max} = 3.0\text{ m/s}$, the closing rate is $v_{\text{rel}} = 6.0\text{ m/s}$.
   - Under $a_{\max} = 4.0\text{ m/s}^2$, physical braking to a full stop requires:
     $$d_{\text{stop}} = \frac{v_{\text{rel}}^2}{2 a_{\text{rel}}} = \frac{6^2}{2 \times 8} = 2.25\text{ meters}$$
   - Activating reactive repulsion at $d < 4.5\text{ m}$ was too late: physical deceleration could not stop before penetrating the $3.0\text{ m}$ safety margin.
2. **Mathematical Formulation of Reactive Safety Filter**:
   - Developed `ReactiveSafetyFilter` implementing Control Barrier Function (CBF) constraints with Reciprocal Velocity Obstacle (RVO) half-deficit sharing:
     $$v_{\text{closing, max}} = \max\left(0, \frac{d - d_{\min} - \delta}{2 \Delta t}\right)$$
   - Expanded the proactive avoidance envelope to $d_{\text{avoid}} = d_{\min} + 3.2\text{ m} = 6.2\text{ m}$, providing ample distance for smooth deceleration and orthogonal deflection.
   - Added 3-pass multi-body contact relaxation in `execute_motion_step` to prevent crowding displacement cascades.

### Empirical Observations:
- **Zero Hard Collisions**: `test_swarm_scalability` for both 50 agents and 100 agents achieved **0 hard collisions** across all ticks!
- All 26 unit, integration, stress, and determinism tests passed with $100\%$ success rate.

---

## Attempt 4: Multi-Objective Scaling, Progress Incentivization & Full Stress Validation

### What Changed and Why:
1. **Diagnosis of Premature Inactivity**:
   - In early full-mission runs, agents moved sluggishly because kinetic energy consumption ($f_{\text{energy}} \propto \text{total travel}$) penalized forward motion by $+0.091$, while distance improvement only yielded $-0.007$ due to large arena diagonal normalization ($141.4\text{ m}$). Standing still had a superior fitness ($0.19$) compared to moving ($0.228$)!
2. **Reformulation of Objective Metrics**:
   - Replaced raw path length energy with maneuver acceleration jitter penalty $f_{\text{energy}} = \frac{\sum \|\mathbf{a}_k\|}{K a_{\max}}$, ensuring constant-velocity cruise incurred $0.0$ penalty.
   - Introduced progress shortfall penalty $f_{\text{delay}} = 1.0 - \frac{\Delta d}{v_{\max} H \Delta t}$, heavily penalizing stagnant or sluggish behavior.
   - Distributed targets progressively ($x \in [22, 95]$) so near, mid, and far tasks could be serviced sequentially.

### Empirical Observations:
- In full stress mission runs ($15$ agents, $120$ ticks, moving obstacles, communication blackout, and abrupt agent failure):
  - Task completion jumped from $0\%$ to **$36.4\%+$** (sustained mission throughput).
  - Hard collisions remained strictly **$0$** throughout all perturbations.
  - Surviving agents detected neighbor failure at tick $45$ and successfully serviced orphan tasks through local auction re-bidding.

---

## Attempt 5: Comprehensive Master Modification Plan Execution & Hardening

### What Changed and Why:
1. **Security & Secrets Hardening**:
   - Permanently deleted active `.env` file containing external third-party API credentials.
   - Hardened `.gitignore` and `.dockerignore` against `.env*`, `*.pem`, `*.key`, `*.zip`, `.venv/`, `node_modules/`, `__MACOSX/`, `.DS_Store`, `._*`.
   - Created clean `.env.example` with empty `ADSO_*` placeholders.
   - Eliminated legacy repository remnants and external simulator artifacts (`backend/`, `frontend/`).
2. **Architecture Standardization to AD-QPSO**:
   - Unified naming across codebase, benchmarks, and docs to **AD-QPSO** (`PROPOSED`) and Classical PSO (`BASELINE`).
   - Implemented quantum delta-potential well sampling with Mean Best Position ($\mathbf{mbest}$) and smooth sigmoid conflict penalty in `src/optimization/qpso.py`.
   - Added backwards-compatible shim in `src/optimization/qpso_engine.py`.
3. **Decentralized Staggered Replanning**:
   - Introduced dynamic replanning scheduler in `src/simulation/engine.py` using `REPLAN_THRESHOLD = 5` and selective agent quotas (`max_replans = max(2, N // 4)`) to eliminate tick horizon clustering.
4. **Safety Layer & Control Barrier Functions**:
   - Integrated configurable `avoidance_margin = 3.2` into `SafetyConstraintsConfig`, `SafetyValidator`, and `TrajectoryRepair`.
   - Added `repair_trajectory()` adapter supporting active tangential APF detour synthesis and smooth emergency braking.
5. **Testing & Validation Hardening**:
   - Added dedicated security unit test `test_security.py` (scanning for `.env`, secrets regexes, restricted CORS).
   - Added `test_latency_budget.py` (strict SLA pass/fail validation).
   - Added `test_metrics_bounds.py` (validating mathematical range invariants).
   - Added `test_docs_consistency.py` (enforcing terminology and lack of legacy terms).
   - Verified scalability stress tests across 10, 25, 50, and 100 agents with bounded CPU execution.
6. **Empirical Results & Deliverables**:
   - Generated `results/benchmark.csv`, `results/ablation.csv`, `results/benchmark_stats.csv` (10 seeds, Mean ± Std), `results/profile.txt` (cProfile call-tree), `metrics.json`, `summary.json`, and 7 publication-grade PNG charts.
   - Built modular `server/` FastAPI application with real-time web dashboard and clean root `server.py` wrapper.
   - Verified 0 hard collisions across all baselines and all ablation stages.

