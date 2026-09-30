# Quantum-Inspired Traffic Optimization: Research Experiments

This document explains the experimental methodology and statistical evaluations used to analyze routing metaheuristics in this platform.

---

## 1. Research Principles

 stochastically-driven metaheuristics (PSO, QPSO, IQPSO) are evaluated using three core criteria:
1. **Solution Quality**: The minimum objective fitness achieved.
2. **Convergence Speed**: The number of iterations required to reach a stable near-optimal state.
3. **Robustness**: The consistency of solutions across multiple runs with different random seeds.

We avoid the common pitfall of selecting a single "lucky" run. Instead, every algorithm is run for $R$ independent sessions (typically $R \ge 5$ or $10$), and we analyze the statistical distributions of the final fitness values.

---

## 2. Experimental Indicators

The platform logs and compares:

### Swarm Spatial Diversity
Tracks the physical dispersion of the particles across iterations:
- High diversity indicates the swarm is actively exploring the global search space (wide exploration).
- Rapidly decaying diversity indicates the particles are collapsing toward a common minimum (rapid exploitation).
- Stagnant zero-diversity with poor fitness indicates the swarm has collapsed into a local minimum.

### Convergence Iteration
Identifies the exact iteration $t_{\text{conv}}$ where the fitness reached within 5% of its final optimized value. This helps measure if QPSO or Improved QPSO converges faster than classical PSO.

### Mann-Whitney U test (Hypothesis Verification)
Since metaheuristic fitness distributions are rarely normally distributed (they are often skewed and bounded), parametric tests like the t-test can be scientifically invalid. 

We apply the non-parametric **Mann-Whitney U Test** at significance level $\alpha = 0.05$:
- **Null Hypothesis ($H_0$)**: The final fitness distributions of the baseline algorithm (PSO/QPSO) and the Proposed Improved QPSO are identical.
- **Alternative Hypothesis ($H_1$)**: The Proposed Improved QPSO has a statistically superior (lower) median fitness distribution.
- **Result Interpretation**:
  - If $p < 0.05$, we reject $H_0$. The improvements of Improved QPSO are statistically significant.
  - If $p \ge 0.05$, we fail to reject $H_0$. The difference in performance is within the margin of random fluctuations.

---

## 3. How to Conduct Experiments

1. **Configure Network**: Set your preferred network topology (e.g., Grid, Random, or Scale-free).
2. **Apply Traffic/Incident Constraints**:
   - Set the clock slider to 08:00 or 17:00 to simulate high-congestion peak traffic.
   - Inject roadblocks/accidents on key segments to force dynamic rerouting.
3. **Execute Benchmark**:
   - Set the runs count (e.g., 5 or 10).
   - Click **Run Statistical Benchmark**.
4. **Analyze the Results**:
   - Switch to the **Convergence & Diversity** tab to view the average trajectory.
   - Switch to the **Research Analytics** tab to view the final performance table and the Mann-Whitney U test p-values.
