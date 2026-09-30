# Quantum-Inspired Traffic Optimization: Algorithm & Encoding Details

This document explains the route representation and encoding mappings used to translate continuous particle positions into discrete graph routing solutions.

---

## 1. Shortest Path Priority-Based Encoding

A classical challenge of Particle Swarm Optimization is that particles search in continuous space ($\mathbb{R}^D$), whereas routing is a discrete combinatorial problem on graphs.

To solve the Shortest Path (SP) problem from source $s$ to target $d$:
1. The particle position vector $X \in \mathbb{R}^N$ has dimensions equal to the number of nodes in the graph ($N = |V|$).
2. The coordinate $x_i$ represents the **priority** of node $i$.

### Path Reconstruction Algorithm
To decode a position vector $X$ into a path $P$:
1. Initialize the current node $u = s$. Path $P = [s]$. Visited set $V_{\text{visited}} = \{s\}$.
2. While $u \neq d$:
   - Fetch all neighbors of $u$, denoted as $N(u)$.
   - Filter to unvisited neighbors: $N_{\text{unvisited}} = \{v \in N(u) \mid v \notin V_{\text{visited}}\}$.
   - If $N_{\text{unvisited}}$ is empty:
     - **Backtrack**: Remove $u$ from the path $P$.
     - Set $u$ to the new tail of $P$.
     - If $P$ is empty, path search has failed. Break.
   - If $N_{\text{unvisited}}$ is not empty, calculate a priority score for each $v \in N_{\text{unvisited}}$:
     $$\text{Score}(v) = x_v + \lambda \cdot H(v, d)$$
     Where $H(v, d)$ is a target heuristic (inverse distance to destination) to guide the swarm, and $\lambda$ is a weight (default $0.5$).
   - Choose neighbor $v$ with the highest $\text{Score}(v)$.
   - Append $v$ to $P$, add $v$ to $V_{\text{visited}}$, and set $u = v$.

### Repair Mechanism
If the path search fails to connect to $d$ due to graph disconnections, we repair the path using standard Dijkstra's algorithm based on current travel times. However, we add a massive constraint penalty (e.g., $+200.0$) to the fitness value, guiding particles away from invalid node priorities.

---

## 2. Vehicle Routing Problem (VRP) Random Keys Encoding

For a VRP with a central depot (Node 0) and $C$ customer locations:
1. The particle position vector $X \in \mathbb{R}^C$ has dimensions equal to the number of customers.
2. The coordinate $x_i$ represents the "random key" of customer $c_i$.

### Permutation Decoding
To translate the continuous vector $X$ into a sequence:
1. Sort the indices of $X$ in ascending order of their values.
2. This mapping yields a permutation $\pi$ of the customers.
   - *Example*: If $X = [0.8, -1.2, 0.4]$ for customers $[C_1, C_2, C_3]$, sorting gives keys $[-1.2, 0.4, 0.8]$ which corresponds to indices $[1, 2, 0]$. The customer sequence is $\pi = [C_2, C_3, C_1]$.

### Constrained Split Heuristic
To split the sequence $\pi$ into $K$ vehicle routes matching capacity $Q$:
1. Initialize route list $R = []$, current vehicle route $r = []$, and current cargo load $L = 0.0$.
2. For each customer $c \in \pi$:
   - Retrieve customer demand $d_c$.
   - If $L + d_c \le Q$:
     - Append $c$ to current route $r$.
     - Update load: $L = L + d_c$.
   - If $L + d_c > Q$:
     - Close current route: append $r$ to $R$.
     - Start a new route for the next vehicle: $r = [c]$, reset load $L = d_c$.
3. Append any remaining route to $R$.
4. Each vehicle route starts and ends at the depot: $0 \to r_1 \to r_2 \to \dots \to r_n \to 0$.

---

## 3. Dynamic BPR Traffic Congestion Model

To simulate dynamic road conditions, the travel time on each edge $(u, v)$ is calculated using the **Bureau of Public Roads (BPR)** model:

$$T(u, v) = T_0(u, v) \cdot \left( 1 + \beta \left( \frac{F(u, v)}{C(u, v)} \right)^\gamma \right)$$

Where:
- $T_0(u, v) = \frac{\text{Length}(u, v)}{\text{SpeedLimit}(u, v)}$ is the free-flow travel time.
- $F(u, v)$ is the current flow volume (vehicles/hour).
- $C(u, v)$ is the edge capacity (vehicles/hour).
- $\beta = 0.15$ and $\gamma = 4.0$ are calibration constants.

### Incident Impact
When an accident or road block is injected:
- The capacity is reduced: $C_{\text{effective}} = C \cdot (1 - \text{Severity})$.
- The speed limit is reduced: $S_{\text{effective}} = S \cdot (1 - 0.7 \cdot \text{Severity})$.
- The travel time spikes, shifting the routing selections dynamically.
