# Quantum-Inspired Traffic Optimization: REST API Reference

The backend exposes a FastAPI REST API running on port `8000`. Full interactive documentation is available at `http://localhost:8000/docs`.

---

## 1. Graph Endpoints

### Get Active Graph
Returns the current nodes and edges of the active simulation network.
- **URL**: `/api/graph`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "nodes": [{"id": 0, "lat": 37.7749, "lon": -122.4194, "type": "depot", "label": "Jct_0_0"}],
    "edges": [{"source": 0, "target": 1, "length": 0.5, "speed_limit": 50, "capacity": 1500, "flow": 420.0, "current_travel_time": 0.0101, "incidents": []}]
  }
  ```

### Generate Graph
Generates a new spatial network topology and overrides the active state.
- **URL**: `/api/graph/generate`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "topology": "grid",
    "size": 36,
    "spacing_km": 0.5
  }
  ```
  *(Topologies: `"grid"`, `"random"`, `"scale_free"`)*
- **Response**: Returns the new graph payload.

---

## 2. Traffic & Incident Endpoints

### Get Traffic Status
Gets the current clock time and active blockages.
- **URL**: `/api/traffic/status`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "hour": 12.0,
    "active_incidents": []
  }
  ```

### Update Clock Time
Sets the simulation hour of the day to trigger diurnal flow multiplier adjustments.
- **URL**: `/api/traffic/time`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "hour": 8.5
  }
  ```

### Inject Incident
Adds a road blockage or accident onto a directed segment.
- **URL**: `/api/traffic/incident`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "source": 0,
    "target": 1,
    "incident_type": "accident",
    "severity": 0.8,
    "description": "Multi-car collision blocking 2 lanes"
  }
  ```
  *(Incident types: `"accident"`, `"roadwork"`, `"weather"`, `"breakdown"`)*

### Remove Incident
Clears an incident by its ID.
- **URL**: `/api/traffic/incident/{incident_id}`
- **Method**: `DELETE`

### Clear All Incidents
Clears all traffic incidents.
- **URL**: `/api/traffic/incident/clear`
- **Method**: `POST`

---

## 3. Optimization Endpoints

### Shortest Path Routing
Solves shortest path routing between two junctions.
- **URL**: `/api/optimize/shortest-path`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "source": 0,
    "target": 15,
    "algorithm": "iqpso",
    "swarm_size": 30,
    "iterations": 50,
    "weights": null
  }
  ```
  *(Algorithms: `"pso"`, `"qpso"`, `"iqpso"`)*

### Vehicle Routing Problem (VRP) Logistics
Calculates fleet schedules to distribute cargo to customer nodes.
- **URL**: `/api/optimize/vrp`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "depot_id": 0,
    "customer_ids": [5, 12, 18, 22, 28, 30],
    "vehicle_capacity": 100.0,
    "max_vehicles": 4,
    "swarm_size": 40,
    "iterations": 60,
    "weights": null
  }
  ```

---

## 4. Benchmarking Endpoints

### Run Comparative Research Benchmark
Executes PSO, QPSO, and IQPSO concurrently on the active graph, compiles multiple runs, and runs Mann-Whitney hypothesis tests.
- **URL**: `/api/benchmark/run`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "problem_type": "shortest_path",
    "source": 0,
    "target": 15,
    "swarm_size": 30,
    "iterations": 50,
    "runs": 5,
    "weights": null
  }
  ```
  *(Or `"problem_type": "vrp"`)*
