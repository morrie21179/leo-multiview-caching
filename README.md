# LEO Caching Simulation

This repository provides a simulation framework for studying **caching strategies in Low Earth Orbit (LEO) satellite networks**. The system models satellites, ground stations, users, and network links (including or excluding Inter-Satellite Links, ISLs) to simulate realistic video content delivery under storage and latency constraints.

---

## Simulation Variants

This project includes **four simulation scripts**, each representing a different caching architecture or strategy.

---

### Variable Definination

    - storage_constraint_Z: [200, ...., 2000]
    - VIEWS_PER_VIDEO: [4,...., 32]
    - DIBR Constraint: [1, 2, 3, 4] (at main function in each code)

### 1. `main.py` – Baseline Simulation with ISL

This is the most complete simulation model, supporting:
- **LEO satellite mobility**
- **Caching with DIBR view synthesis**
- **Inter-Satellite Links (ISL) for content sharing**
- **Proactive ground-station-assisted cache updates**

#### Key Logic & Functions

- **`run_simulation(Z)`**  
  Runs the full simulation for a given satellite cache size `Z`, including user request generation, caching decisions, latency measurements, and DIBR logic.

- **`get_tau_j(view_j, sat, nearest_gs, hops_to_gs=0)`**  
  Computes the "cost" (latency + hops) of retrieving a requested view. Considers:
  - Local cache (zero cost)
  - Neighboring satellites via ISL
  - Ground station if not cached

- **`calculate_local_popularity(...)`**  
  Uses LFU + recency heuristics to assign popularity scores to cached views for eviction decisions.

- **`calculate_synthesis_benefit(...)`**  
  Determines how many **virtual views** can be generated if a new view is cached (via Depth-Image-Based Rendering). Higher value = more usefulness.

- **`run_phase2_gs_swap(...)`**  
  Attempts to **swap unpopular views** with globally popular views from the ground station if it would improve cache utility (based on synthesis benefit).

- **`cache_content_with_eviction(...)`**  
  Adds a view to the cache. If full, evicts the least valuable one based on a **hybrid score** (local popularity + DIBR potential + global ranking).

- **`initialize_zipf_cache(...)`**  
  Initializes each satellite's cache using a Zipf distribution, mimicking real-world content demand skew.

- **`calculate_request_latency(...)`**  
  Models realistic latency, factoring in:
  - Satellite-to-user downlink
  - ISL hops
  - Ground station fetch delays

- **Visualization tools**
  - `visualize_leo_satellite_movement(...)`: Static snapshot
  - `create_satellite_movement_animation(...)`: Animated LEO orbits

---

### 2. `main_mobile.py` – Non-Cooperative Model (No ISL)

Simulates a disconnected network where:
- **No Inter-Satellite Links (ISL)** are available
- Each satellite operates independently
- Fetches from local cache or ground stations only

#### Changes from `main.py`

- **`get_tau_j(...)`**:  
  Removed ISL logic; only checks local cache or direct ground station link.

- **`calculate_request_latency(...)`**:  
  Simplified latency model without ISL delay.

- **Simulation loop**:
  Still varies storage size `Z`, and runs performance analysis for each setting.

This version is ideal for comparing how ISL affects hit rate, latency, and efficiency.

---

### 3. `main_RFP.py` – Region Feature Prediction (Cooperative Game-Theory)

Implements **predictive caching** using:
- **Ridge regression** to model content preferences in regions
- **Cooperative Area Formation** based on request similarity
- **Game-theoretic cache optimization**

#### Core Components

- **`predict_region_features(...)`**  
  Trains a ridge regression model using recent user requests to estimate **regional feature vectors** (representing preferred content types).

- **`divide_cooperative_areas(...)`**  
  Compares similarity of adjacent satellite regions (via cosine similarity of feature vectors). If high similarity → **group into cooperative caching area**.

- **`run_cooperative_caching_game(...)`**  
  Each satellite within a group **plays a non-cooperative game**:
  - Updates its cache to maximize its own utility
  - Considers benefit to others (shared content diversity)
  - Repeats until equilibrium

- **`get_tau_j(...)`**  
  Modified to first check:
  - Local cache
  - Caches in cooperative group
  - Then ground station

- **`cache_content_with_eviction(...)`**  
  Simplified to use LFU/LRU score, since caching is handled by the cooperative game.

- **`calculate_request_latency(...)`**  
  Includes:
  - Bandwidth sharing
  - Elevation-angle-based throughput models

This model evaluates **learning + cooperation**, particularly effective in clustered urban or demand-skewed regions.

---

### 4. `main_SCA.py` – Spot-Beam & Cooperative MPC Strategy

This strategy involves:
- Simulating **spot beam coverage** via user clustering
- Assigning **non-overlapping blocks** of the most popular views to satellites

#### Core Logic (in `BhandariEtAlAlgorithm` class)

- **`_get_global_popularity()`**  
  Builds a Zipf-based ranking of content across the entire network.

- **`update_cache_placement()`**  
  Steps:
  1. Performs **K-Means++ clustering** on users per satellite
  2. Assigns each satellite a **unique segment** of the most popular content
  3. Ensures **diverse caching** across satellites

- **`get_tau_j(...)`**  
  Enforces:
  - No cache hit from other satellites (no ISL)
  - ISL used **only for routing** to ground

- **`calculate_request_latency(...)`**  
  Models **bandwidth sharing** — satellite downlink is split among all active users

- **`run_simulation(...)`**  
  Calls `update_cache_placement()` periodically during simulation.

### 5.  container_0617.py

Includes clean constants and a well-documented data rate function for
- main.py
- main_mobile.py
- main_SCA.py

### 6. container_RFP.py

Defines communication parameters for RFP-based cooperative caching models

---

## Plotting Tools

### `plot_size.py`  
Generates comparison plots by **cache size**:
- Hit rate
- Latency
- ISL hops
- Miss cost
- Cost

### `plot_view.py`  
Plots results by **number of available views**:
- View synthesis impact on caching performance

### `plot_DIBR.py`  
Explores **Depth-Image-Based Rendering (DIBR)** constraints:
- Synthesizable views
- DIBR-related cost and hit rate

---

