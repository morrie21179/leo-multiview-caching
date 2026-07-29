# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

# The user's simulation environment is assumed to have scikit-learn.
# K-Means++ is explicitly mentioned in the paper for clustering[cite: 10, 41, 139].
from sklearn.cluster import KMeans

from container_0617 import User, Satellite, GroundStation
from container_0617 import generate_zipf_distribution, calculate_rate_mbps, SPEED_OF_LIGHT_KM_S
# from mpl_toolkits.basemap import Basemap

# --- Simulation Parameters ---
video_size_per_view_mb = 6
Total_timeslot = 240

# --- Video Content Structure & Popularity Parameters ---
NUM_VIDEOS = 2500
VIEWS_PER_VIDEO = 16  # 4, 8, 12, 16, 20, 24, 28, 32
TOTAL_VIEWS = NUM_VIDEOS * VIEWS_PER_VIDEO
ZIPF_ALPHA = 0.8

# --- Poisson Process Parameters ---
arrival_rate_lambda = 60
departure_probability = 0.15

# Function to get the cost of fetching/accessing a single view
# NOTE: This version disables ISL for cache hits, as requested.
def get_tau_j(view_j, sat, nearest_gs, hops_to_gs=0):
    cost_serving, cost_isl_hop, cost_gs_fetch, cost_miss_penalty = 5, 10, 20, 50
    if sat.is_view_cached(view_j):
        return cost_serving
    # ISL for cache hits is disabled by commenting out this block.
    # elif view_j in sat.neighbor_caches:
    #     return cost_isl_hop + cost_serving
    elif nearest_gs:
        # ISL cost is still correctly applied for routing to the GS during a cache miss.
        return cost_miss_penalty + cost_gs_fetch + (hops_to_gs * cost_isl_hop) + cost_serving
    else:
        # This case implies no GS connection is possible.
        print(f"Warning: No GS connection for view {view_j} on satellite {sat.sat_name}.")
        exit()
        return float('inf')

def calculate_request_latency(sat, user, time, view_sets, nearest_gs, hops_to_gs, sat_table, ordered_names, active_user_list):
    """
    Calculates the real latency for a user's request based on the DP plan.
    Latency = max(latency of each transmitted view).
    
    MODIFICATION: Now considers bandwidth sharing among all users being served by the satellite.
    The satellite's total bandwidth is equally distributed among all serving users.
    """
    view_size_mbit = user.video_size_mb * 8
    max_latency = 0.0

    # Calculate number of users being served by this satellite
    # Calculate number of users being served by this satellite from active_user_list
    num_serving_users = sum(1 for user in active_user_list if user.sat == sat)
    if num_serving_users == 0:
        return 0  # No users being served

    # Calculate Downlink (Satellite -> User) latency components with bandwidth sharing
    dist_s_u = sat.distance_to_user(time, user)
    prop_s_u = dist_s_u / SPEED_OF_LIGHT_KM_S
    
    # Total satellite downlink capacity (Mbps) - adjust this value based on your satellite specs
    total_downlink_capacity_mbps = 650
    
    # Effective data rate per user (equal bandwidth sharing)
    effective_rate_s_u_mbps = total_downlink_capacity_mbps / num_serving_users
    
    if effective_rate_s_u_mbps == 0: 
        return float('inf')
    
    # Transmission time using the effective (shared) data rate
    tx_s_u = view_size_mbit / effective_rate_s_u_mbps
    
    # Calculate latency for each transmitted view based on its source
    for view in view_sets.get('local', set()):
        latency = prop_s_u + tx_s_u
        if latency > max_latency: 
            max_latency = latency

    for view in view_sets.get('isl', set()):
        # ISL (Inter-Satellite Link) transmission
        isl_prop_latency = 1000 / SPEED_OF_LIGHT_KM_S # Avg distance for ISL
        
        # For ISL, we might also want to consider sharing if multiple satellites
        # are requesting data simultaneously, but for simplicity, we'll use full ISL capacity
        isl_tx_latency = view_size_mbit / (sat.isl_data_rate_gbps * 1000)
        
        # Total latency: ISL transmission + downlink to user (with bandwidth sharing)
        latency = isl_prop_latency + isl_tx_latency + prop_s_u + tx_s_u
        if latency > max_latency: 
            max_latency = latency

    if nearest_gs:
        for view in view_sets.get('gs', set()):
            # Find the satellite actually connecting to the GS
            connected_sat = sat
            if hops_to_gs > 0:
                # This is a simplification; a real system would trace the exact path.
                # We'll use the original satellite's position as a proxy.
                pass 
            
            # Ground station to satellite uplink
            dist_gs_s = nearest_gs.calculate_distance_to_satellite(
                connected_sat.lat[time], connected_sat.lon[time], connected_sat.alt[time]
            )
            prop_gs_s = dist_gs_s / SPEED_OF_LIGHT_KM_S
            
            # Uplink rate from ground station (typically not shared among users on the same satellite)
            rate_gs_s_mbps = 2500 
            if rate_gs_s_mbps == 0: 
                continue
            tx_gs_s = view_size_mbit / rate_gs_s_mbps
            
            # ISL hop latency if data needs to be forwarded through other satellites
            isl_hop_latency = hops_to_gs * (
                (1000 / SPEED_OF_LIGHT_KM_S) + 
                (view_size_mbit / (sat.isl_data_rate_gbps * 1000))
            )
            
            # Total latency: GS uplink + ISL forwarding + downlink to user (with bandwidth sharing)
            latency = prop_gs_s + tx_gs_s + isl_hop_latency + prop_s_u + tx_s_u
            if latency > max_latency: 
                max_latency = latency

    return max_latency

# # --- Latency Calculation Function ---
# def calculate_request_latency(sat, user, time, view_sets, nearest_gs, hops_to_gs, sat_table, ordered_names):
#     view_size_mbit = user.video_size_mb * 8
#     max_latency = 0.0
#     dist_s_u = sat.distance_to_user(time, user)
#     prop_s_u = dist_s_u / SPEED_OF_LIGHT_KM_S
#     rate_s_u_mbps = 650
#     if rate_s_u_mbps == 0: return float('inf')
#     tx_s_u = view_size_mbit / rate_s_u_mbps

#     for view in view_sets.get('local', set()):
#         latency = prop_s_u + tx_s_u
#         if latency > max_latency: 
#             max_latency = latency

#     # This loop will not run if V_ISL is empty (as it is in this version for hits)
#     for view in view_sets.get('isl', set()):
#         isl_prop_latency = 1000 / SPEED_OF_LIGHT_KM_S
#         isl_tx_latency = view_size_mbit / (sat.isl_data_rate_gbps * 1000)
#         latency = isl_prop_latency + isl_tx_latency + prop_s_u + tx_s_u
#         if latency > max_latency: max_latency = latency

#     if nearest_gs:
#         for view in view_sets.get('gs', set()):
#             connected_sat = sat
#             dist_gs_s = nearest_gs.calculate_distance_to_satellite(connected_sat.lat[time], connected_sat.lon[time], connected_sat.alt[time])
#             prop_gs_s = dist_gs_s / SPEED_OF_LIGHT_KM_S
#             rate_gs_s_mbps = 2500
#             if rate_gs_s_mbps == 0: continue
#             tx_gs_s = view_size_mbit / rate_gs_s_mbps
#             isl_hop_latency = hops_to_gs * ( (1000 / SPEED_OF_LIGHT_KM_S) + (view_size_mbit / (sat.isl_data_rate_gbps * 1000)) )
#             latency = prop_gs_s + tx_gs_s + isl_hop_latency + prop_s_u + tx_s_u
#             if latency > max_latency: max_latency = latency
#     return max_latency

def visualize_leo_satellite_movement(satellite_table, ground_station_list, user_list, timeslot_to_visualize=0):
    fig = plt.figure(figsize=(15, 10))
    m = Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, resolution='c')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.fillcontinents(color='lightgray', lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')
    m.drawparallels(np.arange(-80, 81, 20), labels=[1,0,0,0], fontsize=8)
    m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1], fontsize=8)
    gs_lats, gs_lons = [gs.lat for gs in ground_station_list], [gs.lon for gs in ground_station_list]
    gs_x, gs_y = m(gs_lons, gs_lats)
    m.scatter(gs_x, gs_y, c='red', s=80, marker='^', label='Ground Stations', edgecolors='black', linewidths=1)
    user_lats, user_lons = [user.lat for user in user_list], [user.lon for user in user_list]
    user_x, user_y = m(user_lons, user_lats)
    m.scatter(user_x, user_y, c='green', s=10, marker='s', label='Users', edgecolors='darkgreen', linewidths=0.5, alpha=0.7)
    sat_colors = plt.cm.tab10(np.linspace(0, 1, len(satellite_table)))
    first_satellite = True
    for i, (sat_name, sat) in enumerate(satellite_table.items()):
        try:
            if timeslot_to_visualize < len(sat.lat):
                sat_lat, sat_lon = sat.lat.iloc[timeslot_to_visualize], sat.lon.iloc[timeslot_to_visualize]
                sat_x, sat_y = m(sat_lon, sat_lat)
                label = 'LEO Satellites' if first_satellite else None
                m.scatter(sat_x, sat_y, c=[sat_colors[i]], s=120, marker='o', label=label, edgecolors='black', linewidths=1, alpha=0.8)
                plt.annotate(sat_name, (sat_x, sat_y), xytext=(5, 5), textcoords='offset points', fontsize=6, color='blue', fontweight='bold')
                first_satellite = False
        except Exception as e:
            print(f"Error plotting satellite {sat_name}: {e}")
            continue
    plt.legend(loc='lower left')
    plt.title(f'LEO Satellite Network, Ground Stations, and Users at Timeslot {timeslot_to_visualize}\n'
              f'Satellites: {len(satellite_table)}, Ground Stations: {len(ground_station_list)}, Users: {len(user_list)}',
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def initialize_zipf_cache(satellite, storage_constraint, num_videos, views_per_video, zipf_alpha, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed + hash(satellite.sat_name))
    satellite.cache_state.clear()
    video_pop_dist = generate_zipf_distribution(num_videos, zipf_alpha)
    angle_pop_dist = generate_zipf_distribution(views_per_video, zipf_alpha)
    video_indices, view_angle_indices = np.arange(num_videos), np.arange(views_per_video)
    views_to_cache = set()
    while len(views_to_cache) < storage_constraint:
        video_id = np.random.choice(video_indices, p=video_pop_dist)
        angle_id = np.random.choice(view_angle_indices, p=angle_pop_dist)
        global_view_id = video_id * views_per_video + angle_id
        views_to_cache.add(global_view_id)
    satellite.cache_state = views_to_cache

def cache_content_with_eviction(satellite, content_id, timeslot, num_videos, views_per_video, zipf_alpha):
    if satellite.is_view_cached(content_id):
        satellite.last_access_time[content_id] = timeslot
        satellite.access_frequency[content_id] = satellite.access_frequency.get(content_id, 0) + 1
        return
    if len(satellite.cache_state) < satellite.storage_constraint_Z:
        satellite.cache_view(content_id)
        satellite.last_access_time[content_id] = timeslot
        satellite.access_frequency[content_id] = 1
        return
    video_pop_dist = generate_zipf_distribution(num_videos, zipf_alpha)
    popularity_scores = {}
    for view_id in satellite.cache_state:
        video_id = view_id // views_per_video
        global_pop_score = video_pop_dist[video_id] if video_id < len(video_pop_dist) else 0
        frequency_score = satellite.access_frequency.get(view_id, 0)
        last_access = satellite.last_access_time.get(view_id, 0)
        recency_score = 1.0 / (timeslot - last_access + 1)
        popularity_scores[view_id] = (0.5 * global_pop_score) + (0.3 * frequency_score) + (0.2 * recency_score)
    if popularity_scores:
        view_to_evict = min(popularity_scores, key=popularity_scores.get)
        satellite.evict_view(view_to_evict)
        if view_to_evict in satellite.last_access_time: del satellite.last_access_time[view_to_evict]
        if view_to_evict in satellite.access_frequency: del satellite.access_frequency[view_to_evict]
    satellite.cache_view(content_id)
    satellite.last_access_time[content_id] = timeslot
    satellite.access_frequency[content_id] = 1

################################################################################
#          NEW ALGORITHM FROM "EFFICIENT CONTENT CACHING..." PAPER             #
################################################################################

class BhandariEtAlAlgorithm:
    """
    Implementation of the caching strategy from the paper "Efficient Content Caching
    for Delivery Time Minimization in the LEO Satellite Networks" by Bhandari et al.[cite: 3].

    This implementation adapts the paper's concepts to the existing simulation framework.
    It focuses on two main contributions identified in the paper:
    1.  User Clustering (Spot Beam Formation): Uses K-Means++ to group users based on their
        geographic location, simulating the paper's spot beam formation (Algorithm 1)[cite: 10, 41, 153].
    2.  Caching Strategy: Implements a cooperative Most Popular Caching (MPC) strategy.
        The paper concludes that MPC is highly effective. To avoid redundancy where all
        satellites cache the same content, this version distributes unique, contiguous
        blocks of the most popular content across the constellation, creating a diverse
        and efficient network-wide cache.
    """
    def __init__(self, satellites, all_users, num_videos, views_per_video,
                 num_spot_beams=8, zipf_alpha=0.8):
        self.satellites = satellites
        self.all_users = {user.user_id: user for user in all_users}
        self.N = num_videos
        self.views_per_video = views_per_video
        self.total_views = num_videos * views_per_video
        # Number of spot beams per satellite, from paper Table I [cite: 281]
        self.M = num_spot_beams
        self.zipf_alpha = zipf_alpha
        # Pre-calculate the global popularity list for efficiency
        self.globally_popular_views = self._get_global_popularity()
        print(f"Algorithm initialized with M={self.M} spot beams per satellite and cooperative MPC.")

    def _get_global_popularity(self):
        """
        Generates a single, global popularity ranking for all viewable content based on Zipf's law.
        Returns a list of view IDs sorted from most popular to least popular.
        """
        view_popularity = {}
        video_pop_dist = generate_zipf_distribution(self.N, self.zipf_alpha)
        angle_pop_dist = generate_zipf_distribution(self.views_per_video, self.zipf_alpha)

        for video_id in range(self.N):
            for angle_id in range(self.views_per_video):
                global_view_id = video_id * self.views_per_video + angle_id
                popularity_score = video_pop_dist[video_id] * angle_pop_dist[angle_id]
                view_popularity[global_view_id] = popularity_score

        # Sort all views by their popularity score in descending order
        sorted_views = sorted(view_popularity.items(), key=lambda x: x[1], reverse=True)
        return [view[0] for view in sorted_views]

    def update_cache_placement(self, timeslot):
        """
        Executes the two-stage optimization process adapted from the paper.
        This method should be called periodically to update the caching strategy.
        """
        # --- Stage 1: User Clustering into Spot Beams (Simulating Algorithm 1 [cite: 153]) ---
        for sat in self.satellites.values():
            if len(sat.serving_users) < self.M:
                continue # Not enough users to form M clusters

            user_locations = []
            for user_id in sat.serving_users:
                user = self.all_users.get(user_id)
                if user:
                    user_locations.append([user.lat, user.lon])

            if not user_locations:
                continue

            # K-Means++ is used as specified in the paper's Algorithm 1 [cite: 139, 41]
            kmeans = KMeans(n_clusters=self.M, init='k-means++', n_init='auto', random_state=timeslot)
            kmeans.fit(np.array(user_locations))
            # In a more complex simulation, these cluster assignments would inform resource allocation.
            # Here, we complete the step as required by the paper's methodology.

        # --- Stage 2: Cooperative Most Popular Caching (MPC) ---
        # The paper concludes MPC is the most effective caching strategy.
        # We implement a cooperative version to ensure cache diversity across the network.
        sat_names_ordered = sorted(list(self.satellites.keys()))
        content_idx = 0
        total_unique_views = len(self.globally_popular_views)

        # This loop distributes the globally popular content across all satellites.
        for sat_name in sat_names_ordered:
            sat = self.satellites[sat_name]
            sat.cache_state.clear()
            storage_capacity = sat.storage_constraint_Z

            # Each satellite caches a unique, contiguous block of popular content
            views_to_add = []
            while len(views_to_add) < storage_capacity:
                # If we've cached all unique content, loop back to the start.
                # This ensures caches are full even if total storage > total content.
                if content_idx >= total_unique_views:
                    content_idx = 0

                view_to_cache = self.globally_popular_views[content_idx]
                views_to_add.append(view_to_cache)
                content_idx += 1 # Move to the next most popular item for the next slot.

            sat.cache_state = set(views_to_add)

        # After updating all caches, reset the access stats for the new content
        for sat in self.satellites.values():
             sat.access_frequency = {view_id: 1 for view_id in sat.cache_state}
             sat.last_access_time = {view_id: timeslot for view_id in sat.cache_state}

def run_simulation(storage_constraint_Z):
    print(f"\n{'='*60}")
    print(f"RUNNING SIMULATION WITH STORAGE CONSTRAINT: {storage_constraint_Z}")
    print(f"{'='*60}")
    
    video_indices, view_angle_indices = list(range(NUM_VIDEOS)), list(range(VIEWS_PER_VIDEO))

    ##################################################
    #          INITIALIZE SYSTEM COMPONENTS          #
    ##################################################

    users_df = pd.read_csv('data/users.csv')
    all_users = [User(row['id'], row['lat'], row['lon'], row['x'], row['y'], row['z'], video_size_mb=video_size_per_view_mb) for _, row in users_df.iterrows()]
    all_users.sort()
    initial_active_user_count = 150
    active_user_list, inactive_user_list = all_users[:initial_active_user_count], all_users[initial_active_user_count:]

    satellite_table = {}
    for file in os.listdir('data/starlink118/satellite_trace'):
        if file.endswith('.csv'):
            satellite_data = pd.read_csv(f'data/starlink118/satellite_trace/{file}')
            satellite_name = file.split('_')[0]
            sat = Satellite(satellite_name, satellite_data, storage_constraint_Z=storage_constraint_Z, total_views=TOTAL_VIEWS, view_size_mb=video_size_per_view_mb)
            satellite_table[satellite_name] = sat

    random_seed = 42
    for sat_name, sat in satellite_table.items():
        sat.access_frequency, sat.last_access_time = {}, {}
        initialize_zipf_cache(sat, storage_constraint_Z, NUM_VIDEOS, VIEWS_PER_VIDEO, ZIPF_ALPHA, random_seed)
        for view_id in sat.cache_state:
            sat.access_frequency[view_id], sat.last_access_time[view_id] = 1, 0

    ground_station_list = []
    ground_station_locations = [
        {"id": "GS_EU_UK_GOON", "name": "Goonhilly", "lat": 50.05, "lon": -5.18},     # United Kingdom
        {"id": "GS_AF_MZ", "name": "Matola", "lat": -25.92, "lon": 32.42},            # MZ
    ]

    for _, row in pd.DataFrame(ground_station_locations).iterrows():
        gs = GroundStation(row['id'], row['name'], row['lat'], row['lon'], total_views=TOTAL_VIEWS, view_size_mb=video_size_per_view_mb)
        ground_station_list.append(gs)

    # Initialize the new algorithm from the paper
    bhandari_algorithm = BhandariEtAlAlgorithm(
        satellites=satellite_table,
        all_users=all_users,
        num_videos=NUM_VIDEOS,
        views_per_video=VIEWS_PER_VIDEO,
        zipf_alpha=ZIPF_ALPHA
    )

    # Initialize Cost and Metric Trackers
    satellite_costs = {sat_name: 0 for sat_name in satellite_table.keys()}
    timeslot_costs = []
    cache_hit_stats = {sat_name: {'hits': 0, 'misses': 0} for sat_name in satellite_table.keys()}
    total_requests_over_simulation = 0
    average_latency_per_timeslot = []
    average_hops_per_timeslot = []
    total_isl_hops_over_simulation = 0
    total_requests_with_isl_over_simulation = 0
    total_dibr_synthesis_cost = 0
    network_cost_breakdown = {'miss_penalty': 0, 'gs_fetch': 0, 'isl': 0, 'serving': 0}

    ########################################################################################################
    ####################################### MAIN SIMULATION LOOP ###########################################
    ########################################################################################################

    for i in range(Total_timeslot):
        if i % 80 == 0:
            print(f'=========== Time slot {i:03d} | Active Users: {len(active_user_list)} ===========')
        number_of_requests_this_slot, latencies_this_timeslot, hops_this_timeslot, requests_with_isl_this_timeslot = 0, [], 0, 0

        # 1. User Departures & Arrivals
        users_departing = [user for user in active_user_list if random.random() < departure_probability]
        for user in users_departing:
            active_user_list.remove(user)
            inactive_user_list.append(user)
        num_new_arrivals = min(np.random.poisson(arrival_rate_lambda), len(inactive_user_list))
        for _ in range(num_new_arrivals):
            active_user_list.append(inactive_user_list.pop())

        # 2. Reset Connections
        for user in active_user_list: user.sat = None
        for sat in satellite_table.values(): sat.serving_users = []

        # 3. Assign Users to Closest Satellite
        for user in active_user_list:
            best_sat, min_distance = None, float('inf')
            for sat in satellite_table.values():
                if sat.connect_user(i, user):
                    distance = sat.distance_to_user(i, user)
                    if distance < min_distance:
                        min_distance, best_sat = distance, sat
            if best_sat:
                user.sat = best_sat
                best_sat.serving_users.append(user.user_id)

        # 4. PERIODICALLY UPDATE CACHE PLACEMENT USING THE NEW ALGORITHM
        if i > 0 and i % 60 == 0:
            print(f"--- Running Algorithm at timeslot {i} ---")
            bhandari_algorithm.update_cache_placement(i)

        # 5. SIMULATE REQUESTS AND CACHING DECISIONS
        timeslot_total_cost = 0
        sat_names_ordered = sorted(list(satellite_table.keys()))

        for sat_idx, sat_name in enumerate(sat_names_ordered):
            sat = satellite_table[sat_name]
            if not sat.serving_users: continue
            sat_cost = 0

            # Find nearest ground station via expanding search
            nearest_gs, hops_to_gs = None, 1
            for hop in range(len(sat_names_ordered) // 2 + 1):
                indices_to_check = [sat_idx] if hop == 0 else []
                if hop > 0:
                    if sat_idx - hop >= 0: indices_to_check.append(sat_idx - hop)
                    if sat_idx + hop < len(sat_names_ordered): indices_to_check.append(sat_idx + hop)
                if not indices_to_check: continue

                visible_gs_options = []
                for check_idx in indices_to_check:
                    candidate_sat = satellite_table[sat_names_ordered[check_idx]]
                    for gs in ground_station_list:
                        if gs.is_satellite_in_view(candidate_sat, i):
                            dist = gs.calculate_distance_to_satellite(sat.lat.iloc[i], sat.lon.iloc[i], sat.alt.iloc[i])
                            visible_gs_options.append({'gs': gs, 'distance': dist})

                if visible_gs_options:
                    closest_option = min(visible_gs_options, key=lambda x: x['distance'])
                    nearest_gs, hops_to_gs = closest_option['gs'], hop
                    break

            for user_id in sat.serving_users:
                user = next((u for u in active_user_list if u.user_id == user_id), None)
                if not user: continue

                success, request_data = user.generate_request(NUM_VIDEOS, VIEWS_PER_VIDEO, ZIPF_ALPHA, view_range_B=5)
                if not success: continue

                number_of_requests_this_slot += 1
                h, l = request_data['h'], request_data['l']
                D, alpha, T_DIBR = 3, 1.5, 0

                # --- DYNAMIC PROGRAMMING with DIBR cost tracking ---
                mu, prev, mu_dibr = {}, {}, {}
                tau_h = get_tau_j(h, sat, nearest_gs, hops_to_gs)
                mu[h], prev[h], mu_dibr[h] = tau_h, None, 0

                for j in range(h + 1, l + 1):
                    tau_j = get_tau_j(j, sat, nearest_gs, hops_to_gs)
                    min_cost, best_pred, dibr_cost_for_min_step = float('inf'), None, 0
                    for k in range(max(j - D, h), j + 1):
                        dibr_cost = (alpha * (j - k) + T_DIBR) * (j - k - 1) if k < j else 0
                        current_cost = mu.get(k, float('inf')) + dibr_cost + tau_j
                        if current_cost < min_cost:
                            min_cost, best_pred = current_cost, k
                            dibr_cost_for_min_step = mu_dibr.get(k, 0) + dibr_cost
                    mu[j], prev[j], mu_dibr[j] = min_cost, best_pred, dibr_cost_for_min_step

                request_cost = mu.get(l, float('inf'))
                if request_cost == float('inf'): continue

                dibr_cost_for_request = mu_dibr.get(l, 0)
                total_dibr_synthesis_cost += dibr_cost_for_request
                sat_cost += request_cost

                # --- EXECUTE PLAN & TRACK METRICS ---
                transfer_points, curr = [], l
                while curr is not None and curr >= h:
                    transfer_points.append(curr); curr = prev.get(curr)
                V_fetch, V_local_hit, V_ISL, V_ground_station = set(transfer_points), set(), set(), set()

                for v in V_fetch:
                    if sat.is_view_cached(v):
                        V_local_hit.add(v)
                        cache_hit_stats[sat_name]['hits'] += 1
                        sat.last_access_time[v], sat.access_frequency[v] = i, sat.access_frequency.get(v, 0) + 1
                    else:
                        V_ground_station.add(v)
                        cache_hit_stats[sat_name]['misses'] += 1
                        if nearest_gs: nearest_gs.transmit_to_satellite(sat, [v])
                        cache_content_with_eviction(sat, v, i, NUM_VIDEOS, VIEWS_PER_VIDEO, ZIPF_ALPHA)

                cost_serving, cost_isl_hop, cost_gs_fetch, cost_miss_penalty = 5, 10, 20, 50
                network_cost_breakdown['serving'] += len(V_fetch) * cost_serving
                if V_ground_station:
                    network_cost_breakdown['miss_penalty'] += len(V_ground_station) * cost_miss_penalty
                    network_cost_breakdown['gs_fetch'] += len(V_ground_station) * cost_gs_fetch
                    if hops_to_gs > 0:
                        network_cost_breakdown['isl'] += len(V_ground_station) * hops_to_gs * cost_isl_hop

                hops_for_this_request = hops_to_gs * len(V_ground_station)
                if hops_for_this_request > 0:
                    hops_this_timeslot += hops_for_this_request
                    requests_with_isl_this_timeslot += 1

                view_sets_for_latency = {'local': V_local_hit, 'isl': V_ISL, 'gs': V_ground_station}
                # latency = calculate_request_latency(sat, user, i, view_sets_for_latency, nearest_gs, hops_to_gs, satellite_table, sat_names_ordered)
                latency = calculate_request_latency(sat, user, i, view_sets_for_latency, nearest_gs, hops_to_gs, satellite_table, sat_names_ordered, active_user_list)

                if latency != float('inf'): 
                    latencies_this_timeslot.append(latency)

            satellite_costs[sat_name] += sat_cost
            timeslot_total_cost += sat_cost

        total_requests_over_simulation += number_of_requests_this_slot
        timeslot_costs.append(timeslot_total_cost)
        total_isl_hops_over_simulation += hops_this_timeslot
        total_requests_with_isl_over_simulation += requests_with_isl_this_timeslot

        average_latency_per_timeslot.append(sum(latencies_this_timeslot) / len(latencies_this_timeslot) if latencies_this_timeslot else 0)
        average_hops_per_timeslot.append(total_isl_hops_over_simulation / requests_with_isl_this_timeslot if requests_with_isl_this_timeslot > 0 else 0)

    # Calculate metrics
    total_system_cost = sum(timeslot_costs)
    overall_hits = sum(s['hits'] for s in cache_hit_stats.values())
    overall_reqs = sum(s['hits'] + s['misses'] for s in cache_hit_stats.values())
    overall_hit_rate = overall_hits / overall_reqs if overall_reqs > 0 else 0
    
    all_latencies = [lat for lat in average_latency_per_timeslot if lat > 0]
    overall_avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    
    overall_avg_isl_hops = total_isl_hops_over_simulation / total_requests_with_isl_over_simulation if total_requests_with_isl_over_simulation > 0 else 0
    
    total_network_cost = sum(network_cost_breakdown.values())

    # Save plots with SCA in path
    os.makedirs('SCA', exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(Total_timeslot), timeslot_costs, marker='o', linewidth=1.5, markersize=4, label='Total Cost')
    plt.title(f'Total System Cost per Timeslot (Storage: {storage_constraint_Z})')
    plt.xlabel('Timeslot'); plt.ylabel('Total Cost'); plt.grid(True, alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(f'SCA/cost_over_time_{storage_constraint_Z}.png', dpi=300)
    plt.close()

    sat_names_ordered = sorted(list(satellite_table.keys()))
    hit_rates = [cache_hit_stats[s]['hits'] / (cache_hit_stats[s]['hits'] + cache_hit_stats[s]['misses']) if (cache_hit_stats[s]['hits'] + cache_hit_stats[s]['misses']) > 0 else 0 for s in sat_names_ordered]
    plt.figure(figsize=(12, 6))
    plt.bar(sat_names_ordered, hit_rates, color='lightgreen', edgecolor='darkgreen')
    plt.title(f'Cache Hit Rate per Satellite (Storage: {storage_constraint_Z})')
    plt.xlabel('Satellite'); plt.ylabel('Cache Hit Rate'); plt.xticks(rotation=45, ha='right'); plt.ylim(0, 1); plt.grid(True, axis='y', alpha=0.4); plt.tight_layout()
    plt.savefig(f'SCA/cache_hit_rates_{storage_constraint_Z}.png', dpi=300)
    plt.close()

    return {
        'storage': storage_constraint_Z,
        'total_cost': total_system_cost,
        'hit_rate': overall_hit_rate,
        'avg_latency': overall_avg_latency,
        'avg_isl_hops': overall_avg_isl_hops,
        'dibr_cost': total_dibr_synthesis_cost,
        'total_cache_miss_cost': network_cost_breakdown['miss_penalty'],
        'network_cost': total_network_cost,
        'Total Requests': total_requests_over_simulation,
    }

# Main execution
if __name__ == "__main__":
    # storage_sizes = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    storage_sizes = [600]

    results = []
    
    for storage_constraint_Z in storage_sizes:
        result = run_simulation(storage_constraint_Z)
        results.append(result)
    
    # Print final results table
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Storage':<8} {'Total Cost':<12} {'Hit Rate':<10} {'Avg Latency':<12} {'Avg ISL Hops':<13} {'DIBR Cost':<12} {'Cache Miss Cost':<15} {'Network Cost':<12} {'Total Requests':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['storage']:<8} {result['total_cost']:<12.2f} {result['hit_rate']:<10.3f} {result['avg_latency']:<12.4f} {result['avg_isl_hops']:<13.3f} {result['dibr_cost']:<12.2f} {result['total_cache_miss_cost']:<12.2f} {result['network_cost']:<12.2f} {result['Total Requests']:<12}")


    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('SCA/SCA_results.csv', index=False)
    print(f"\nResults saved to 'SCA/SCA_results.csv'")
