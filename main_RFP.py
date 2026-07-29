# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from container_RFP import User, Satellite, GroundStation, calculate_rate_mbps
# from container_0617 import User, Satellite, GroundStation
# from container_0617 import generate_zipf_distribution, calculate_rate_mbps, SPEED_OF_LIGHT_KM_S
# from mpl_toolkits.basemap import Basemap

EARTH_RADIUS = 6371 # in km
BOLTZMANN_K = 1.38e-23
NOISE_TEMP_K = 290
SPEED_OF_LIGHT_KM_S = 300000 # Speed of light in km/s

def generate_zipf_distribution(N, alpha):
    """Generates a Zipf probability distribution for N items."""
    if N <= 0: return []
    x = np.arange(1, N + 1)
    weights = x ** (-alpha)
    return weights / np.sum(weights) if np.sum(weights) > 0 else []

def find_nearest_ground_station_with_expanding_search(satellite_table, sat_names_ordered, current_sat_idx, ground_station_list, time, original_sat):
    """
    Find the nearest ground station using expanding search approach.
    Returns: (ground_station, hops) or (None, -1) if not found
    """
    max_hops = len(sat_names_ordered) // 2 + 1
    
    for hop in range(max_hops):
        indices_to_check = []
        if hop == 0:
            indices_to_check.append(current_sat_idx)
        else:
            # Check neighbors at hop distance
            if current_sat_idx - hop >= 0:
                indices_to_check.append(current_sat_idx - hop)
            if current_sat_idx + hop < len(sat_names_ordered):
                indices_to_check.append(current_sat_idx + hop)
        
        if not indices_to_check:
            continue
        
        # Find all ground stations visible from satellites at this hop distance
        visible_gs_options = []
        for check_idx in indices_to_check:
            candidate_sat = satellite_table[sat_names_ordered[check_idx]]
            for gs in ground_station_list:
                if gs.is_satellite_in_view(candidate_sat, time):
                    # Calculate distance from the original satellite to this GS
                    dist_from_original_sat = gs.calculate_distance_to_satellite(
                        original_sat.lat[time], original_sat.lon[time], original_sat.alt[time]
                    )
                    visible_gs_options.append({'gs': gs, 'distance': dist_from_original_sat})
        
        # If we found any visible GS at this hop level, find the closest one and return
        if visible_gs_options:
            closest_option = min(visible_gs_options, key=lambda x: x['distance'])
            return closest_option['gs'], hop
    
    return None, -1

def get_tau_j(view_j, sat, nearest_gs, hops_to_gs=0, neighbor_sats={}):
    """
    Calculates the cost of fetching a single view 'j'.
    This version is adapted for main_RFP.py's cooperative areas.
    """
    cost_serving, cost_isl_hop, cost_gs_fetch, cost_miss_penalty = 5, 10, 20, 50

    if sat.is_view_cached(view_j):
        return cost_serving

    # Check for the view in cooperative neighbors
    is_in_neighbor_cache = False
    min_hops = float('inf')
    for neighbor_sat, hops in neighbor_sats.items():
        if neighbor_sat.is_view_cached(view_j):
            is_in_neighbor_cache = True
            if hops < min_hops:
                min_hops = hops
    
    if is_in_neighbor_cache:
        # Use realistic hop cost from RFP model
        return (cost_isl_hop * min_hops) + cost_serving

    if nearest_gs:
        # Cost to fetch from the ground station
        return cost_miss_penalty + cost_gs_fetch + (hops_to_gs * cost_isl_hop) + cost_serving
    
    # Failsafe if no GS is available
    # return cost_miss_penalty

def cache_content_with_eviction(satellite, content_id, timeslot):
    """
    Cache content with eviction logic. If the cache is full, evict the
    least frequently and recently used item.
    """
    if satellite.is_view_cached(content_id):
        satellite.last_access_time[content_id] = timeslot
        satellite.access_frequency[content_id] = satellite.access_frequency.get(content_id, 0) + 1
        return

    if len(satellite.cache_state) < satellite.storage_constraint_Z:
        satellite.cache_view(content_id)
        satellite.last_access_time[content_id] = timeslot
        satellite.access_frequency[content_id] = 1
        return

    # --- Eviction Logic ---
    min_score = float('inf')
    view_to_evict = -1
    for view_id in satellite.cache_state:
        frequency = satellite.access_frequency.get(view_id, 0)
        recency = 1.0 / (timeslot - satellite.last_access_time.get(view_id, 0) + 1)
        score = frequency + recency # Simple LFU/LRU hybrid score
        if score < min_score:
            min_score = score
            view_to_evict = view_id
            
    if view_to_evict != -1:
        satellite.evict_view(view_to_evict)
        if view_to_evict in satellite.last_access_time:
            del satellite.last_access_time[view_to_evict]
        if view_to_evict in satellite.access_frequency:
            del satellite.access_frequency[view_to_evict]
            
    satellite.cache_view(content_id)
    satellite.last_access_time[content_id] = timeslot
    satellite.access_frequency[content_id] = 1

# def calculate_request_latency(sat, user, time, view_sets, nearest_gs, hops_to_gs, sat_table, ordered_names, active_user_list):
#     """
#     Calculates the real latency for a user's request based on the DP plan.
#     Latency = max(latency of each transmitted view).
    
#     MODIFICATION: Now considers bandwidth sharing among all users being served by the satellite.
#     The satellite's total bandwidth is equally distributed among all serving users.
#     """
#     view_size_mbit = user.video_size_mb * 8
#     max_latency = 0.0

#     # Calculate number of users being served by this satellite
#     # Calculate number of users being served by this satellite from active_user_list
#     num_serving_users = sum(1 for user in active_user_list if user.sat == sat)
#     if num_serving_users == 0:
#         return float('inf')  # No users being served

#     # Calculate Downlink (Satellite -> User) latency components with bandwidth sharing
#     dist_s_u = sat.distance_to_user(time, user)
#     prop_s_u = dist_s_u / SPEED_OF_LIGHT_KM_S
    
#     # Total satellite downlink capacity (Mbps) - adjust this value based on your satellite specs
#     total_downlink_capacity_mbps = 650
    
#     # Effective data rate per user (equal bandwidth sharing)
#     effective_rate_s_u_mbps = total_downlink_capacity_mbps / num_serving_users
    
#     if effective_rate_s_u_mbps == 0: 
#         return float('inf')
    
#     # Transmission time using the effective (shared) data rate
#     tx_s_u = view_size_mbit / effective_rate_s_u_mbps

    
#     # Calculate latency for each transmitted view based on its source
#     for view in view_sets.get('local', set()):
#         latency = prop_s_u + tx_s_u
#         if latency > max_latency: 
#             max_latency = latency


#     for view in view_sets.get('isl', set()):
#         # ISL (Inter-Satellite Link) transmission
#         isl_prop_latency = 1000 / SPEED_OF_LIGHT_KM_S # Avg distance for ISL
        
#         # For ISL, we might also want to consider sharing if multiple satellites
#         # are requesting data simultaneously, but for simplicity, we'll use full ISL capacity
#         isl_tx_latency = view_size_mbit / (100 * 1000)  # sat.isl_data_rate_gbps * 1000
        
#         # Total latency: ISL transmission + downlink to user (with bandwidth sharing)
#         latency = isl_prop_latency + isl_tx_latency + prop_s_u + tx_s_u
#         if latency > max_latency: 
#             max_latency = latency

#     if nearest_gs:
#         for view in view_sets.get('gs', set()):
#             # Find the satellite actually connecting to the GS
#             connected_sat = sat
#             if hops_to_gs > 0:
#                 # This is a simplification; a real system would trace the exact path.
#                 # We'll use the original satellite's position as a proxy.
#                 pass 
            
#             # Ground station to satellite uplink
#             dist_gs_s = nearest_gs.calculate_distance_to_satellite(
#                 connected_sat.lat[time], connected_sat.lon[time], connected_sat.alt[time]
#             )
#             prop_gs_s = dist_gs_s / SPEED_OF_LIGHT_KM_S
            
#             # Uplink rate from ground station (typically not shared among users on the same satellite)
#             rate_gs_s_mbps = 2500 
#             if rate_gs_s_mbps == 0: 
#                 continue
#             tx_gs_s = view_size_mbit / rate_gs_s_mbps
            
#             # ISL hop latency if data needs to be forwarded through other satellites
#             isl_hop_latency = hops_to_gs * (
#                 (1000 / SPEED_OF_LIGHT_KM_S) + 
#                 (view_size_mbit / (sat.isl_data_rate_gbps * 1000))
#             )
            
#             # Total latency: GS uplink + ISL forwarding + downlink to user (with bandwidth sharing)
#             latency = prop_gs_s + tx_gs_s + isl_hop_latency + prop_s_u + tx_s_u
#             if latency > max_latency: 
#                 max_latency = latency

#     return max_latency

def calculate_request_latency(sat, user, time, view_sets, nearest_gs, hops_to_gs, sat_table, ordered_names, active_user_list):
    """
    Updated latency calculation using modern LEO communication parameters and elevation angles.
    """
    view_size_mbit = user.video_size_mb * 8
    max_latency = 0.0

    # Count active users
    active_user_ids = {user.user_id for user in active_user_list}
    num_serving_users = sum(1 for user_id in sat.serving_users if user_id in active_user_ids)
    if num_serving_users == 0:
        return float('inf')

    # Calculate distance and elevation angle
    dist_s_u = sat.distance_to_user(time, user)
    prop_s_u = dist_s_u / SPEED_OF_LIGHT_KM_S
    
    # Calculate elevation angle for more accurate link modeling
    elevation_angle = sat.calculate_elevation_angle(time, user)
    
    # Calculate DYNAMIC downlink rate based on actual link conditions
    downlink_rate_mbps = calculate_rate_mbps(
        tx_power_watt=sat.tx_power_watt,           # 15.0 W
        tx_gain_dbi=sat.antenna_gain_dbi,         # 48.0 dBi  
        rx_gain_dbi=user.antenna_gain_dbi,        # 12.0 dBi (updated)
        distance_km=dist_s_u,
        bandwidth_hz=user.bandwidth_hz,           # 250 MHz (updated)
        frequency_ghz=sat.downlink_freq_ghz,      # 14.0 GHz (Ku-band)
        elevation_angle_deg=elevation_angle
    )
    
    # Bandwidth sharing among users (with minimum guarantee)
    # if downlink_rate_mbps == float('inf'):
    #     effective_rate_per_user_mbps = float('inf')
    # else:
    #     # Ensure minimum rate per user (realistic scheduling)
    #     min_rate_guarantee = 10.0  # 10 Mbps minimum per user
            # effective_rate_per_user_mbps = max(downlink_rate_mbps / num_serving_users, min_rate_guarantee)
    effective_rate_per_user_mbps = downlink_rate_mbps / num_serving_users
    # LOCAL cache hits (BEST performance)
    for view in view_sets.get('local', set()):
        if effective_rate_per_user_mbps == float('inf'):
            tx_time = 0
        else:
            # Local cache gets full effective bandwidth
            tx_time = view_size_mbit / effective_rate_per_user_mbps
        
        latency = prop_s_u + tx_time
        if latency > max_latency: 
            max_latency = latency

    # ISL fetches (MEDIUM performance)
    for view in view_sets.get('isl', set()):
        # ISL rate (optical links)
        isl_rate_mbps = sat.isl_data_rate_gbps * 1000  # 100 Gbps -> 100,000 Mbps
        
        # ISL delays
        isl_prop_latency = 1000 / SPEED_OF_LIGHT_KM_S
        isl_tx_latency = view_size_mbit / isl_rate_mbps if isl_rate_mbps > 0 else 0
        
        # Reduced effective bandwidth due to ISL traffic overhead
        if effective_rate_per_user_mbps == float('inf'):
            downlink_tx_time = 0
        else:
            reduced_rate = effective_rate_per_user_mbps * 0.7  # 70% efficiency
            downlink_tx_time = view_size_mbit / reduced_rate
        
        latency = prop_s_u + isl_prop_latency + isl_tx_latency + downlink_tx_time
        if latency > max_latency: 
            max_latency = latency

    # Ground Station fetches (WORST performance)
    if nearest_gs:
        for view in view_sets.get('gs', set()):
            # Calculate GS uplink rate with elevation consideration
            gs_elevation = nearest_gs.calculate_elevation_angle_to_sat(sat, time)
            gs_uplink_rate_mbps = nearest_gs.get_uplink_rate_to_sat(sat, time)
            
            # GS uplink delays
            dist_gs_s = nearest_gs.calculate_distance_to_satellite(
                sat.lat[time], sat.lon[time], sat.alt[time]
            )
            prop_gs_s = dist_gs_s / SPEED_OF_LIGHT_KM_S
            
            if gs_uplink_rate_mbps == float('inf'):
                tx_gs_s = 0
            else:
                tx_gs_s = view_size_mbit / gs_uplink_rate_mbps
            
            # ISL forwarding delays
            isl_rate_mbps = sat.isl_data_rate_gbps * 1000
            isl_hop_latency = hops_to_gs * (
                (1000 / SPEED_OF_LIGHT_KM_S) + 
                (view_size_mbit / isl_rate_mbps if isl_rate_mbps > 0 else 0)
            )
            
            # Much reduced effective bandwidth due to GS fetch overhead
            if effective_rate_per_user_mbps == float('inf'):
                downlink_tx_time = 0
            else:
                reduced_rate = effective_rate_per_user_mbps * 0.4  # 40% efficiency
                downlink_tx_time = view_size_mbit / reduced_rate
            
            latency = prop_s_u + prop_gs_s + tx_gs_s + isl_hop_latency + downlink_tx_time
            if latency > max_latency: 
                max_latency = latency

    return max_latency


# --- Latency Calculation Function (from main_0629.py) ---
# def calculate_request_latency(sat, user, time, view_sets, nearest_gs, hops_to_gs, sat_table, ordered_names):
#     """
#     Calculates the real latency for a user's request based on the DP plan.
#     Latency = max(latency of each transmitted view).
#     This function provides a reasonable and well-structured model for latency.
#     """
#     view_size_mbit = user.video_size_mb * 8
#     max_latency = 0.0

#     # Calculate Downlink (Satellite -> User) latency components once
#     dist_s_u = sat.distance_to_user(time, user)
#     prop_s_u = dist_s_u / SPEED_OF_LIGHT_KM_S
#     # The data rate is simplified to a constant value for this simulation.
#     # A more detailed model could calculate this dynamically.
#     rate_s_u_mbps = 650
#     if rate_s_u_mbps == 0: 
#         return float('inf')
#     tx_s_u = view_size_mbit / rate_s_u_mbps
    
#     # Calculate latency for each transmitted view based on its source
#     for view in view_sets.get('local', set()):
#         latency = prop_s_u + tx_s_u
#         if latency > max_latency: 
#             max_latency = latency

#     for view in view_sets.get('isl', set()):
#         # Assume 1-hop for simplicity for neighbor cache
#         isl_prop_latency = 1000 / SPEED_OF_LIGHT_KM_S # Avg distance for ISL
#         isl_tx_latency = view_size_mbit / (sat.isl_data_rate_gbps * 1000)
#         latency = isl_prop_latency + isl_tx_latency + prop_s_u + tx_s_u
#         if latency > max_latency: max_latency = latency

#     if nearest_gs:
#         for view in view_sets.get('gs', set()):
#             # Find the satellite actually connecting to the GS
#             connected_sat = sat
#             if hops_to_gs > 0:
#                 # This is a simplification; a real system would trace the exact path.
#                 # We'll use the original satellite's position as a proxy.
#                 pass 
            
#             dist_gs_s = nearest_gs.calculate_distance_to_satellite(connected_sat.lat[time], connected_sat.lon[time], connected_sat.alt[time])
#             prop_gs_s = dist_gs_s / SPEED_OF_LIGHT_KM_S
#             # Uplink rate is simplified to a constant value.
#             rate_gs_s_mbps = 2500 
#             if rate_gs_s_mbps == 0: continue
#             tx_gs_s = view_size_mbit / rate_gs_s_mbps
            
#             isl_hop_latency = hops_to_gs * ( (1000 / SPEED_OF_LIGHT_KM_S) + (view_size_mbit / (sat.isl_data_rate_gbps * 1000)) )
            
#             latency = prop_gs_s + tx_gs_s + isl_hop_latency + prop_s_u + tx_s_u
#             if latency > max_latency: max_latency = latency

#     return max_latency

def predict_region_features(satellite, mu=0.1):
    """Stage 1: Predict region features using Ridge Regression."""
    if not satellite.request_history:
        return satellite.region_features 

    # L = max(1, int(len(satellite.request_history) * 0.1)) 
    # popular_contents = sorted(satellite.request_history.items(), key=lambda item: item[1], reverse=True)[:L]
    
    # content_indices = [item[0] for item in popular_contents]
    # real_popularity_R = np.array([item[1] for item in popular_contents])
    
    # CL = np.array([CONTENT_FEATURES[idx] for idx in content_indices if idx < len(CONTENT_FEATURES)])
    # if CL.shape[0] == 0: return satellite.region_features

    # total_popularity = np.sum(real_popularity_R)
    # if total_popularity == 0: return satellite.region_features
    
    # popularity_weights = real_popularity_R / total_popularity
    # R_hat_L = np.diag(popularity_weights)

    L = max(1, int(len(satellite.request_history) * 0.1))
    popular_contents_unfiltered = sorted(satellite.request_history.items(), key=lambda item: item[1], reverse=True)[:L]

    # 1. Filter the list for valid content IDs first. This is the key change.
    popular_contents = [item for item in popular_contents_unfiltered if item[0] < len(CONTENT_FEATURES)]

    # If the filtered list is empty, exit.
    if not popular_contents:
        return np.zeros(CONTENT_FEATURES.shape[1])

    # 2. Build all subsequent arrays from the *same* filtered list.
    content_indices = [item[0] for item in popular_contents]
    real_popularity_R = np.array([item[1] for item in popular_contents])
    
    CL = np.array([CONTENT_FEATURES[idx] for idx in content_indices]) # The 'if' condition is no longer needed here.

    total_popularity = np.sum(real_popularity_R)
    if total_popularity == 0:
        return np.zeros(CONTENT_FEATURES.shape[1])
    
    # 3. R_hat_L is now built from the same data as CL.
    popularity_weights = real_popularity_R / total_popularity
    R_hat_L = np.diag(popularity_weights)
    # print("CL shape:", CL.shape)
    # print("R_hat_L shape:", R_hat_L.shape)

    try:
        term1 = CL.T @ R_hat_L.T @ R_hat_L @ CL
        term2 = mu * np.identity(term1.shape[0])
        inverse_term = np.linalg.inv(term1 + term2)
        term3 = CL.T @ R_hat_L.T @ R_hat_L @ real_popularity_R
        predicted_features = inverse_term @ term3
        return predicted_features
    except np.linalg.LinAlgError:
        return satellite.region_features

def divide_cooperative_areas(satellite_table, similarity_threshold, sat_names_ordered):
    """Algorithm 1: Cooperative Area Division."""
    all_regions = set(satellite_table.keys())
    divided_regions = set()
    cooperative_areas = []
    
    # Sort undivided satellites by total request count to start with the busiest
    sats_by_activity = sorted(satellite_table.values(), 
                              key=lambda s: sum(s.request_history.values()), 
                              reverse=True)

    for start_sat in sats_by_activity:
        if start_sat.sat_name in divided_regions:
            continue

        new_area = {start_sat.sat_name}
        
        while True:
            added_in_iteration = False
            # Find all unique adjacent satellites to the current area members
            adjacent_regions = set()
            for sat_name in new_area:
                current_sat_idx = sat_names_ordered.index(sat_name)
                for neighbor_idx in [current_sat_idx - 1, current_sat_idx + 1]:
                    if 0 <= neighbor_idx < len(sat_names_ordered):
                        adj_name = sat_names_ordered[neighbor_idx]
                        if adj_name not in new_area and adj_name not in divided_regions:
                            adjacent_regions.add(adj_name)

            for adj_name in adjacent_regions:
                adj_sat = satellite_table[adj_name]
                is_similar_to_all = True
                for area_sat_name in new_area:
                    area_sat = satellite_table[area_sat_name]
                    # Calculate Cosine similarity
                    norm_g_adj = np.linalg.norm(adj_sat.region_features)
                    norm_g_area = np.linalg.norm(area_sat.region_features)

                    if norm_g_adj == 0 or norm_g_area == 0:
                        similarity = 0.0
                    else:
                        similarity = np.dot(adj_sat.region_features, area_sat.region_features) / (norm_g_adj * norm_g_area)

                    if similarity < similarity_threshold:
                        is_similar_to_all = False
                        break

                if is_similar_to_all:
                    new_area.add(adj_name)
                    added_in_iteration = True
            
            if not added_in_iteration:
                break
        
        cooperative_areas.append(list(new_area))
        divided_regions.update(new_area)
        
    return cooperative_areas

def run_cooperative_caching_game(cooperative_area, satellite_table, sat_names_ordered):
    """Algorithm 2: Cooperative Caching based on Non-Cooperative Game."""
    # Initialization: each satellite caches top popular content based on its own predicted popularity
    for sat_name in cooperative_area:
        sat = satellite_table[sat_name]
        sat.cache_state.clear()
        local_content_pop = {c: np.dot(sat.region_features, CONTENT_FEATURES[c]) for c in range(TOTAL_VIEWS)}
        top_contents = sorted(local_content_pop, key=local_content_pop.get, reverse=True)[:sat.storage_constraint_Z]
        sat.cache_state = set(top_contents)

    for iteration in range(MAX_GAME_ITERATIONS):
        changed = False
        for sat_name in cooperative_area:
            sat = satellite_table[sat_name]
            current_sat_idx = sat_names_ordered.index(sat_name)
            
            cache_benefits = {}
            for m in range(TOTAL_VIEWS):
                predicted_pop_m = np.dot(sat.region_features, CONTENT_FEATURES[m])
                u_direct = predicted_pop_m * (DELAY_TP - DELAY_TS)
                
                u_coop = 0
                for other_sat_name in cooperative_area:
                    if other_sat_name == sat_name: continue
                    other_sat = satellite_table[other_sat_name]
                    other_sat_idx = sat_names_ordered.index(other_sat_name)
                    hops = abs(current_sat_idx - other_sat_idx)
                    
                    if hops <= MAX_COOP_HOPS:
                        predicted_pop_other = np.dot(other_sat.region_features, CONTENT_FEATURES[m])
                        if m not in other_sat.cache_state:
                             u_coop += predicted_pop_other * (DELAY_TP - DELAY_TS - hops * DELAY_SS)
                
                cache_benefits[m] = u_direct + u_coop

            new_cache = set(sorted(cache_benefits, key=cache_benefits.get, reverse=True)[:sat.storage_constraint_Z])
            if new_cache != sat.cache_state:
                sat.cache_state = new_cache
                changed = True
        
        if not changed:
            break

def run_simulation(storage_constraint_Z):
    """Run simulation for a given storage constraint"""
    
    # --- Simulation Parameters ---
    video_size_per_view_mb = 4
    Total_timeslot = 240

    # --- Video Content Structure & Popularity Parameters ---
    NUM_VIDEOS = 2500
    VIEWS_PER_VIDEO = 16
    TOTAL_VIEWS = NUM_VIDEOS * VIEWS_PER_VIDEO
    ZIPF_ALPHA = 0.8
    
    # --- Generate structured features for content based on genre popularity ---
    NUM_GENRES = VIEWS_PER_VIDEO # Let each feature dimension represent a genre
    CONTENT_FEATURES = np.zeros((TOTAL_VIEWS, NUM_GENRES))
    for content_id in range(TOTAL_VIEWS):
        # Assign a primary genre based on content ID (simulates groups of similar content)
        primary_genre = (content_id // (TOTAL_VIEWS // NUM_GENRES)) % NUM_GENRES
        CONTENT_FEATURES[content_id, primary_genre] = 1
        # Add a secondary genre with some probability, based on Zipf
        if random.random() < 0.2:
            genre_pop = generate_zipf_distribution(NUM_GENRES, 1.2)
            secondary_genre = np.random.choice(np.arange(NUM_GENRES), p=genre_pop)
            if secondary_genre != primary_genre:
                 CONTENT_FEATURES[content_id, secondary_genre] = 1

    # --- Poisson Process Parameters ---
    arrival_rate_lambda = 60
    departure_probability = 0.15

    # --- RFP and DIBR Parameters ---
    RFP_UPDATE_INTERVAL = 80 # Update region features and cooperative areas every 60 timeslots
    SIMILARITY_THRESHOLD = 0.5 # beta
    MAX_GAME_ITERATIONS = 5 # P
    DELAY_TP = 500 # ms, delay from cloud server
    DELAY_TS = 300 # ms, delay from serving satellite
    DELAY_SS = 40 # ms, delay for inter-satellite link
    MAX_COOP_HOPS = int((DELAY_TP - DELAY_TS) / DELAY_SS) # K

    # --- Main Simulation Setup ---
    video_indices = list(range(NUM_VIDEOS))
    view_angle_indices = list(range(VIEWS_PER_VIDEO))

    ##################################################
    #          INITIALIZE SYSTEM COMPONENTS          #
    ##################################################

    # Load filtered user data
    users_df = pd.read_csv('data/users.csv')
    all_users = [User(row['id'], row['lat'], row['lon'], row['x'], row['y'], row['z'], video_size_mb=video_size_per_view_mb) for _, row in users_df.iterrows()]
    all_users.sort()

    initial_active_user_count = 150
    active_user_list = all_users[:initial_active_user_count]
    inactive_user_list = all_users[initial_active_user_count:]

    # --- MODIFIED: Initialize ALL Satellites ---
    satellite_table = {}
    sat_files = sorted([f for f in os.listdir('data/starlink118/satellite_trace') if f.endswith('.csv')])
    sat_names_ordered = []

    for file in sat_files:
        satellite_data = pd.read_csv(f'data/starlink118/satellite_trace/{file}')
        satellite_name = file.split('_')[0]
        sat = Satellite(satellite_name, satellite_data,
                       storage_constraint_Z=storage_constraint_Z,
                       total_views=TOTAL_VIEWS,
                       view_size_mb=video_size_per_view_mb,
                       region_id=None)
        satellite_table[satellite_name] = sat
        sat_names_ordered.append(satellite_name)

    # --- Initialize Ground Stations ---
    ground_station_list = []
    ground_station_locations = [
        # Europe
        # {"id": "GS_EU_FR_VILL", "name": "Villenave d'Ornon", "lat": 44.78, "lon": -0.55}, # France
        # {"id": "GS_EU_DE_AERZ", "name": "Aerzen", "lat": 52.05, "lon": 9.26},         # Germany
        # {"id": "GS_EU_DE_USIN", "name": "Usingen", "lat": 50.33, "lon": 8.53},        # Germany
        # {"id": "GS_EU_IE_BALL", "name": "Ballinspittle", "lat": 51.65, "lon": -8.62}, # Ireland
        # {"id": "GS_EU_IE_ELFO", "name": "Elfordstown", "lat": 51.89, "lon": -8.24},   # Ireland
        # {"id": "GS_EU_IT_FOGG", "name": "Foggia", "lat": 41.46, "lon": 15.55},        # Italy
        # {"id": "GS_EU_IT_MARS", "name": "Marsala", "lat": 37.80, "lon": 12.43},       # Italy
        # {"id": "GS_EU_IT_MILA", "name": "Milano", "lat": 45.46, "lon": 9.19},         # Italy
        # {"id": "GS_EU_LT_KAUN", "name": "Kaunas", "lat": 54.90, "lon": 23.90},        # Lithuania
        # {"id": "GS_EU_NO_TROM", "name": "Tromsø", "lat": 69.65, "lon": 18.96},        # Norway
        # {"id": "GS_EU_PL_WOLA", "name": "Wola Krobowska", "lat": 52.12, "lon": 20.68},# Poland
        # {"id": "GS_EU_PT_ALFO", "name": "Alfouvar de Cima", "lat": 40.21, "lon": -8.42}, # Portugal
        # {"id": "GS_EU_PT_COVI", "name": "Covilha", "lat": 40.28, "lon": -7.50},       # Portugal
        # {"id": "GS_EU_ES_IBI", "name": "Ibi", "lat": 38.62, "lon": -0.57},            # Spain
        # {"id": "GS_EU_ES_LEPE", "name": "Lepe", "lat": 37.25, "lon": -7.20},          # Spain
        # {"id": "GS_EU_ES_VILL", "name": "Villarejo de Salvanes", "lat": 40.17, "lon": -3.27}, # Spain
        # {"id": "GS_EU_UK_CHAL", "name": "Chalfont Grove", "lat": 51.61, "lon": -0.56}, # United Kingdom
        {"id": "GS_EU_UK_GOON", "name": "Goonhilly", "lat": 50.05, "lon": -5.18},     # United Kingdom
        # {"id": "GS_EU_UK_IOM", "name": "Isle of Man", "lat": 54.23, "lon": -4.55},    # United Kingdom

        # Africa
        # {"id": "GS_AF_NG_IKIR", "name": "Ikire", "lat": 7.38, "lon": 4.18},           # Nigeria
        # {"id": "GS_AF_NG_LEKK", "name": "Lekki", "lat": 6.45, "lon": 4.09},           # Nigeria
        # {"id": "GS_AF_Kenya", "name": "Nairobi", "lat": -1.29, "lon": 36.82},         # Kenya
        {"id": "GS_AF_MZ", "name": "Matola", "lat": -25.92, "lon": 32.42},            # MZ

        # Midddle East
        # {"id": "GS_ME_OMAN", "name": "Murayjat", "lat": 23.72, "lon": 57.78},  # 23.716. Longitude : 57.783
        # {"id": "GS_ME_TURKEY", "name": "Muallim", "lat": 36.92, "lon": 38.06}, # 36.921. Longitude : 38.059
    ]

    for loc in ground_station_locations:
        gs = GroundStation(loc['id'], loc['name'], loc['lat'], loc['lon'], total_views=TOTAL_VIEWS, view_size_mb=video_size_per_view_mb)
        ground_station_list.append(gs)

    # Initialize Tracking Variables
    satellite_costs = {sat_name: 0 for sat_name in satellite_table.keys()}
    timeslot_costs = []
    cache_hit_stats = {sat_name: {'hits': 0, 'misses': 0, 'dibr_hits': 0} for sat_name in satellite_table.keys()}
    total_requests_over_simulation = 0
    cooperative_areas = []

    # Initialize trackers for the enhanced metrics
    total_dibr_synthesis_cost = 0
    network_cost_breakdown = {
        'miss_penalty': 0, 
        'gs_fetch': 0, 
        'isl': 0, 
        'serving': 0
    }

    # Additional tracking for service latency calculation
    total_service_delay = 0
    total_successful_requests = 0
    average_latency_per_timeslot = []

    total_isl_hops_over_simulation = 0
    total_requests_with_isl_over_simulation = 0
    average_hops_per_timeslot = []

    ########################################################################################################
    ####################################### MAIN SIMULATION LOOP ###########################################
    ########################################################################################################

    for i in range(Total_timeslot):
        if i % 60 == 0:
            print(f'Storage {storage_constraint_Z} - Time slot {i:03d} | Active Users: {len(active_user_list)}')
        number_of_requests = 0
        
        # === Initialize per-timeslot tracking ===
        timeslot_latencies = []
        
        # --- Periodically run RFP and Game Theory Caching ---
        if i % RFP_UPDATE_INTERVAL == 0:
            for sat in satellite_table.values():
                sat.region_features = predict_region_features(sat)
            cooperative_areas = divide_cooperative_areas(satellite_table, SIMILARITY_THRESHOLD, sat_names_ordered)
            for area in cooperative_areas:
                run_cooperative_caching_game(area, satellite_table, sat_names_ordered)
        
        # --- USER DYNAMICS (Arrivals/Departures) ---
        users_departing = [user for user in active_user_list if random.random() < departure_probability]
        for user in users_departing:
            active_user_list.remove(user)
            inactive_user_list.append(user)
            
        num_new_arrivals = min(np.random.poisson(arrival_rate_lambda), len(inactive_user_list))
        for _ in range(num_new_arrivals):
            active_user_list.append(inactive_user_list.pop())

        # --- SATELLITE-USER CONNECTION ---
        for user in active_user_list: user.sat = None
        for sat in satellite_table.values(): sat.serving_users = []
        
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
                
        # --- PROCESS USER REQUESTS ---
        timeslot_total_cost = 0

        for sat_idx, sat_name in enumerate(sat_names_ordered):
            sat = satellite_table[sat_name]
            if not sat.serving_users: continue
            
            nearest_gs, hops_to_gs = find_nearest_ground_station_with_expanding_search(
                satellite_table, sat_names_ordered, sat_idx, ground_station_list, i, sat
            )
            
            # Find the cooperative area the current satellite belongs to
            current_coop_area_sats = {} # Dict of {satellite_object: hops}
            for area in cooperative_areas:
                if sat.sat_name in area:
                    for neighbor_name in area:
                        if neighbor_name != sat.sat_name:
                            neighbor_sat = satellite_table[neighbor_name]
                            neighbor_sat_idx = sat_names_ordered.index(neighbor_name)
                            hops = abs(sat_idx - neighbor_sat_idx)
                            current_coop_area_sats[neighbor_sat] = hops
                    break

            for user_id in sat.serving_users:
                user = next((u for u in active_user_list if u.user_id == user_id), None)
                if not user: continue

                success, request_data = user.generate_request(
                    NUM_VIDEOS, VIEWS_PER_VIDEO, ZIPF_ALPHA, view_range_B=5
                )

                if not success: continue
                
                number_of_requests += 1
                h, l = request_data['h'], request_data['l']
                D, alpha, T_DIBR = 3, 1.5, 0

                # Update request history for the serving satellite before processing
                for view_to_get in range(h, l + 1):
                    sat.request_history[view_to_get] = sat.request_history.get(view_to_get, 0) + 1

                # --- DYNAMIC PROGRAMMING (DP) CALCULATION ---
                mu, prev = {}, {}
                tau_h = get_tau_j(h, sat, nearest_gs, hops_to_gs, current_coop_area_sats)
                mu[h], prev[h] = tau_h, None

                for j in range(h + 1, l + 1):
                    tau_j = get_tau_j(j, sat, nearest_gs, hops_to_gs, current_coop_area_sats)
                    min_cost, best_pred = float('inf'), None
                    for k in range(max(j - D, h), j + 1):
                        dibr_cost = (alpha * (j - k) + T_DIBR) * (j - k - 1) if k < j else 0
                        current_cost = mu.get(k, float('inf')) + dibr_cost + tau_j
                        if current_cost < min_cost:
                            min_cost, best_pred = current_cost, k
                    mu[j], prev[j] = min_cost, best_pred
                
                request_cost = mu.get(l, float('inf'))
                timeslot_total_cost += request_cost
                satellite_costs[sat.sat_name] += request_cost

                # Calculate DIBR synthesis cost for this request
                transfer_points, curr = [], l
                while curr is not None and curr >= h:
                    transfer_points.append(curr)
                    curr = prev.get(curr)
                
                V_fetch = set(transfer_points)
                all_req_views = set(range(h, l + 1))
                non_fetched_views = all_req_views - V_fetch
                
                # Calculate DIBR cost based on DP solution
                request_dibr_cost = 0
                for j in range(h + 1, l + 1):
                    if j not in V_fetch:
                        # Find the previous fetch point for this view
                        prev_fetch = None
                        for k in range(j - 1, h - 1, -1):
                            if k in V_fetch:
                                prev_fetch = k
                                break
                        if prev_fetch is not None:
                            distance = j - prev_fetch
                            if distance <= D:
                                request_dibr_cost += (alpha * distance + T_DIBR) * distance
                
                total_dibr_synthesis_cost += request_dibr_cost
                
                # Calculate network transmission costs
                cost_serving, cost_isl_hop, cost_gs_fetch, cost_miss_penalty = 5, 10, 20, 50
                request_network_cost = 0

                hops_for_this_request = 0
                
                for v_fetch in V_fetch:
                    # Serving cost for every view transmitted
                    network_cost_breakdown['serving'] += cost_serving
                    request_network_cost += cost_serving
                    
                    if sat.is_view_cached(v_fetch):
                        # Local cache hit - only serving cost (already counted above)
                        pass
                    elif any(neighbor_sat.is_view_cached(v_fetch) for neighbor_sat in current_coop_area_sats.keys()):
                        # Cooperative cache hit - add ISL cost
                        min_hops = min(hops for neighbor_sat, hops in current_coop_area_sats.items() 
                                     if neighbor_sat.is_view_cached(v_fetch))
                        isl_cost = cost_isl_hop * min_hops
                        network_cost_breakdown['isl'] += isl_cost
                        request_network_cost += isl_cost
                        hops_for_this_request += min_hops
                    else:
                        # Ground station fetch - add all costs
                        network_cost_breakdown['miss_penalty'] += cost_miss_penalty
                        network_cost_breakdown['gs_fetch'] += cost_gs_fetch
                        if hops_to_gs > 0:
                            gs_isl_cost = hops_to_gs * cost_isl_hop
                            network_cost_breakdown['isl'] += gs_isl_cost
                            request_network_cost += gs_isl_cost
                            hops_for_this_request += hops_to_gs
                        request_network_cost += cost_miss_penalty + cost_gs_fetch
                
                if hops_for_this_request > 0:
                    total_isl_hops_over_simulation += hops_for_this_request
                    total_requests_with_isl_over_simulation += 1

                # --- EXECUTE THE DP PLAN AND UPDATE CACHES/STATS ---
                V_local_hit, V_ISL, V_ground_station = set(), set(), set()
                
                for v_fetch in V_fetch:
                    # Case 1: True local cache hit
                    if sat.is_view_cached(v_fetch):
                        cache_hit_stats[sat.sat_name]['hits'] += 1
                        sat.last_access_time[v_fetch] = i
                        sat.access_frequency[v_fetch] = sat.access_frequency.get(v_fetch, 0) + 1
                        V_local_hit.add(v_fetch)
                        continue
                    
                    # Case 2: Cooperative hit (found in a neighbor's cache)
                    is_in_neighbor = any(neighbor_sat.is_view_cached(v_fetch) for neighbor_sat in current_coop_area_sats.keys())
                    if is_in_neighbor:
                        cache_hit_stats[sat.sat_name]['hits'] += 1 
                        cache_content_with_eviction(sat, v_fetch, i)
                        V_ISL.add(v_fetch)
                        continue

                    # Case 3: Cache miss (must be fetched from Ground Station)
                    cache_hit_stats[sat.sat_name]['misses'] += 1
                    cache_content_with_eviction(sat, v_fetch, i)
                    V_ground_station.add(v_fetch)

                # DIBR hits are calculated based on the views NOT fetched
                cache_hit_stats[sat.sat_name]['dibr_hits'] += len(non_fetched_views)

                # Calculate latency
                view_sets_for_latency = {'local': V_local_hit, 'isl': V_ISL, 'gs': V_ground_station}
                latency = calculate_request_latency(sat, user, i, view_sets_for_latency, nearest_gs, hops_to_gs, satellite_table, sat_names_ordered, active_user_list)
                
                if latency != float('inf'):
                    timeslot_latencies.append(latency)
                    total_service_delay += latency
                    total_successful_requests += 1

        total_requests_over_simulation += number_of_requests
        timeslot_costs.append(timeslot_total_cost)
        
        # Calculate average latency for this timeslot
        if timeslot_latencies:
            avg_lat = sum(timeslot_latencies) / len(timeslot_latencies)
            average_latency_per_timeslot.append(avg_lat)
        else:
            average_latency_per_timeslot.append(0)

    # Calculate results
    total_system_cost = sum(satellite_costs.values())
    overall_hits = sum(stats['hits'] for stats in cache_hit_stats.values())
    overall_misses = sum(stats['misses'] for stats in cache_hit_stats.values())
    total_lookups = overall_hits + overall_misses
    overall_hit_rate = overall_hits / total_lookups if total_lookups > 0 else 0
    
    overall_avg_latency = sum(average_latency_per_timeslot) / Total_timeslot if Total_timeslot > 0 else 0
    overall_avg_isl_hops = total_isl_hops_over_simulation / total_requests_with_isl_over_simulation if total_requests_with_isl_over_simulation > 0 else 0
    total_network_transmission_cost = sum(network_cost_breakdown.values())
    total_network_transmission_cost = network_cost_breakdown['miss_penalty']
    
    return {
        'storage': storage_constraint_Z,
        'total_cost': total_system_cost,
        'hit_rate': overall_hit_rate,
        'avg_latency': overall_avg_latency,
        'avg_isl_hops': overall_avg_isl_hops,
        'dibr_cost': total_dibr_synthesis_cost,
        'network_cost': total_network_transmission_cost,
        'timeslot_costs': timeslot_costs,
        'cache_hit_stats': cache_hit_stats,
        'satellite_table': satellite_table,
        'total_requests': total_requests_over_simulation,
    }

# --- Global variables needed for the simulation ---
# --- Video Content Structure & Popularity Parameters ---
NUM_VIDEOS = 2500
VIEWS_PER_VIDEO = 16
TOTAL_VIEWS = NUM_VIDEOS * VIEWS_PER_VIDEO
ZIPF_ALPHA = 0.8

# --- Generate structured features for content based on genre popularity ---
NUM_GENRES = VIEWS_PER_VIDEO # Let each feature dimension represent a genre
CONTENT_FEATURES = np.zeros((TOTAL_VIEWS, NUM_GENRES))
for content_id in range(TOTAL_VIEWS):
    # Assign a primary genre based on content ID (simulates groups of similar content)
    primary_genre = (content_id // (TOTAL_VIEWS // NUM_GENRES)) % NUM_GENRES
    CONTENT_FEATURES[content_id, primary_genre] = 1
    # Add a secondary genre with some probability, based on Zipf
    if random.random() < 0.2:
        genre_pop = generate_zipf_distribution(NUM_GENRES, 1.2)
        secondary_genre = np.random.choice(np.arange(NUM_GENRES), p=genre_pop)
        if secondary_genre != primary_genre:
             CONTENT_FEATURES[content_id, secondary_genre] = 1

# --- RFP and DIBR Parameters ---
RFP_UPDATE_INTERVAL = 120 # Update region features and cooperative areas every 60 timeslots
SIMILARITY_THRESHOLD = 0.3 # beta
MAX_GAME_ITERATIONS = 5 # P
DELAY_TP = 500 # ms, delay from cloud server
DELAY_TS = 300 # ms, delay from serving satellite
DELAY_SS = 40 # ms, delay for inter-satellite link
MAX_COOP_HOPS = int((DELAY_TP - DELAY_TS) / DELAY_SS) # K

# --- Main execution ---
if __name__ == "__main__":
    # Storage constraint sizes to test
    # storage_sizes = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    storage_sizes = [200, 400, 600]
    
    # Create results directory if it doesn't exist
    results_dir = "results_RFP"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    all_results = []
    
    for storage_size in storage_sizes:
        print(f"\n{'='*20} Running simulation for Storage Size: {storage_size} {'='*20}")
        
        # Run simulation
        result = run_simulation(storage_size)
        all_results.append(result)
        
        # Save plots for this storage size
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(result['timeslot_costs'])), result['timeslot_costs'], 
                marker='o', linewidth=2, markersize=4, color='red', markerfacecolor='orange')
        plt.xlabel('Timeslot')
        plt.ylabel('Total System Cost')
        plt.title(f'Total System Cost per Timeslot (RFP Caching, Storage={storage_size})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/cost_over_time_rfp_storage_{storage_size}.png', dpi=300)
        plt.close()
        
        # Hit rates plot
        plt.figure(figsize=(15, 6))
        sat_names = list(result['satellite_table'].keys())
        hit_rates = []
        for sat_name in sat_names:
            stats = result['cache_hit_stats'][sat_name]
            total_lookups = stats['hits'] + stats['misses']
            hit_rate = stats['hits'] / total_lookups if total_lookups > 0 else 0
            hit_rates.append(hit_rate)

        plt.bar(range(len(sat_names)), hit_rates, color='lightblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('Satellite')
        plt.ylabel('Cache Hit Rate')
        plt.title(f'Cache Hit Rate per Satellite (RFP, Storage={storage_size})')
        plt.xticks(range(len(sat_names)), sat_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/hit_rates_rfp_storage_{storage_size}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY RESULTS FOR ALL STORAGE SIZES")
    print("="*80)
    print(f"{'Storage':<8} {'Total Cost':<12} {'Hit Rate':<10} {'Avg Latency':<12} {'Avg ISL Hops':<13} {'DIBR Cost':<11} {'Network Cost':<12} {'Cache Misses':<12}")
    print("-" * 92)

    for result in all_results:
        total_misses = sum(stats['misses'] for stats in result['cache_hit_stats'].values())
        print(f"{result['storage']:<8} "
            f"{result['total_cost']:<12.0f} "
            f"{result['hit_rate']:<10.3f} "
            f"{result['avg_latency']:<12.4f} "
            f"{result['avg_isl_hops']:<13.3f} "
            f"{result['dibr_cost']:<11.0f} "
            f"{result['network_cost']:<12.0f} "
            f"{total_misses*50:<12}"
            f"{result['total_requests']:<12}")

    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'Storage': result['storage'],
            'Total_Cost': result['total_cost'],
            'Hit_Rate': result['hit_rate'],
            'Avg_Latency': result['avg_latency'],
            'Avg_ISL_Hops': result['avg_isl_hops'],
            'DIBR_Cost': result['dibr_cost'],
            'Network_Cost': result['network_cost'],
            'Total_Cache_Misses': sum(stats['misses'] for stats in result['cache_hit_stats'].values()),
            'Total_Requests': result['total_requests']
        }
        for result in all_results
    ])
    
    results_df.to_csv(f'{results_dir}/rfp_simulation_results.csv', index=False)
    
    print(f"\nAll results saved to {results_dir}/")
    print("RFP simulation completed successfully!")


# print("\n" + "="*80)
# print("SUMMARY RESULTS FOR ALL STORAGE SIZES")
# print("="*80)
# print(f"{'Storage':<8} {'Total Cost':<12} {'Hit Rate':<10} {'Avg Latency':<12} {'Avg ISL Hops':<13} {'DIBR Cost':<11} {'Network Cost':<12} {'Cache Misses':<12}")
# print("-" * 92)

# for result in all_results:
#     total_misses = sum(stats['misses'] for stats in result['cache_hit_stats'].values())
#     print(f"{result['storage']:<8} "
#           f"{result['total_cost']:<12.0f} "
#           f"{result['hit_rate']:<10.3f} "
#           f"{result['avg_latency']:<12.4f} "
#           f"{result['avg_isl_hops']:<13.3f} "
#           f"{result['dibr_cost']:<11.0f} "
#           f"{result['network_cost']:<12.0f} "
#           f"{total_misses:<12}")

# # Save results to CSV
# results_df = pd.DataFrame([
#     {
#         'Storage': result['storage'],
#         'Total_Cost': result['total_cost'],
#         'Hit_Rate': result['hit_rate'],
#         'Avg_Latency': result['avg_latency'],
#         'Avg_ISL_Hops': result['avg_isl_hops'],
#         'DIBR_Cost': result['dibr_cost'],
#         'Network_Cost': result['network_cost'],
#         'Total_Cache_Misses': sum(stats['misses'] for stats in result['cache_hit_stats'].values())
#     }
#     for result in all_results
# ])