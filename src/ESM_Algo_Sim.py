# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from container_0617 import User, Satellite, GroundStation
from container_0617 import generate_zipf_distribution, calculate_rate_mbps, SPEED_OF_LIGHT_KM_S
from container_ESM import ESMMatching

# --- Simulation Parameters ---
video_size_per_view_mb = 60
Total_timeslot = 240
storage_constraint_Z = 500 # 125, "500", 875, 1250, 1625
# (5%:200, 10%:400, 15%:600, 20%:800, 25%:1000, 30%:1200,
#  35%:1400, 40%:1600, 45%:1800, 50%:2000 of total content size)

# --- Video Content Structure & Popularity Parameters ---
NUM_VIDEOS = 2500
VIEWS_PER_VIDEO = 16
TOTAL_VIEWS = NUM_VIDEOS * VIEWS_PER_VIDEO
ZIPF_ALPHA = 0.8

# --- Poisson Process Parameters ---
arrival_rate_lambda = 60       # 60(0.25), 108(0.45)
departure_probability = 0.15   # 0.15, 0.35

def initialize_esm_cache(satellite, storage_constraint, num_videos, views_per_video, zipf_alpha, random_seed=None):
    """Initialize satellite cache using ESM-compatible random initialization"""
    if random_seed is not None:
        random.seed(random_seed + hash(satellite.sat_name))
    
    satellite.cache_state.clear()
    
    # Generate popularity distribution
    total_views = num_videos * views_per_video
    popularity_dist = generate_zipf_distribution(total_views, zipf_alpha)
    
    if len(popularity_dist) == 0:
        return
    
    # Select contents based on popularity (biased random selection)
    available_views = list(range(total_views))
    
    # Cache the most popular contents with some randomness
    for _ in range(min(storage_constraint, len(available_views))):
        # Weighted random selection based on popularity
        selected_view = np.random.choice(available_views, p=popularity_dist)
        satellite.cache_state.add(selected_view)
        
        # Remove selected view and renormalize probabilities
        idx = available_views.index(selected_view)
        available_views.pop(idx)
        popularity_dist = np.delete(popularity_dist, idx)
        
        if len(popularity_dist) > 0:
            popularity_dist = popularity_dist / np.sum(popularity_dist)
        else:
            break

# Function to get the cost of fetching/accessing a single view (for DIBR compatibility)
def get_tau_j(view_j, sat, nearest_gs, hops_to_gs=0):
    cost_serving, cost_isl_hop, cost_gs_fetch, cost_miss_penalty = 5, 10, 20, 50
    if sat.is_view_cached(view_j): 
        return cost_serving
    elif view_j in sat.neighbor_caches: 
        return cost_isl_hop + cost_serving
    elif nearest_gs: 
        return cost_miss_penalty + cost_gs_fetch + (hops_to_gs * cost_isl_hop) + cost_serving
    else: 
        return cost_miss_penalty

def calculate_request_latency(sat, user, time, view_sets, nearest_gs, hops_to_gs, sat_table, ordered_names):
    """Calculate the real latency for a user's request based on the DP plan"""
    view_size_mbit = user.video_size_mb * 8
    max_latency = 0.0

    # Calculate Downlink (Satellite -> User) latency components once
    dist_s_u = sat.distance_to_user(time, user)
    prop_s_u = dist_s_u / SPEED_OF_LIGHT_KM_S
    rate_s_u_mbps = calculate_rate_mbps(
        sat.tx_power_watt, sat.antenna_gain_dbi, user.antenna_gain_dbi,
        dist_s_u, user.bandwidth_hz, sat.downlink_freq_ghz
    )
    if rate_s_u_mbps == 0: return float('inf')
    tx_s_u = view_size_mbit / rate_s_u_mbps
    
    # Calculate latency for each transmitted view based on its source
    for view in view_sets.get('local', set()):
        latency = prop_s_u + tx_s_u
        if latency > max_latency: max_latency = latency

    for view in view_sets.get('isl', set()):
        isl_prop_latency = 550 / SPEED_OF_LIGHT_KM_S
        isl_tx_latency = view_size_mbit / (sat.isl_data_rate_gbps * 1000)
        latency = isl_prop_latency + isl_tx_latency + prop_s_u + tx_s_u
        if latency > max_latency: max_latency = latency

    if nearest_gs:
        for view in view_sets.get('gs', set()):
            connected_sat = sat
            dist_gs_s = nearest_gs.calculate_distance_to_satellite(connected_sat.lat[time], connected_sat.lon[time], connected_sat.alt[time])
            prop_gs_s = dist_gs_s / SPEED_OF_LIGHT_KM_S
            rate_gs_s_mbps = nearest_gs.get_uplink_rate_to_sat(connected_sat, time)
            if rate_gs_s_mbps == 0: continue
            tx_gs_s = view_size_mbit / rate_gs_s_mbps
            
            isl_hop_latency = hops_to_gs * ( (550 / SPEED_OF_LIGHT_KM_S) + (view_size_mbit / (sat.isl_data_rate_gbps * 1000)) )
            
            latency = prop_gs_s + tx_gs_s + isl_hop_latency + prop_s_u + tx_s_u
            if latency > max_latency: max_latency = latency

    return max_latency

def cache_content_with_eviction(satellite, content_id, timeslot, num_videos, views_per_video, zipf_alpha):
    """Cache content with eviction logic - ESM compatible"""
    if satellite.is_view_cached(content_id):
        satellite.last_access_time[content_id] = timeslot
        satellite.access_frequency[content_id] = satellite.access_frequency.get(content_id, 0) + 1
        return

    if len(satellite.cache_state) < satellite.storage_constraint_Z:
        satellite.cache_view(content_id)
        satellite.last_access_time[content_id] = timeslot
        satellite.access_frequency[content_id] = 1
        return

    # Eviction logic using simple LRU for compatibility with ESM
    if satellite.cache_state:
        # Find least recently used item
        oldest_time = min(satellite.last_access_time.get(view, 0) for view in satellite.cache_state)
        view_to_evict = next(view for view in satellite.cache_state 
                           if satellite.last_access_time.get(view, 0) == oldest_time)
        
        satellite.evict_view(view_to_evict)
        if view_to_evict in satellite.last_access_time:
            del satellite.last_access_time[view_to_evict]
        if view_to_evict in satellite.access_frequency:
            del satellite.access_frequency[view_to_evict]
            
    satellite.cache_view(content_id)
    satellite.last_access_time[content_id] = timeslot
    satellite.access_frequency[content_id] = 1

##################################################
#          INITIALIZE SYSTEM COMPONENTS          #
##################################################

# Load filtered user data
users_df = pd.read_csv('data/users.csv')
all_users = []
for _, row in users_df.iterrows():
    user = User(row['id'], row['lat'], row['lon'], row['x'], row['y'], row['z'], video_size_mb=video_size_per_view_mb)
    all_users.append(user)
all_users.sort()

# Split users into active and inactive pools
initial_active_user_count = 150
active_user_list = all_users[:initial_active_user_count]
inactive_user_list = all_users[initial_active_user_count:]

# Initialize Satellites
satellite_table = {}
for file in os.listdir('data/starlink118/satellite_trace'):
    if file.endswith('.csv'):
        satellite_data = pd.read_csv(f'data/starlink118/satellite_trace/{file}')
        satellite_name = file.split('_')[0]
        sat = Satellite(satellite_name, satellite_data, 
                       storage_constraint_Z=storage_constraint_Z, 
                       total_views=TOTAL_VIEWS, 
                       view_size_mb=video_size_per_view_mb)
        satellite_table[satellite_name] = sat

# Initialize Caches using ESM-compatible method
print("Initializing satellite caches for ESM algorithm...")
random_seed = 42
for sat_name, sat in satellite_table.items():
    sat.neighbor_caches = set()
    sat.access_frequency = {}
    sat.last_access_time = {}
    
    initialize_esm_cache(
        sat,
        storage_constraint_Z,
        NUM_VIDEOS,
        VIEWS_PER_VIDEO,
        ZIPF_ALPHA,
        random_seed
    )
    
    for view_id in sat.cache_state:
        sat.access_frequency[view_id] = 1
        sat.last_access_time[view_id] = 0

# Initialize Ground Stations
print("\nGenerating ground station data...")
ground_station_list = []
ground_station_locations = [
    # North America
    {"id": "GS_US_W", "name": "Washington", "lat": 47.75, "lon": -120.74},
    {"id": "GS_US_C", "name": "Texas", "lat": 31.96, "lon": -99.90},
    {"id": "GS_US_E", "name": "Virginia", "lat": 37.43, "lon": -78.65},
    {"id": "GS_CAN_C", "name": "Manitoba", "lat": 55.00, "lon": -97.00},
    # Europe
    {"id": "GS_EU_UK", "name": "London", "lat": 51.50, "lon": -0.12},
    {"id": "GS_EU_DE", "name": "Frankfurt", "lat": 50.11, "lon": 8.68},
    {"id": "GS_EU_FR", "name": "Paris", "lat": 48.85, "lon": 2.35},
    {"id": "GS_EU_SP", "name": "Madrid", "lat": 40.41, "lon": -3.70},
    {"id": "GS_EU_IT", "name": "Rome", "lat": 41.90, "lon": 12.50},
    {"id": "GS_EU_NL", "name": "Amsterdam", "lat": 52.37, "lon": 4.90},
    # Asia
    {"id": "GS_AS_JP", "name": "Tokyo", "lat": 35.68, "lon": 139.69},
    {"id": "GS_AS_CN", "name": "Beijing", "lat": 39.90, "lon": 116.40},
    {"id": "GS_AS_SG", "name": "Singapore", "lat": 1.35, "lon": 103.81},
    {"id": "GS_AS_IN", "name": "Mumbai", "lat": 19.07, "lon": 72.87},
    # Oceania
    {"id": "GS_AU_SYD", "name": "Sydney", "lat": -33.86, "lon": 151.20},
    {"id": "GS_NZ_AUK", "name": "Auckland", "lat": -36.84, "lon": 174.76}
]

ground_stations_df = pd.DataFrame(ground_station_locations)
for _, row in ground_stations_df.iterrows():
    gs = GroundStation(
        row['id'], row['name'], row['lat'], row['lon'],
        total_views=TOTAL_VIEWS,
        view_size_mb=video_size_per_view_mb
    )
    ground_station_list.append(gs)
print(f"Created {len(ground_station_list)} ground stations")

# Initialize ESM Algorithm
esm_algorithm = ESMMatching(
    satellite_table,
    ground_station_list,
    active_user_list,
    NUM_VIDEOS,
    VIEWS_PER_VIDEO,
    ZIPF_ALPHA
)

# Initialize Cost Tracking
satellite_costs = {sat_name: 0 for sat_name in satellite_table.keys()}
timeslot_costs = []
cache_hit_stats = {sat_name: {'hits': 0, 'misses': 0} for sat_name in satellite_table.keys()}

total_requests_over_simulation = 0
average_latency_per_timeslot = []
average_hops_per_timeslot = []

# ESM-specific tracking
esm_iterations_per_timeslot = []
esm_swap_operations = []

########################################################################################################
####################################### MAIN SIMULATION LOOP ###########################################
########################################################################################################

for i in range(Total_timeslot):
    print(f'=========== Time slot {i:03d} | Active Users: {len(active_user_list)} ===========')

    latencies_this_timeslot = []
    hops_this_timeslot = 0
    requests_with_isl_this_timeslot = 0
    number_of_requests = 0

    # 1. SIMULATE USER DEPARTURES
    users_departing = []
    for user in active_user_list:
        if random.random() < departure_probability:
            users_departing.append(user)
    
    for user in users_departing:
        active_user_list.remove(user)
        inactive_user_list.append(user)
    
    if users_departing:
        print(f"--- {len(users_departing)} users departed.")

    # 2. SIMULATE USER ARRIVALS (POISSON PROCESS)
    num_new_arrivals = np.random.poisson(arrival_rate_lambda)
    num_new_arrivals = min(num_new_arrivals, len(inactive_user_list))
    
    if num_new_arrivals > 0:
        print(f"--- {num_new_arrivals} new users arrived.")
        for _ in range(num_new_arrivals):
            new_user = inactive_user_list.pop()
            active_user_list.append(new_user)
    
    # Update ESM algorithm with current active users
    esm_algorithm.users = active_user_list
            
    # Reset connections for all active users
    for user in active_user_list:
        user.sat = None
        user.elevation = 0
    
    for sat in satellite_table.values():
        sat.serving_users = []
    
    # 3. FIND BEST (CLOSEST) SATELLITE FOR EACH ACTIVE USER
    for user in active_user_list:
        best_sat = None
        min_distance = float('inf')
        
        for sat in satellite_table.values():
            is_covered = sat.connect_user(i, user)
            if is_covered:
                distance = sat.distance_to_user(i, user)
                if distance < min_distance:
                    min_distance = distance
                    best_sat = sat
        
        if best_sat:
            user.sat = best_sat
            best_sat.serving_users.append(user.user_id)

    # 4. RUN ESM ALGORITHM TO OPTIMIZE CACHE PLACEMENT
    print(f"Running ESM optimization for timeslot {i}")
    # if i % 10 == 0:  # Only run ESM every 10 timeslots to save computation
    #     try:
    #         esm_iterations = esm_algorithm.exchange_stable_matching(i, max_iterations=20)
    #     except Exception as e:
    #         print(f"ESM failed: {e}")
    #         esm_iterations = 0
    # else:
    #     esm_iterations = 0  # Skip ESM for most timeslots
    esm_iterations = esm_algorithm.exchange_stable_matching(i, max_iterations=3)
    esm_iterations_per_timeslot.append(esm_iterations)

    # Update neighbor cache info after ESM optimization
    sat_names = list(satellite_table.keys())
    for sat in satellite_table.values():
        current_idx = sat_names.index(sat.sat_name)
        neighbor_cache_union = set()
        if current_idx > 0:
            neighbor_cache_union.update(satellite_table[sat_names[current_idx - 1]].cache_state)
        if current_idx < len(sat_names) - 1:
            neighbor_cache_union.update(satellite_table[sat_names[current_idx + 1]].cache_state)
        sat.neighbor_caches = neighbor_cache_union

    # 5. SIMULATE REQUESTS AND CACHING DECISIONS
    timeslot_total_cost = 0
    sat_names_ordered = sorted(list(satellite_table.keys()))
    
    for sat_idx, sat_name in enumerate(sat_names_ordered):
        sat = satellite_table[sat_name]
        if not sat.serving_users:
            continue

        sat_cost = 0
        
        # Find nearest ground station with expanding search
        nearest_gs = None
        hops_to_gs = -1
        
        for hop in range(len(sat_names_ordered) // 2 + 1):
            indices_to_check = []
            if hop == 0:
                indices_to_check.append(sat_idx)
            else:
                if sat_idx - hop >= 0:
                    indices_to_check.append(sat_idx - hop)
                if sat_idx + hop < len(sat_names_ordered):
                    indices_to_check.append(sat_idx + hop)

            if not indices_to_check:
                continue

            visible_gs_options = []
            for check_idx in indices_to_check:
                candidate_sat = satellite_table[sat_names_ordered[check_idx]]
                for gs in ground_station_list:
                    if gs.is_satellite_in_view(candidate_sat, i):
                        dist_from_original_sat = gs.calculate_distance_to_satellite(
                            sat.lat.iloc[i], sat.lon.iloc[i], sat.alt.iloc[i]
                        )
                        visible_gs_options.append({'gs': gs, 'distance': dist_from_original_sat})
            
            if visible_gs_options:
                closest_option = min(visible_gs_options, key=lambda x: x['distance'])
                nearest_gs = closest_option['gs']
                hops_to_gs = hop
                break

        # Process requests for each user served by this satellite
        for user_id in sat.serving_users:
            user = next((u for u in active_user_list if u.user_id == user_id), None)
            if not user: continue

            # Generate request using existing logic
            success, request_data = user.generate_request(
                NUM_VIDEOS,
                VIEWS_PER_VIDEO,
                ZIPF_ALPHA,
                view_range_B=3
            )

            if not success:
                continue

            number_of_requests += 1
            h, l = request_data['h'], request_data['l']
            D, alpha, T_DIBR = 3, 1.5, 2

            # DYNAMIC PROGRAMMING for DIBR synthesis
            mu, prev = {}, {}
            tau_h = get_tau_j(h, sat, nearest_gs, hops_to_gs)
            mu[h], prev[h] = tau_h, None

            for j in range(h + 1, l + 1):
                tau_j = get_tau_j(j, sat, nearest_gs, hops_to_gs)
                min_cost, best_pred = float('inf'), None
                for k in range(max(j - D, h), j + 1):
                    dibr_cost = (alpha * (j - k) + T_DIBR) * (j - k - 1) if k < j else 0
                    current_cost = mu.get(k, float('inf')) + dibr_cost + tau_j
                    if current_cost < min_cost:
                        min_cost, best_pred = current_cost, k
                mu[j], prev[j] = min_cost, best_pred
            
            request_cost = mu.get(l, float('inf'))
            sat_cost += request_cost

            # EXECUTE THE PLAN
            transfer_points = []
            curr = l
            while curr is not None and curr >= h:
                transfer_points.append(curr)
                curr = prev.get(curr)
            
            V_fetch = set(transfer_points)
            V_DIBR = set()
            sorted_transfers = sorted(list(V_fetch))
            for idx in range(len(sorted_transfers) - 1):
                start_view = sorted_transfers[idx]
                end_view = sorted_transfers[idx+1]
                for synth_view in range(start_view + 1, end_view):
                    V_DIBR.add(synth_view)

            V_local_hit, V_ISL, V_ground_station = set(), set(), set()
            
            for v in V_fetch:
                current_timeslot = i
                if sat.is_view_cached(v):
                    V_local_hit.add(v)
                    cache_hit_stats[sat.sat_name]['hits'] += 1
                    sat.last_access_time[v] = current_timeslot
                    sat.access_frequency[v] = sat.access_frequency.get(v, 0) + 1
                elif v in sat.neighbor_caches:
                    V_ISL.add(v)
                    cache_hit_stats[sat.sat_name]['hits'] += 1
                    cache_content_with_eviction(sat, v, current_timeslot, NUM_VIDEOS, VIEWS_PER_VIDEO, ZIPF_ALPHA)
                else:
                    V_ground_station.add(v)
                    cache_hit_stats[sat.sat_name]['misses'] += 1
                    if nearest_gs and nearest_gs.has_view(v):
                        nearest_gs.transmit_to_satellite(sat, [v])
                    cache_content_with_eviction(sat, v, current_timeslot, NUM_VIDEOS, VIEWS_PER_VIDEO, ZIPF_ALPHA)
        
            # Calculate ISL hops for this request
            hops_for_this_request = (1 * len(V_ISL)) + (hops_to_gs * len(V_ground_station))
            if hops_for_this_request > 0:
                hops_this_timeslot += hops_for_this_request
                requests_with_isl_this_timeslot += 1

            # Prepare view sets for latency calculation
            view_sets_for_latency = {
                'local': V_local_hit,
                'isl': V_ISL,
                'gs': V_ground_station
            }
            
            # Calculate and store latency
            if request_cost != float('inf'):
                 latency = calculate_request_latency(sat, user, i, view_sets_for_latency, nearest_gs, hops_to_gs, satellite_table, sat_names_ordered)
                 if latency != float('inf'):
                     latencies_this_timeslot.append(latency)
                
        satellite_costs[sat.sat_name] += sat_cost
        timeslot_total_cost += sat_cost

    served_users_this_timeslot = sum(len(sat.serving_users) for sat in satellite_table.values())
    timeslot_costs.append(timeslot_total_cost)
    total_requests_over_simulation += number_of_requests

    # Calculate average latency for this timeslot
    if latencies_this_timeslot:
        avg_lat = sum(latencies_this_timeslot) / len(latencies_this_timeslot)
        average_latency_per_timeslot.append(avg_lat)
    else:
        average_latency_per_timeslot.append(0)

    # Calculate average ISL hops
    if requests_with_isl_this_timeslot > 0:
        avg_hops_for_slot = hops_this_timeslot / requests_with_isl_this_timeslot
        average_hops_per_timeslot.append(avg_hops_for_slot)
    else:
        average_hops_per_timeslot.append(0)

    print(f'Served users this timeslot: {served_users_this_timeslot}')
    print(f'Total cost this timeslot: {timeslot_total_cost}')
    print(f'ESM iterations: {esm_iterations}')
    print("-" * 20)

# FINAL PERFORMANCE SUMMARY
print("="*50)
print("FINAL PERFORMANCE SUMMARY - LEO COOPERATIVE CACHING WITH ESM ALGORITHM")
print("="*50)
total_system_cost = sum(satellite_costs.values())

for sat_name, cost in sorted(satellite_costs.items()):
    total_requests = cache_hit_stats[sat_name]['hits'] + cache_hit_stats[sat_name]['misses']
    hit_rate = cache_hit_stats[sat_name]['hits'] / total_requests if total_requests > 0 else 0
    sat = satellite_table[sat_name]
    cache_utilization = sat.get_cache_utilization()
    print(f'Satellite {sat_name}: Total cost = {cost}, Hit rate = {hit_rate:.3f}, '
          f'Total requests = {total_requests}, '
          f'Cache utilization = {cache_utilization:.3f} ({len(sat.cache_state)}/{sat.storage_constraint_Z}), '
          f'Storage: {sat.get_storage_used_mb():.1f}MB/{sat.get_storage_capacity_mb():.1f}MB')

print(f'\nGround Stations:')
for gs in ground_station_list:
    stats = gs.get_transmission_statistics()
    total_storage_gb = (gs.total_views * gs.view_size_mb) / 1024
    print(f'  {gs.station_id}: {gs.name} at ({gs.lat:.2f}, {gs.lon:.2f}) - '
          f'Transmissions: {stats["total_transmissions"]}, Views sent: {stats["total_views_transmitted"]}, '
          f'Data transmitted: {stats["total_data_transmitted_gb"]:.2f}GB, Total storage: {total_storage_gb:.1f}GB')

print(f'\nTotal system cost: {total_system_cost}')
print(f'Total requests processed during simulation: {total_requests_over_simulation}')
overall_hits = sum(stats['hits'] for stats in cache_hit_stats.values())
overall_requests = sum(stats['hits'] + stats['misses'] for stats in cache_hit_stats.values())
overall_hit_rate = overall_hits / overall_requests if overall_requests > 0 else 0
print(f'Overall cache hit rate: {overall_hit_rate:.3f}')

# ESM-specific statistics
total_esm_iterations = sum(esm_iterations_per_timeslot)
average_esm_iterations = total_esm_iterations / Total_timeslot
print(f'\nESM Algorithm Statistics:')
print(f'Total ESM iterations across all timeslots: {total_esm_iterations}')
print(f'Average ESM iterations per timeslot: {average_esm_iterations:.2f}')

# System storage statistics
total_system_storage_used_mb = sum(sat.get_storage_used_mb() for sat in satellite_table.values())
total_system_storage_capacity_mb = sum(sat.get_storage_capacity_mb() for sat in satellite_table.values())
print(f'Total system storage: {total_system_storage_used_mb:.1f}MB/{total_system_storage_capacity_mb:.1f}MB '
      f'({total_system_storage_used_mb/total_system_storage_capacity_mb*100:.1f}% utilized)')

# Create plotting data
sat_names = list(satellite_table.keys())
sat_costs = [satellite_costs[sat_name] for sat_name in sat_names]

# Plot 1: Total cost per satellite (bar chart)
plt.figure(figsize=(12, 6))
plt.bar(range(len(sat_names)), sat_costs, color='skyblue', edgecolor='navy', alpha=0.7)
plt.xlabel('Satellite')
plt.ylabel('Total Cost')
plt.title('Total Cost per Satellite (ESM Algorithm)')
plt.xticks(range(len(sat_names)), sat_names, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

for i, cost in enumerate(sat_costs):
    plt.text(i, cost + max(sat_costs) * 0.01, str(cost), 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('esm_total_cost_per_satellite.png', dpi=300, bbox_inches='tight')
# plt.show()

# Plot 2: Cost over time (line chart)
plt.figure(figsize=(12, 6))
plt.plot(range(Total_timeslot), timeslot_costs, marker='o', linewidth=2, 
         markersize=6, color='red', markerfacecolor='orange')
plt.xlabel('Timeslot')
plt.ylabel('Total Cost')
plt.title('Total System Cost per Timeslot (ESM Algorithm)')
plt.grid(True, alpha=0.3)
plt.xlim(-0.5, Total_timeslot - 0.5)
plt.tight_layout()
plt.savefig('esm_cost_over_time.png', dpi=300, bbox_inches='tight')
# plt.show()

# Plot 3: Cache hit rates per satellite
hit_rates = [cache_hit_stats[sat]['hits'] / (cache_hit_stats[sat]['hits'] + cache_hit_stats[sat]['misses']) 
             if (cache_hit_stats[sat]['hits'] + cache_hit_stats[sat]['misses']) > 0 else 0 
             for sat in sat_names]

plt.figure(figsize=(12, 6))
plt.bar(range(len(sat_names)), hit_rates, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
plt.xlabel('Satellite')
plt.ylabel('Cache Hit Rate')
plt.title('Cache Hit Rate per Satellite (ESM Algorithm)')
plt.xticks(range(len(sat_names)), sat_names, rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

for i, rate in enumerate(hit_rates):
    plt.text(i, rate + 0.01, f'{rate:.3f}', 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('esm_cache_hit_rates_per_satellite.png', dpi=300, bbox_inches='tight')
# plt.show()

# Plot 4: ESM iterations per timeslot
plt.figure(figsize=(12, 6))
plt.plot(range(Total_timeslot), esm_iterations_per_timeslot, marker='s', linewidth=2, 
         markersize=4, color='purple', alpha=0.7)
plt.xlabel('Timeslot')
plt.ylabel('ESM Iterations')
plt.title('ESM Algorithm Iterations per Timeslot')
plt.grid(True, alpha=0.3)
plt.xlim(-0.5, Total_timeslot - 0.5)
plt.tight_layout()
plt.savefig('esm_iterations_per_timeslot.png', dpi=300, bbox_inches='tight')
# plt.show()

# Plot 5: Average latency over time
plt.figure(figsize=(12, 6))
plt.plot(range(Total_timeslot), average_latency_per_timeslot, marker='.', linestyle='-', color='purple')
plt.xlabel('Timeslot')
plt.ylabel('Average Request Latency (seconds)')
plt.title('Average User Request Latency per Timeslot (ESM Algorithm)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('esm_average_latency_over_time.png', dpi=300, bbox_inches='tight')
# plt.show()

# Plot 6: Average ISL hops over time
plt.figure(figsize=(12, 6))
plt.plot(range(Total_timeslot), average_hops_per_timeslot, marker='.', linestyle='-', color='green')
plt.xlabel('Timeslot')
plt.ylabel('Average ISL Hops per Request')
plt.title('Average ISL Hops per Request (ESM Algorithm)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('esm_average_isl_hops_over_time.png', dpi=300, bbox_inches='tight')
# plt.show()

# Additional statistics
print(f'\nAdditional Performance Metrics:')
print(f'Average cost per timeslot: {total_system_cost / Total_timeslot:.2f}')
avg_requests_per_timeslot = total_requests_over_simulation / Total_timeslot
print(f'Average number of user requests per timeslot: {avg_requests_per_timeslot:.2f}')
print(f'Peak timeslot cost: {max(timeslot_costs)}')
print(f'Minimum timeslot cost: {min(timeslot_costs)}')

# Calculate and print the overall average latency
overall_average_latency = sum(average_latency_per_timeslot) / Total_timeslot
print(f'Overall average request latency across all timeslots: {overall_average_latency:.4f} seconds')
print(f'Average latency across each request: {sum(average_latency_per_timeslot) / total_requests_over_simulation:.4f} seconds')

# Calculate and print the overall average ISL hops per request
print(f'Overall average ISL hops per request across all timeslots: {sum(average_hops_per_timeslot) / Total_timeslot:.4f}')


