# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

from container_0617 import User, Satellite, GroundStation
from container_0617 import generate_zipf_distribution, calculate_rate_mbps, SPEED_OF_LIGHT_KM_S
from mpl_toolkits.basemap import Basemap

# --- Simulation Parameters ---
video_size_per_view_mb = 60
Total_timeslot = 240
# storage_constraint_Z_values = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]  # Storage constraints for satellites
storage_constraint_Z_values = [600] 

# --- Video Content Structure & Popularity Parameters ---
NUM_VIDEOS = 2500
VIEWS_PER_VIDEO = 16  # 4, 8, 12, 16, 20, 24, 28, 32
TOTAL_VIEWS = NUM_VIDEOS * VIEWS_PER_VIDEO
ZIPF_ALPHA = 0.8

# --- Poisson Process Parameters ---
arrival_rate_lambda = 60
departure_probability = 0.15

# Store results for all configurations
all_results = {}

# Function to get the cost of fetching/accessing a single view
def get_tau_j(view_j, sat, nearest_gs, hops_to_gs=0):
    cost_serving, cost_isl_hop, cost_gs_fetch, cost_miss_penalty = 5, 10, 20, 50
    if sat.is_view_cached(view_j): 
        return cost_serving
    elif nearest_gs: 
        return cost_miss_penalty + cost_gs_fetch + (hops_to_gs * cost_isl_hop) + cost_serving
    else: 
        print(f"Warning: No GS connection for view {view_j} on satellite {sat.sat_name}.")
        return float('inf')
    
def calculate_local_popularity(satellite, view_id, timeslot):
    recency = 1.0 / (timeslot - satellite.last_access_time.get(view_id, timeslot - 1) + 1)
    frequency = satellite.access_frequency.get(view_id, 0)
    return frequency + recency

def calculate_synthesis_benefit(satellite, view_to_add, D):
    if view_to_add in satellite.cache_state: return 0
    new_cache_state = satellite.cache_state | {view_to_add}
    newly_synthesizable = 0
    for partner_view in satellite.cache_state:
        dist = abs(view_to_add - partner_view)
        if 1 < dist <= D:
            for i in range(min(view_to_add, partner_view) + 1, max(view_to_add, partner_view)):
                already_possible = False
                for v1 in satellite.cache_state:
                    for v2 in satellite.cache_state:
                        if v1 < i < v2 and abs(v1-v2) <= D:
                            already_possible = True
                            break
                    if already_possible: break
                if not already_possible:
                    newly_synthesizable += 1
    return newly_synthesizable

def calculate_request_latency(sat, user, time, view_sets, nearest_gs, hops_to_gs, sat_table, ordered_names):
    view_size_mbit = user.video_size_mb * 8
    max_latency = 0.0
    dist_s_u = sat.distance_to_user(time, user)
    prop_s_u = dist_s_u / SPEED_OF_LIGHT_KM_S
    rate_s_u_mbps = 650
    if rate_s_u_mbps == 0: return float('inf')
    tx_s_u = view_size_mbit / rate_s_u_mbps
    
    for view in view_sets.get('local', set()):
        latency = prop_s_u + tx_s_u
        if latency > max_latency: max_latency = latency

    for view in view_sets.get('isl', set()):
        isl_prop_latency = 1000 / SPEED_OF_LIGHT_KM_S
        isl_tx_latency = view_size_mbit / (sat.isl_data_rate_gbps * 1000)
        latency = isl_prop_latency + isl_tx_latency + prop_s_u + tx_s_u
        if latency > max_latency: max_latency = latency

    if nearest_gs:
        for view in view_sets.get('gs', set()):
            connected_sat = sat
            dist_gs_s = nearest_gs.calculate_distance_to_satellite(connected_sat.lat[time], connected_sat.lon[time], connected_sat.alt[time])
            prop_gs_s = dist_gs_s / SPEED_OF_LIGHT_KM_S
            rate_gs_s_mbps = 2500
            if rate_gs_s_mbps == 0: continue
            tx_gs_s = view_size_mbit / rate_gs_s_mbps
            isl_hop_latency = hops_to_gs * ( (1000 / SPEED_OF_LIGHT_KM_S) + (view_size_mbit / (sat.isl_data_rate_gbps * 1000)) )
            latency = prop_gs_s + tx_gs_s + isl_hop_latency + prop_s_u + tx_s_u
            if latency > max_latency: max_latency = latency
    return max_latency

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
    m.scatter(gs_x, gs_y, c='red', s=100, marker='^', label='Ground Stations', edgecolors='black', linewidths=1)
    user_lats, user_lons = [user.lat for user in user_list], [user.lon for user in user_list]
    user_x, user_y = m(user_lons, user_lats)
    m.scatter(user_x, user_y, c='green', s=30, marker='s', label='Users', edgecolors='darkgreen', linewidths=0.5, alpha=0.7)
    sat_colors = plt.cm.tab10(np.linspace(0, 1, len(satellite_table)))
    for i, (sat_name, sat) in enumerate(satellite_table.items()):
        try:
            if timeslot_to_visualize < len(sat.lat):
                sat_lat, sat_lon = sat.lat.iloc[timeslot_to_visualize], sat.lon.iloc[timeslot_to_visualize]
                sat_x, sat_y = m(sat_lon, sat_lat)
                m.scatter(sat_x, sat_y, c=[sat_colors[i]], s=80, marker='o', edgecolors='black', linewidths=1, alpha=0.8)
                plt.annotate(sat_name, (sat_x, sat_y), xytext=(5, 5), textcoords='offset points', fontsize=6, color='blue', fontweight='bold')
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

video_indices, view_angle_indices = list(range(NUM_VIDEOS)), list(range(VIEWS_PER_VIDEO))

##################################################
#          INITIALIZE SYSTEM COMPONENTS          #
##################################################

# Load users and ground stations (only once)
users_df = pd.read_csv('data/users.csv')
all_users = [User(row['id'], row['lat'], row['lon'], row['x'], row['y'], row['z'], video_size_mb=video_size_per_view_mb) for _, row in users_df.iterrows()]
all_users.sort()

ground_station_locations = [
    # Europe
    # {"id": "GS_EU_FR_VILL", "name": "Villenave d'Ornon", "lat": 44.78, "lon": -0.55}, # France
    # {"id": "GS_EU_DE_AERZ", "name": "Aerzen", "lat": 52.05, "lon": 9.26},         # Germany
    # # {"id": "GS_EU_DE_USIN", "name": "Usingen", "lat": 50.33, "lon": 8.53},        # Germany
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

ground_station_list = []
for _, row in pd.DataFrame(ground_station_locations).iterrows():
    gs = GroundStation(row['id'], row['name'], row['lat'], row['lon'], total_views=TOTAL_VIEWS, view_size_mb=video_size_per_view_mb)
    ground_station_list.append(gs)

# Load satellite data (only once)
satellite_data_files = {}
for file in os.listdir('data/starlink118/satellite_trace'):
    if file.endswith('.csv'):
        satellite_data = pd.read_csv(f'data/starlink118/satellite_trace/{file}')
        satellite_name = file.split('_')[0]
        satellite_data_files[satellite_name] = satellite_data

print(f"Starting simulation with {len(storage_constraint_Z_values)} different storage constraints...")

# Main loop for different storage constraint values
for storage_idx, storage_constraint_Z in enumerate(storage_constraint_Z_values):
    print(f"\n{'='*80}")
    print(f"Running simulation {storage_idx + 1}/{len(storage_constraint_Z_values)} with storage constraint: {storage_constraint_Z}")
    print(f"{'='*80}")
    
    # Initialize satellites for this storage constraint
    satellite_table = {}
    for satellite_name, satellite_data in satellite_data_files.items():
        sat = Satellite(satellite_name, satellite_data, storage_constraint_Z=storage_constraint_Z, total_views=TOTAL_VIEWS, view_size_mb=video_size_per_view_mb)
        satellite_table[satellite_name] = sat

    # Initialize users for this run
    initial_active_user_count = 150
    active_user_list = copy.deepcopy(all_users[:initial_active_user_count])
    inactive_user_list = copy.deepcopy(all_users[initial_active_user_count:])

    print("Initializing satellite caches based on Zipf popularity...")
    random_seed = 42
    for sat_name, sat in satellite_table.items():
        sat.access_frequency, sat.last_access_time = {}, {}
        initialize_zipf_cache(sat, storage_constraint_Z, NUM_VIDEOS, VIEWS_PER_VIDEO, ZIPF_ALPHA, random_seed)
        for view_id in sat.cache_state:
            sat.access_frequency[view_id], sat.last_access_time[view_id] = 1, 0

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
            print(f'=========== Storage {storage_constraint_Z} | Time slot {i:03d} | Active Users: {len(active_user_list)} ===========')
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
        
        # 4. SIMULATE REQUESTS AND CACHING DECISIONS
        timeslot_total_cost = 0
        sat_names_ordered = sorted(list(satellite_table.keys()))
        
        for sat_idx, sat_name in enumerate(sat_names_ordered):
            sat = satellite_table[sat_name]
            if not sat.serving_users: continue
            sat_cost = 0
            
            # Find nearest ground station via expanding search
            nearest_gs, hops_to_gs = None, -1
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
                latency = calculate_request_latency(sat, user, i, view_sets_for_latency, nearest_gs, hops_to_gs, satellite_table, sat_names_ordered)
                if latency != float('inf'): latencies_this_timeslot.append(latency)
                    
            satellite_costs[sat_name] += sat_cost
            timeslot_total_cost += sat_cost

        total_requests_over_simulation += number_of_requests_this_slot
        timeslot_costs.append(timeslot_total_cost)
        total_isl_hops_over_simulation += hops_this_timeslot
        total_requests_with_isl_over_simulation += requests_with_isl_this_timeslot

        average_latency_per_timeslot.append(sum(latencies_this_timeslot) / len(latencies_this_timeslot) if latencies_this_timeslot else 0)
        average_hops_per_timeslot.append(hops_this_timeslot / requests_with_isl_this_timeslot if requests_with_isl_this_timeslot > 0 else 0)

    # Store results for this storage constraint
    total_system_cost = sum(timeslot_costs)
    overall_hits = sum(s['hits'] for s in cache_hit_stats.values())
    overall_reqs = sum(s['hits'] + s['misses'] for s in cache_hit_stats.values())
    overall_hit_rate = overall_hits / overall_reqs if overall_reqs > 0 else 0
    
    all_latencies = [lat for lat in average_latency_per_timeslot if lat > 0]
    overall_avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    overall_avg_isl_hops = total_isl_hops_over_simulation / total_requests_with_isl_over_simulation if total_requests_with_isl_over_simulation > 0 else 0
    
    all_results[storage_constraint_Z] = {
        'total_system_cost': total_system_cost,
        'total_requests': total_requests_over_simulation,
        'overall_hit_rate': overall_hit_rate,
        'overall_avg_latency': overall_avg_latency,
        'overall_avg_isl_hops': overall_avg_isl_hops,
        'total_dibr_synthesis_cost': total_dibr_synthesis_cost,
        'network_cost_breakdown': network_cost_breakdown.copy(),
        'total cache misses cost': network_cost_breakdown['miss_penalty'],
        'timeslot_costs': timeslot_costs.copy(),
        'satellite_costs': satellite_costs.copy(),
        'cache_hit_stats': cache_hit_stats.copy(),
        'average_latency_per_timeslot': average_latency_per_timeslot.copy()
    }
    
    # FINAL PERFORMANCE SUMMARY for this storage constraint
    print("="*50 + f"\nFINAL PERFORMANCE SUMMARY (Storage: {storage_constraint_Z})\n" + "="*50)
    print(f'Total system cost: {total_system_cost:.2f}')
    print(f'Total requests processed: {total_requests_over_simulation}')
    print(f'Overall cache hit rate: {overall_hit_rate:.3f}')
    print(f'Overall average latency: {overall_avg_latency:.4f} seconds')
    print(f'Overall average ISL hops: {overall_avg_isl_hops:.3f}')
    print(f'Total DIBR synthesis cost: {total_dibr_synthesis_cost:.2f}')
    print(f'total cache misses cost: {network_cost_breakdown["miss_penalty"]:.2f}')
    
    # Save plots for this storage constraint
    plt.figure(figsize=(12, 6))
    plt.plot(range(Total_timeslot), timeslot_costs, marker='o', linewidth=1.5, markersize=4, label='Total Cost')
    plt.title(f'Total System Cost per Timeslot (Storage: {storage_constraint_Z})')
    plt.xlabel('Timeslot'); plt.ylabel('Total Cost'); plt.grid(True, alpha=0.4); plt.legend(); plt.tight_layout()
    # plt.savefig(f'cost_over_time_storage_{storage_constraint_Z}.png', dpi=300)
    plt.close()

# Generate comparison plots across all storage constraints
print("\n" + "="*80)
print("GENERATING COMPARISON PLOTS ACROSS ALL STORAGE CONSTRAINTS")
print("="*80)

# Plot 1: Total System Cost vs Storage Constraint
storage_values = list(all_results.keys())
total_costs = [all_results[s]['total_system_cost'] for s in storage_values]
hit_rates = [all_results[s]['overall_hit_rate'] for s in storage_values]
avg_latencies = [all_results[s]['overall_avg_latency'] for s in storage_values]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(storage_values, total_costs, marker='o', linewidth=2, markersize=8)
plt.title('Total System Cost vs Storage Constraint')
plt.xlabel('Storage Constraint (Z)')
plt.ylabel('Total System Cost')
plt.grid(True, alpha=0.4)

plt.subplot(1, 3, 2)
plt.plot(storage_values, hit_rates, marker='s', linewidth=2, markersize=8, color='green')
plt.title('Cache Hit Rate vs Storage Constraint')
plt.xlabel('Storage Constraint (Z)')
plt.ylabel('Overall Cache Hit Rate')
plt.grid(True, alpha=0.4)

plt.subplot(1, 3, 3)
plt.plot(storage_values, avg_latencies, marker='^', linewidth=2, markersize=8, color='purple')
plt.title('Average Latency vs Storage Constraint')
plt.xlabel('Storage Constraint (Z)')
plt.ylabel('Average Latency (seconds)')
plt.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('storage_constraint_comparison_mobile.png', dpi=300, bbox_inches='tight')
# plt.show()

# Save summary results to CSV
summary_df = pd.DataFrame([
    {
        'Storage_Constraint': s,
        'Total_System_Cost': all_results[s]['total_system_cost'],
        'Total_Requests': all_results[s]['total_requests'],
        'Overall_Hit_Rate': all_results[s]['overall_hit_rate'],
        'Average_Latency': all_results[s]['overall_avg_latency'],
        'Average_ISL_Hops': all_results[s]['overall_avg_isl_hops'],
        'DIBR_Synthesis_Cost': all_results[s]['total_dibr_synthesis_cost'],
        'Total Cache miss': all_results[s]['total cache misses cost']
    }
    for s in storage_values
])
summary_df.to_csv('storage_constraint_results_summary_mobile.csv', index=False)

print(f"\nSimulation completed for all {len(storage_constraint_Z_values)} storage constraints.")
print("Results saved to 'storage_constraint_results_summary.csv'")
print("Comparison plots saved to 'storage_constraint_comparison.png'")
