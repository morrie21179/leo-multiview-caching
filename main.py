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
storage_constraint_Z_list = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
# storage_constraint_Z_list = [600]

# --- Video Content Structure & Popularity Parameters ---
NUM_VIDEOS = 2500
VIEWS_PER_VIDEO = 16
TOTAL_VIEWS = NUM_VIDEOS * VIEWS_PER_VIDEO
ZIPF_ALPHA = 0.8

# --- Poisson Process Parameters ---
arrival_rate_lambda = 60       # 20, 108
departure_probability = 0.15   # 0.15, 0.35

# Function to get the cost of fetching/accessing a single view
def get_tau_j(view_j, sat, nearest_gs, hops_to_gs=0):
    cost_serving, cost_isl_hop, cost_gs_fetch, cost_miss_penalty = 5, 10, 20, 50  # 3, 10, 4, 50
    if sat.is_view_cached(view_j): 
        return cost_serving
    elif view_j in sat.neighbor_caches: 
        return cost_isl_hop + cost_serving
    elif nearest_gs: 
        return cost_miss_penalty + cost_gs_fetch + (hops_to_gs * cost_isl_hop) + cost_serving
    else: 
        # This case implies no GS connection, which should be handled by the logic
        # that finds the nearest GS. A large penalty signifies failure.
        return float('inf')
    
def calculate_local_popularity(satellite, view_id, timeslot):
    """
    Calculates a score representing P_n(v) for a view in the cache.
    A lower score is less popular.
    """
    # A simple but effective metric for local popularity: recency + frequency
    recency = 1.0 / (timeslot - satellite.last_access_time.get(view_id, timeslot - 1) + 1)
    frequency = satellite.access_frequency.get(view_id, 0)
    return frequency + recency

def calculate_synthesis_benefit(satellite, view_to_add, D):
    """
    Calculates Delta_fetch(u): the number of NEW views that can be synthesized
    by adding view_to_add. A higher number is better.
    """
    if view_to_add in satellite.cache_state: return 0

    # Create a temporary new cache state to evaluate the benefit
    new_cache_state = satellite.cache_state | {view_to_add}
    newly_synthesizable = 0

    # Check the gaps created by the newly added view
    for partner_view in satellite.cache_state:
        dist = abs(view_to_add - partner_view)
        if 1 < dist <= D:
            # Check each view in the gap
            for i in range(min(view_to_add, partner_view) + 1, max(view_to_add, partner_view)):
                # Is this view already synthesizable by another pair?
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

def run_phase2_gs_swap(satellite, timeslot, D=3):
    if not satellite.cache_state: return 0

    min_score, view_to_drop = float('inf'), -1
    for view_id in satellite.cache_state:
        recency = 1.0 / (timeslot - satellite.last_access_time.get(view_id, timeslot - 1) + 1)
        frequency = satellite.access_frequency.get(view_id, 0)
        local_score = frequency + recency
        if local_score < min_score:
            min_score = local_score
            view_to_drop = view_id
    
    global_pop_dist = generate_zipf_distribution(TOTAL_VIEWS, ZIPF_ALPHA)
    if not any(global_pop_dist): return 0

    candidate_indices = np.argsort(global_pop_dist)[-int(TOTAL_VIEWS * 0.1):]
    
    best_view_to_fetch, max_fetch_score = -1, -1.0
    gamma = 0.5

    for u in candidate_indices:
        if u in satellite.cache_state or u in satellite.neighbor_caches: continue
        
        p_gs_score = global_pop_dist[u]
        min_dist_to_cache = min([abs(u - v) for v in satellite.cache_state] or [D + 1])
        delta_fetch = min_dist_to_cache if min_dist_to_cache > 1 else 0
        
        fetch_score = p_gs_score + gamma * (delta_fetch / D)
        if fetch_score > max_fetch_score:
            max_fetch_score = fetch_score
            best_view_to_fetch = u

    if best_view_to_fetch != -1 and view_to_drop is not None:
        # print(f"    [P2 ACTION] Sat {satellite.sat_name}: Swapping {view_to_drop} for {best_view_to_fetch}")
        satellite.evict_view(view_to_drop)
        satellite.cache_view(best_view_to_fetch)
        satellite.access_frequency[best_view_to_fetch] = 1
        satellite.last_access_time[best_view_to_fetch] = timeslot
        return 20
        
    return 10

# --- Latency Calculation Function ---
def calculate_request_latency(sat, user, time, view_sets, nearest_gs, hops_to_gs, sat_table, ordered_names):
    """
    Calculates the real latency for a user's request based on the DP plan.
    Latency = max(latency of each transmitted view).
    This function provides a reasonable and well-structured model for latency.
    """
    view_size_mbit = user.video_size_mb * 8
    max_latency = 0.0

    # Calculate Downlink (Satellite -> User) latency components once
    dist_s_u = sat.distance_to_user(time, user)
    prop_s_u = dist_s_u / SPEED_OF_LIGHT_KM_S
    # The data rate is simplified to a constant value for this simulation.
    # A more detailed model could calculate this dynamically.
    rate_s_u_mbps = 650
    if rate_s_u_mbps == 0: 
        return float('inf')
    tx_s_u = view_size_mbit / rate_s_u_mbps
    
    # Calculate latency for each transmitted view based on its source
    for view in view_sets.get('local', set()):
        latency = prop_s_u + tx_s_u
        if latency > max_latency: 
            max_latency = latency

    for view in view_sets.get('isl', set()):
        # Assume 1-hop for simplicity for neighbor cache
        isl_prop_latency = 1000 / SPEED_OF_LIGHT_KM_S # Avg distance for ISL
        isl_tx_latency = view_size_mbit / (sat.isl_data_rate_gbps * 1000)
        latency = isl_prop_latency + isl_tx_latency + prop_s_u + tx_s_u
        if latency > max_latency: max_latency = latency

    if nearest_gs:
        for view in view_sets.get('gs', set()):
            # Find the satellite actually connecting to the GS
            connected_sat = sat
            if hops_to_gs > 0:
                # This is a simplification; a real system would trace the exact path.
                # We'll use the original satellite's position as a proxy.
                pass 
            
            dist_gs_s = nearest_gs.calculate_distance_to_satellite(connected_sat.lat[time], connected_sat.lon[time], connected_sat.alt[time])
            prop_gs_s = dist_gs_s / SPEED_OF_LIGHT_KM_S
            # Uplink rate is simplified to a constant value.
            rate_gs_s_mbps = 2500 
            if rate_gs_s_mbps == 0: continue
            tx_gs_s = view_size_mbit / rate_gs_s_mbps
            
            isl_hop_latency = hops_to_gs * ( (1000 / SPEED_OF_LIGHT_KM_S) + (view_size_mbit / (sat.isl_data_rate_gbps * 1000)) )
            
            latency = prop_gs_s + tx_gs_s + isl_hop_latency + prop_s_u + tx_s_u
            if latency > max_latency: max_latency = latency

    return max_latency

def visualize_leo_satellite_movement(satellite_table, ground_station_list, user_list, timeslot_to_visualize=0):
    """
    Visualize LEO satellite positions, ground stations, and users on Earth for a specific timeslot
    
    Args:
        satellite_table: Dictionary of satellite objects
        ground_station_list: List of ground station objects
        user_list: List of user objects
        timeslot_to_visualize: Which timeslot to visualize (default: 0)
    """
    import matplotlib.pyplot as plt
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Create map projection
    m = Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80,
                llcrnrlon=-180, urcrnrlon=180, resolution='c')
    
    # Draw map features
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.fillcontinents(color='lightgray', lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')
    
    # Draw parallels and meridians
    m.drawparallels(np.arange(-80, 81, 20), labels=[1,0,0,0], fontsize=8)
    m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1], fontsize=8)
    
    # Plot ground stations
    gs_lats = [gs.lat for gs in ground_station_list]
    gs_lons = [gs.lon for gs in ground_station_list]
    gs_x, gs_y = m(gs_lons, gs_lats)
    
    m.scatter(gs_x, gs_y, c='red', s=100, marker='^', 
              label='Ground Stations', edgecolors='black', linewidths=1)
    
    # Plot users
    user_lats = [user.lat for user in user_list]
    user_lons = [user.lon for user in user_list]
    user_x, user_y = m(user_lons, user_lats)
    
    m.scatter(user_x, user_y, c='green', s=30, marker='s', 
              label='Users', edgecolors='darkgreen', linewidths=0.5, alpha=0.7)
    
    # Plot satellites for the specified timeslot
    sat_colors = plt.cm.tab10(np.linspace(0, 1, len(satellite_table)))
    
    for i, (sat_name, sat) in enumerate(satellite_table.items()):
        try:
            # Get satellite position using the get_position method from container.py
            if timeslot_to_visualize < len(sat.lat):
                sat_lat = sat.lat.iloc[timeslot_to_visualize]
                sat_lon = sat.lon.iloc[timeslot_to_visualize]
                
                # Convert to map projection
                sat_x, sat_y = m(sat_lon, sat_lat)
                
                # Plot satellite
                m.scatter(sat_x, sat_y, c=[sat_colors[i]], s=80, marker='o', 
                         edgecolors='black', linewidths=1, alpha=0.8)
                
                # Add satellite label
                plt.annotate(sat_name, (sat_x, sat_y), xytext=(5, 5), 
                            textcoords='offset points', fontsize=6, 
                            color='blue', fontweight='bold')
        except Exception as e:
            print(f"Error plotting satellite {sat_name}: {e}")
            continue
    
    # Add legend
    plt.legend(loc='lower left')
    
    # Add title
    plt.title(f'LEO Satellite Network, Ground Stations, and Users at Timeslot {timeslot_to_visualize}\n'
              f'Satellites: {len(satellite_table)}, Ground Stations: {len(ground_station_list)}, Users: {len(user_list)}', 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_satellite_movement_animation(satellite_table, ground_station_list, user_list, total_timeslots=24):
    """
    Create an animation showing satellite movement over time with ground stations and users
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create map projection
    m = Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80,
                llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=ax)
    
    # Draw static map features
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.fillcontinents(color='lightgray', lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')
    m.drawparallels(np.arange(-80, 81, 20), labels=[1,0,0,0], fontsize=8)
    m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1], fontsize=8)
    
    # Plot ground stations (static)
    gs_lats = [gs.lat for gs in ground_station_list]
    gs_lons = [gs.lon for gs in ground_station_list]
    gs_x, gs_y = m(gs_lons, gs_lats)
    m.scatter(gs_x, gs_y, c='red', s=100, marker='^', 
              label='Ground Stations', edgecolors='black', linewidths=1)
    
    # Plot users (static)
    user_lats = [user.lat for user in user_list]
    user_lons = [user.lon for user in user_list]
    user_x, user_y = m(user_lons, user_lats)
    m.scatter(user_x, user_y, c='green', s=30, marker='s', 
              label='Users', edgecolors='darkgreen', linewidths=0.5, alpha=0.7)
    
    # Initialize satellite scatter plot
    sat_colors = plt.cm.tab10(np.linspace(0, 1, len(satellite_table)))
    satellite_scatters = {}
    
    # Create individual scatter plots for each satellite
    for i, sat_name in enumerate(satellite_table.keys()):
        scatter = ax.scatter([], [], s=120, marker='o', c=[sat_colors[i]], 
                           edgecolors='black', linewidths=2, alpha=0.9, 
                           label=f'Sat-{i+1}' if i < 5 else "")
        satellite_scatters[sat_name] = scatter
    
    # Title text
    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes, 
                        ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    def animate(frame):
        valid_satellites = 0
        
        for sat_name, sat in satellite_table.items():
            try:
                # Use the satellite data directly from pandas DataFrame
                if frame < len(sat.lat):
                    sat_lat = sat.lat.iloc[frame]
                    sat_lon = sat.lon.iloc[frame]
                    
                    # Validate coordinates
                    if pd.notna(sat_lat) and pd.notna(sat_lon) and \
                       (-90 <= sat_lat <= 90) and (-180 <= sat_lon <= 180):
                        
                        # Convert to map projection
                        sat_x, sat_y = m(sat_lon, sat_lat)
                        
                        # Update satellite position
                        if sat_name in satellite_scatters:
                            satellite_scatters[sat_name].set_offsets([[sat_x, sat_y]])
                            valid_satellites += 1
                    else:
                        # Clear invalid position
                        if sat_name in satellite_scatters:
                            satellite_scatters[sat_name].set_offsets(np.empty((0, 2)))
                else:
                    # Frame beyond data range
                    if sat_name in satellite_scatters:
                        satellite_scatters[sat_name].set_offsets(np.empty((0, 2)))
                        
            except Exception as e:
                if frame == 0:
                    print(f"Error animating satellite {sat_name}: {e}")
                if sat_name in satellite_scatters:
                    satellite_scatters[sat_name].set_offsets(np.empty((0, 2)))
        
        # Update title
        title_text.set_text(f'LEO Satellite Movement - Timeslot {frame}/{total_timeslots-1}\n'
                           f'Valid Satellites: {valid_satellites}/{len(satellite_table)}, '
                           f'Ground Stations: {len(ground_station_list)}, Users: {len(user_list)}')
        
        return list(satellite_scatters.values()) + [title_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=total_timeslots, 
                                  interval=1000, blit=False, repeat=True)
    
    # Add legend
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    plt.tight_layout()
    
    # Save animation
    try:
        print("Saving animation... This may take a moment.")
        anim.save('satellite_movement_animation.gif', writer='pillow', fps=1, dpi=100)
        print("Animation saved as 'satellite_movement_animation.gif'")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    return fig, anim

def initialize_zipf_cache(satellite, storage_constraint, num_videos, views_per_video, zipf_alpha, random_seed=None):
    """
    Initializes cache based on a two-tiered Zipf popularity model using a
    single alpha factor.
    """
    if random_seed is not None:
        # Use a seed unique to each satellite for varied initial caches
        random.seed(random_seed + hash(satellite.sat_name))
    
    satellite.cache_state.clear()
    
    # Generate the popularity distributions internally
    video_pop_dist = generate_zipf_distribution(num_videos, zipf_alpha)
    angle_pop_dist = generate_zipf_distribution(views_per_video, zipf_alpha)
    
    video_indices = np.arange(num_videos)
    view_angle_indices = np.arange(views_per_video)
    
    views_to_cache = set()
    # Fill the cache up to its storage constraint
    while len(views_to_cache) < storage_constraint:
        # Select a video based on its popularity
        video_id = np.random.choice(video_indices, p=video_pop_dist)
        # Select a view angle based on its popularity
        angle_id = np.random.choice(view_angle_indices, p=angle_pop_dist)
        
        # Calculate the unique global ID for the view
        global_view_id = video_id * views_per_video + angle_id
        views_to_cache.add(global_view_id)
        
    satellite.cache_state = views_to_cache

def cache_content_with_eviction(satellite, content_id, timeslot, num_videos, views_per_video, zipf_alpha):
    """
    Cache content with eviction logic based on a hybrid popularity score,
    using a single Zipf factor.
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
    # Generate the video popularity distribution for scoring
    video_pop_dist = generate_zipf_distribution(num_videos, zipf_alpha)
    
    popularity_scores = {}
    for view_id in satellite.cache_state:
        # Factor 1: Global Video Popularity (from Zipf)
        video_id = view_id // views_per_video
        # Ensure video_id is within the bounds of the distribution list
        if video_id < len(video_pop_dist):
            global_pop_score = video_pop_dist[video_id]
        else:
            global_pop_score = 0
        
        # Factor 2: Local Access Frequency
        frequency_score = satellite.access_frequency.get(view_id, 0)
        
        # Factor 3: Recency
        last_access = satellite.last_access_time.get(view_id, 0)
        recency_score = 1.0 / (timeslot - last_access + 1)

        # Combined popularity score
        popularity_scores[view_id] = (0.5 * global_pop_score) + \
                                     (0.3 * frequency_score) + \
                                     (0.2 * recency_score)

    if popularity_scores:
        view_to_evict = min(popularity_scores, key=popularity_scores.get)
        satellite.evict_view(view_to_evict)
        
        if view_to_evict in satellite.last_access_time:
            del satellite.last_access_time[view_to_evict]
        if view_to_evict in satellite.access_frequency:
            del satellite.access_frequency[view_to_evict]
            
    satellite.cache_view(content_id)
    satellite.last_access_time[content_id] = timeslot
    satellite.access_frequency[content_id] = 1

def run_simulation(storage_constraint_Z):
    """
    Run the full simulation for a given storage constraint.
    Returns the simulation results.
    """
    print(f"\n{'='*60}")
    print(f"RUNNING SIMULATION WITH STORAGE CONSTRAINT: {storage_constraint_Z}")
    print(f"{'='*60}")
    
    # Pre-calculate popularity distributions
    video_indices = list(range(NUM_VIDEOS))
    view_angle_indices = list(range(VIEWS_PER_VIDEO))

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

    # --- Initialize Satellites ---
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

    # --- Initialize Caches and Popularity Tracking ---
    print("Initializing satellite caches based on Zipf popularity...")
    random_seed = 42
    for sat_name, sat in satellite_table.items():
        sat.neighbor_caches = set()
        sat.access_frequency = {}
        sat.last_access_time = {}
        
        initialize_zipf_cache(
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

    # --- Initialize Ground Stations ---
    print("\nGenerating ground station data...")
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

        # # # Africa
        # {"id": "GS_AF_NG_IKIR", "name": "Ikire", "lat": 7.38, "lon": 4.18},           # Nigeria
        # {"id": "GS_AF_NG_LEKK", "name": "Lekki", "lat": 6.45, "lon": 4.09},           # Nigeria
        # {"id": "GS_AF_Kenya", "name": "Nairobi", "lat": -1.29, "lon": 36.82},         # Kenya
        {"id": "GS_AF_MZ", "name": "Matola", "lat": -25.92, "lon": 32.42},            # MZ

        # Middle East
        # {"id": "GS_ME_OMAN", "name": "Murayjat", "lat": 23.72, "lon": 57.78},  # 23.716. Longitude : 57.783
        # {"id": "GS_ME_TURKEY", "name": "Muallim", "lat": 36.92, "lon": 38.06}, # 36.921. Longitude : 38.059
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

    # Initialize Cost Tracking
    satellite_costs = {sat_name: 0 for sat_name in satellite_table.keys()}
    timeslot_costs = []
    cache_hit_stats = {sat_name: {'hits': 0, 'misses': 0} for sat_name in satellite_table.keys()}

    total_requests_over_simulation = 0
    average_latency_per_timeslot = []
    average_hops_per_timeslot = []
    total_isl_hops_over_simulation = 0
    total_requests_with_isl_over_simulation = 0

    timeslot_costs_with_phase2 = []
    timeslot_costs_without_phase2 = []

    # Initialize trackers for the new metrics
    total_dibr_synthesis_cost = 0
    network_cost_breakdown = {
        'miss_penalty': 0, 
        'gs_fetch': 0, 
        'isl': 0, 
        'serving': 0
    }

    ########################################################################################################
    ####################################### MAIN SIMULATION LOOP ###########################################
    ########################################################################################################

    for i in range(Total_timeslot):
        if i % 60 == 0:
            print(f'=========== Time slot {i:03d} | Active Users: {len(active_user_list)} ===========')

        latencies_this_timeslot = []
        hops_this_timeslot = 0
        requests_with_isl_this_timeslot = 0
        number_of_requests = 0
        phase1_total_cost_this_slot = 0
        phase2_additional_cost_this_slot = 0

        # 1. SIMULATE USER DEPARTURES
        users_departing = []
        for user in active_user_list:
            if random.random() < departure_probability:
                users_departing.append(user)
        
        for user in users_departing:
            active_user_list.remove(user)
            inactive_user_list.append(user)

        # 2. SIMULATE USER ARRIVALS (POISSON PROCESS)
        num_new_arrivals = np.random.poisson(arrival_rate_lambda)
        num_new_arrivals = min(num_new_arrivals, len(inactive_user_list))
        
        if num_new_arrivals > 0:
            for _ in range(num_new_arrivals):
                new_user = inactive_user_list.pop()
                active_user_list.append(new_user)
                
        # Reset connections for all active users
        for user in active_user_list:
            user.sat = None
            user.elevation = 0
        
        for sat in satellite_table.values():
            sat.serving_users = []
        
        # 3. FIND BEST (CLOSEST) SATELLITE FOR EACH *ACTIVE* USER
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

        # Update neighbor cache info
        sat_names = list(satellite_table.keys())
        for sat in satellite_table.values():
            current_idx = sat_names.index(sat.sat_name)
            neighbor_cache_union = set()
            if current_idx > 0:
                neighbor_cache_union.update(satellite_table[sat_names[current_idx - 1]].cache_state)
            if current_idx < len(sat_names) - 1:
                neighbor_cache_union.update(satellite_table[sat_names[current_idx + 1]].cache_state)
            sat.neighbor_caches = neighbor_cache_union

        # 4. SIMULATE REQUESTS AND CACHING DECISIONS
        timeslot_total_cost = 0
        sat_names_ordered = sorted(list(satellite_table.keys()))
        
        for sat_idx, sat_name in enumerate(sat_names_ordered):
            sat = satellite_table[sat_name]
            if not sat.serving_users:
                continue

            sat_cost = 0
            
            # Find nearest ground station via expanding search
            nearest_gs, hops_to_gs = None, -1
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

            cost_p1_for_this_sat = 0
            for user_id in sat.serving_users:
                user = next((u for u in active_user_list if u.user_id == user_id), None)
                if not user: continue

                success, request_data = user.generate_request(
                    NUM_VIDEOS, VIEWS_PER_VIDEO, ZIPF_ALPHA, view_range_B=5
                )

                if not success:
                    continue

                number_of_requests += 1
                h, l = request_data['h'], request_data['l']
                D, alpha, T_DIBR = 3, 1.5, 0

                # --- DYNAMIC PROGRAMMING with detailed cost tracking ---           
                mu, prev, mu_dibr = {}, {}, {}
                tau_h = get_tau_j(h, sat, nearest_gs, hops_to_gs)
                mu[h], prev[h], mu_dibr[h] = tau_h, None, 0

                for j in range(h + 1, l + 1):
                    tau_j = get_tau_j(j, sat, nearest_gs, hops_to_gs)
                    min_cost, best_pred = float('inf'), None
                    dibr_cost_for_min_step = 0

                    for k in range(max(j - D, h), j + 1):
                        dibr_cost = (alpha * (j - k) + T_DIBR) * (j - k - 1) if k < j else 0
                        current_cost = mu.get(k, float('inf')) + dibr_cost + tau_j
                        if current_cost < min_cost:
                            min_cost, best_pred = current_cost, k
                            dibr_cost_for_min_step = mu_dibr.get(k, 0) + dibr_cost
                    
                    mu[j], prev[j] = min_cost, best_pred
                    mu_dibr[j] = dibr_cost_for_min_step

                request_cost = mu.get(l, float('inf'))
                
                if request_cost == float('inf'):
                    continue

                # Accumulate DIBR cost for this request
                dibr_cost_for_request = mu_dibr.get(l, 0)
                total_dibr_synthesis_cost += dibr_cost_for_request
                
                sat_cost += request_cost
                cost_p1_for_this_sat += request_cost

                # --- EXECUTE THE PLAN ---
                transfer_points, curr = [], l
                while curr is not None and curr >= h:
                    transfer_points.append(curr)
                    curr = prev.get(curr)
                
                V_fetch = set(transfer_points)
                V_DIBR, V_local_hit, V_ISL, V_ground_station = set(), set(), set(), set()
                
                # This logic to find V_DIBR is for understanding, but cost is from DP
                sorted_transfers = sorted(list(V_fetch))
                for idx in range(len(sorted_transfers) - 1):
                    for synth_view in range(sorted_transfers[idx] + 1, sorted_transfers[idx+1]):
                        V_DIBR.add(synth_view)
                
                for v in V_fetch:
                    if sat.is_view_cached(v):
                        V_local_hit.add(v)
                        cache_hit_stats[sat.sat_name]['hits'] += 1
                        sat.last_access_time[v] = i
                        sat.access_frequency[v] = sat.access_frequency.get(v, 0) + 1
                    elif v in sat.neighbor_caches:
                        V_ISL.add(v)
                        cache_hit_stats[sat.sat_name]['hits'] += 1
                        cache_content_with_eviction(sat, v, i, NUM_VIDEOS, VIEWS_PER_VIDEO, ZIPF_ALPHA)
                    else:
                        V_ground_station.add(v)
                        cache_hit_stats[sat.sat_name]['misses'] += 1
                        if nearest_gs and nearest_gs.has_view(v):
                            nearest_gs.transmit_to_satellite(sat, [v])
                        cache_content_with_eviction(sat, v, i, NUM_VIDEOS, VIEWS_PER_VIDEO, ZIPF_ALPHA)

                # Breakdown Network Costs based on the decision
                cost_serving, cost_isl_hop, cost_gs_fetch, cost_miss_penalty = 5, 10, 20, 50
                network_cost_breakdown['serving'] += len(V_fetch) * cost_serving
                network_cost_breakdown['isl'] += len(V_ISL) * cost_isl_hop
                if V_ground_station:
                    network_cost_breakdown['miss_penalty'] += len(V_ground_station) * cost_miss_penalty
                    network_cost_breakdown['gs_fetch'] += len(V_ground_station) * cost_gs_fetch
                    if hops_to_gs > 0:
                        network_cost_breakdown['isl'] += len(V_ground_station) * hops_to_gs * cost_isl_hop
            
                # Calculate total ISL hops for this request
                hops_for_this_request = (1 * len(V_ISL)) + (hops_to_gs * len(V_ground_station))
                if hops_for_this_request > 0:
                    hops_this_timeslot += hops_for_this_request
                    requests_with_isl_this_timeslot += 1

                # Prepare view sets for latency calculation
                view_sets_for_latency = {'local': V_local_hit, 'isl': V_ISL, 'gs': V_ground_station}
                latency = calculate_request_latency(sat, user, i, view_sets_for_latency, nearest_gs, hops_to_gs, satellite_table, sat_names_ordered)
                if latency != float('inf'):
                    latencies_this_timeslot.append(latency)

            satellite_costs[sat.sat_name] += sat_cost
            timeslot_total_cost += sat_cost

            phase1_total_cost_this_slot += sat_cost
            if i != 239:
                if nearest_gs:
                    additional_cost = run_phase2_gs_swap(sat, i)
                    phase2_additional_cost_this_slot += additional_cost

        served_users_this_timeslot = sum(len(sat.serving_users) for sat in satellite_table.values())
        timeslot_costs.append(timeslot_total_cost)
        total_requests_over_simulation += number_of_requests
        
        # Update global ISL hop counters
        total_isl_hops_over_simulation += hops_this_timeslot
        total_requests_with_isl_over_simulation += requests_with_isl_this_timeslot

        timeslot_costs_without_phase2.append(phase1_total_cost_this_slot)
        timeslot_costs_with_phase2.append(phase1_total_cost_this_slot + phase2_additional_cost_this_slot)

        # Calculate average latency for this timeslot
        if latencies_this_timeslot:
            avg_lat = sum(latencies_this_timeslot) / len(latencies_this_timeslot)
            average_latency_per_timeslot.append(avg_lat)
        else:
            average_latency_per_timeslot.append(0)

        # Calculate average ISL hops for this timeslot
        if requests_with_isl_this_timeslot > 0:
                avg_hops_for_slot = hops_this_timeslot / requests_with_isl_this_timeslot
                average_hops_per_timeslot.append(avg_hops_for_slot)
        else:
            average_hops_per_timeslot.append(0)

    # Return results dictionary
    return {
        'storage_constraint': storage_constraint_Z,
        'total_system_cost': sum(satellite_costs.values()),
        'total_requests': total_requests_over_simulation,
        'overall_hit_rate': sum(stats['hits'] for stats in cache_hit_stats.values()) / 
                           (sum(stats['hits'] for stats in cache_hit_stats.values()) + 
                            sum(stats['misses'] for stats in cache_hit_stats.values())) 
                           if (sum(stats['hits'] for stats in cache_hit_stats.values()) + 
                               sum(stats['misses'] for stats in cache_hit_stats.values())) > 0 else 0,
        'overall_average_latency': sum(average_latency_per_timeslot) / Total_timeslot if Total_timeslot > 0 else 0,
        'overall_avg_isl_hops': total_isl_hops_over_simulation / total_requests_with_isl_over_simulation 
                               if total_requests_with_isl_over_simulation > 0 else 0,
        'total_dibr_synthesis_cost': total_dibr_synthesis_cost,
        'total_network_transmission_cost': sum(network_cost_breakdown.values()),
        'network_cost_breakdown': network_cost_breakdown.copy(),
        'satellite_costs': satellite_costs.copy(),
        'cache_hit_stats': cache_hit_stats.copy()
    }

# Main execution
if __name__ == "__main__":
    # Store results for all storage constraints
    all_results = []
    
    # Run simulations for each storage constraint
    for storage_constraint in storage_constraint_Z_list:
        result = run_simulation(storage_constraint)
        all_results.append(result)
        
        # Print summary for this storage constraint
        print(f"\nSUMMARY FOR STORAGE CONSTRAINT {storage_constraint}:")
        print(f"Total System Cost: {result['total_system_cost']:.2f}")
        print(f"Total Requests: {result['total_requests']}")
        print(f"Overall Hit Rate: {result['overall_hit_rate']:.3f}")
        print(f"Overall Average Latency: {result['overall_average_latency']:.4f} seconds")
        print(f"Overall Avg ISL Hops: {result['overall_avg_isl_hops']:.3f}")
        print(f"Total DIBR Synthesis Cost: {result['total_dibr_synthesis_cost']:.2f}")
        print(f"Total Network Transmission Cost: {result['total_network_transmission_cost']:.2f}")
    
    # Print comparative summary
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY FOR ALL STORAGE CONSTRAINTS")
    print(f"{'='*80}")
    print(f"{'Storage':<8} {'Total Cost':<12} {'Hit Rate':<10} {'Avg Latency':<12} {'ISL Hops':<10} {'DIBR Cost':<12} {'Network Cost':<12}")
    print(f"{'-'*80}")
    
    for result in all_results:
        print(f"{result['storage_constraint']:<8} "
              f"{result['total_system_cost']:<12.2f} "
              f"{result['overall_hit_rate']:<10.3f} "
              f"{result['overall_average_latency']:<12.4f} "
              f"{result['overall_avg_isl_hops']:<10.3f} "
              f"{result['total_dibr_synthesis_cost']:<12.2f} "
              f"{result['total_network_transmission_cost']:<12.2f}")
    
    # Save results to CSV for further analysis
    results_df = pd.DataFrame([
        {
            'Storage_Constraint': r['storage_constraint'],
            'Total_System_Cost': r['total_system_cost'],
            'Total_Requests': r['total_requests'],
            'Overall_Hit_Rate': r['overall_hit_rate'],
            'Overall_Average_Latency': r['overall_average_latency'],
            'Overall_Avg_ISL_Hops': r['overall_avg_isl_hops'],
            'Total_DIBR_Synthesis_Cost': r['total_dibr_synthesis_cost'],
            'Total_Network_Transmission_Cost': r['total_network_transmission_cost'],
            'Miss_Penalty_Cost': r['network_cost_breakdown']['miss_penalty'],
            'GS_Fetch_Cost': r['network_cost_breakdown']['gs_fetch'],
            'ISL_Cost': r['network_cost_breakdown']['isl'],
            'Serving_Cost': r['network_cost_breakdown']['serving']
        }
        for r in all_results
    ])
    
    results_df.to_csv('simulation_results_by_storage_constraint.csv', index=False)
    print(f"\nResults saved to 'simulation_results_by_storage_constraint.csv'")
    
    # Create visualization plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    storage_values = [r['storage_constraint'] for r in all_results]
    
    # Plot 1: Total System Cost vs Storage Constraint
    total_costs = [r['total_system_cost'] for r in all_results]
    ax1.plot(storage_values, total_costs, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Storage Constraint (Z)')
    ax1.set_ylabel('Total System Cost')
    ax1.set_title('Total System Cost vs Storage Constraint')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Hit Rate vs Storage Constraint
    hit_rates = [r['overall_hit_rate'] for r in all_results]
    ax2.plot(storage_values, hit_rates, 'g-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Storage Constraint (Z)')
    ax2.set_ylabel('Overall Hit Rate')
    ax2.set_title('Cache Hit Rate vs Storage Constraint')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average Latency vs Storage Constraint
    avg_latencies = [r['overall_average_latency'] for r in all_results]
    ax3.plot(storage_values, avg_latencies, 'r-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Storage Constraint (Z)')
    ax3.set_ylabel('Average Latency (seconds)')
    ax3.set_title('Average Latency vs Storage Constraint')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: DIBR vs Network Costs
    dibr_costs = [r['total_dibr_synthesis_cost'] for r in all_results]
    network_costs = [r['total_network_transmission_cost'] for r in all_results]
    
    width = 0.35
    x = np.arange(len(storage_values))
    ax4.bar(x - width/2, dibr_costs, width, label='DIBR Synthesis Cost', alpha=0.8)
    ax4.bar(x + width/2, network_costs, width, label='Network Transmission Cost', alpha=0.8)
    ax4.set_xlabel('Storage Constraint (Z)')
    ax4.set_ylabel('Cost')
    ax4.set_title('DIBR vs Network Costs by Storage Constraint')
    ax4.set_xticks(x)
    ax4.set_xticklabels(storage_values)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simulation_results_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved to 'simulation_results_comparison.png'")
