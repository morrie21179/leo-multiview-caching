# container.py

import numpy as np
import pandas as pd

EARTH_RADIUS = 6371 # in km
BOLTZMANN_K = 1.38e-23
NOISE_TEMP_K = 290
SPEED_OF_LIGHT_KM_S = 300000 #299792.458 # Speed of light in km/s
SPEED_OF_LIGHT = 300000000  # m/s

# --- New Standalone Function to Calculate Data Rate ---
# def calculate_rate_mbps(tx_power_watt, tx_gain_dbi, rx_gain_dbi, distance_km, bandwidth_hz, frequency_ghz):
#     if distance_km <= 0: 
#         return float('inf')
#     tx_gain_linear = 10**(tx_gain_dbi / 10)
#     rx_gain_linear = 10**(rx_gain_dbi / 10)
#     lambda_m = 0.3 / frequency_ghz
#     path_loss_linear = ((4 * np.pi * distance_km * 1000) / lambda_m)**2
#     if path_loss_linear == 0: 
#         return float('inf')
#     received_power_watt = (tx_power_watt * tx_gain_linear * rx_gain_linear) / path_loss_linear
#     noise_power_watt = BOLTZMANN_K * NOISE_TEMP_K * bandwidth_hz
#     snr = received_power_watt / noise_power_watt
#     rate_bps = bandwidth_hz * np.log2(1 + snr)
#     return rate_bps / 1e6
def calculate_rate_mbps(tx_power_watt, tx_gain_dbi, rx_gain_dbi, distance_km, bandwidth_hz, frequency_ghz, elevation_angle_deg=None):
    """
    Updated LEO satellite data rate calculation following the communication model:
    R = B * log_2(1 + (P * G) / σ²)
    """
    if distance_km <= 0:
        return float('inf')
    
    # Convert gains from dBi to linear
    tx_gain_linear = 10**(tx_gain_dbi / 10)
    rx_gain_linear = 10**(rx_gain_dbi / 10)
    
    # Calculate wavelength
    wavelength_m = 0.3 / frequency_ghz  # c/f in meters (c = 3e8 m/s, f in GHz)
    
    # Free space path loss
    path_loss_linear = ((4 * np.pi * distance_km * 1000) / wavelength_m)**2
    
    # Elevation angle compensation for atmospheric effects
    if elevation_angle_deg is not None and elevation_angle_deg < 90:
        # Additional atmospheric loss at low elevation angles (more realistic)
        atmospheric_loss_db = 0.3 / np.sin(np.radians(max(elevation_angle_deg, 5)))
        atmospheric_loss_linear = 10**(atmospheric_loss_db / 10)
    else:
        atmospheric_loss_linear = 1.0
    
    # Calculate received power
    received_power_watt = (tx_power_watt * tx_gain_linear * rx_gain_linear) / (path_loss_linear * atmospheric_loss_linear)
    
    # Calculate noise power with improved noise figure
    noise_figure_db = 3.0  # Modern receiver noise figure
    noise_figure_linear = 10**(noise_figure_db / 10)
    noise_power_watt = BOLTZMANN_K * NOISE_TEMP_K * bandwidth_hz * noise_figure_linear
    
    # Calculate SNR and Shannon capacity
    snr = received_power_watt / noise_power_watt
    if snr <= 0:
        return 0.0
    
    rate_bps = bandwidth_hz * np.log2(1 + snr)
    return rate_bps / 1e6  # Convert to Mbps

def calculate_elevation_angle(sat_lat, sat_lon, sat_alt, user_lat, user_lon):
    """
    Calculate elevation angle from user to satellite.
    Based on the formula from the slide: θ_{n,k}
    """
    # Convert to radians
    sat_lat_rad = np.radians(sat_lat)
    sat_lon_rad = np.radians(sat_lon)
    user_lat_rad = np.radians(user_lat)
    user_lon_rad = np.radians(user_lon)
    
    # Calculate great circle distance
    delta_lon = sat_lon_rad - user_lon_rad
    cos_central_angle = (np.sin(user_lat_rad) * np.sin(sat_lat_rad) + 
                        np.cos(user_lat_rad) * np.cos(sat_lat_rad) * np.cos(delta_lon))
    cos_central_angle = np.clip(cos_central_angle, -1.0, 1.0)
    central_angle = np.arccos(cos_central_angle)
    
    # Calculate elevation angle using the formula from the slide
    R = EARTH_RADIUS
    h = sat_alt
    
    # Distance formula: d = -R*sin(θ) + sqrt((R*sin(θ))² + h² + 2*h*R)
    # But we need to solve for θ given the central angle
    cos_elevation = (np.sin(central_angle) * R) / np.sqrt(R**2 + h**2 + 2*R*h*np.cos(central_angle))
    cos_elevation = np.clip(cos_elevation, 0.0, 1.0)
    elevation_angle = np.pi/2 - np.arccos(cos_elevation)
    
    return np.degrees(elevation_angle)

def calculate_satellite_user_distance_accurate(sat_lat, sat_lon, sat_alt, user_lat, user_lon):
    """
    Calculate accurate distance using the formula from the slide:
    d_{n,k} = -R*sin(θ_{n,k}) + sqrt((R*sin(θ_{n,k}))² + h² + 2*h*R)
    """
    elevation_rad = np.radians(calculate_elevation_angle(sat_lat, sat_lon, sat_alt, user_lat, user_lon))
    
    R = EARTH_RADIUS
    h = sat_alt
    theta = np.pi/2 - elevation_rad  # Convert elevation to angle from zenith
    
    # Using the exact formula from the slide
    distance = -R * np.sin(theta) + np.sqrt((R * np.sin(theta))**2 + h**2 + 2*h*R)
    
    return distance

def calculate_channel_gain(tx_gain_dbi, rx_gain_dbi, distance_km, frequency_ghz, elevation_angle_deg=None):
    """
    Calculate channel gain G_{n,k} including path loss and antenna gains.
    Uses free space path loss model with elevation angle compensation.
    """
    # Convert gains from dBi to linear
    tx_gain_linear = 10**(tx_gain_dbi / 10)
    rx_gain_linear = 10**(rx_gain_dbi / 10)
    
    # Calculate wavelength
    wavelength_m = SPEED_OF_LIGHT / (frequency_ghz * 1e9)
    
    # Free space path loss (Friis equation)
    path_loss_linear = ((4 * np.pi * distance_km * 1000) / wavelength_m)**2
    
    # Elevation angle compensation (atmospheric loss increases at low elevation)
    if elevation_angle_deg is not None and elevation_angle_deg < 90:
        # Additional atmospheric loss at low elevation angles
        atmospheric_loss_db = 0.5 / np.sin(np.radians(max(elevation_angle_deg, 5)))  # Avoid division by zero
        atmospheric_loss_linear = 10**(atmospheric_loss_db / 10)
    else:
        atmospheric_loss_linear = 1.0
    
    # Total channel gain
    channel_gain = (tx_gain_linear * rx_gain_linear) / (path_loss_linear * atmospheric_loss_linear)
    return channel_gain

def calculate_noise_power(bandwidth_hz, noise_temp_k=None, noise_figure_db=None):
    """
    Calculate noise power σ² = k*T*B*NF
    """
    if noise_temp_k is None:
        noise_temp_k = LEOCommParams.NOISE_TEMP_K
    if noise_figure_db is None:
        noise_figure_db = LEOCommParams.NOISE_FIGURE_DB
    
    noise_figure_linear = 10**(noise_figure_db / 10)
    noise_power_watt = BOLTZMANN_K * noise_temp_k * bandwidth_hz * noise_figure_linear
    
    return noise_power_watt

def calculate_leo_rate_mbps(tx_power_watt, tx_gain_dbi, rx_gain_dbi, distance_km, 
                           bandwidth_hz, frequency_ghz, elevation_angle_deg=None,
                           noise_temp_k=None, noise_figure_db=None):
    """
    Updated LEO satellite data rate calculation based on the communication model:
    R_{n,k} = B_{n,k} * log_2(1 + (P_n * G_{n,k}) / σ²)
    
    This follows the exact formula from the slide you provided.
    """
    if distance_km <= 0:
        return float('inf')
    
    # Calculate channel gain G_{n,k}
    channel_gain = calculate_channel_gain(tx_gain_dbi, rx_gain_dbi, distance_km, 
                                        frequency_ghz, elevation_angle_deg)
    
    # Calculate received power P_n * G_{n,k}
    received_power_watt = tx_power_watt * channel_gain
    
    # Calculate noise power σ²
    noise_power_watt = calculate_noise_power(bandwidth_hz, noise_temp_k, noise_figure_db)
    
    # Calculate SNR
    snr = received_power_watt / noise_power_watt
    
    # Shannon capacity: R = B * log_2(1 + SNR)
    if snr <= 0:
        return 0.0
    
    rate_bps = bandwidth_hz * np.log2(1 + snr)
    rate_mbps = rate_bps / 1e6
    
    return rate_mbps

def calculate_isl_rate_mbps(distance_km):
    """
    Calculate Inter-Satellite Link (ISL) data rate.
    R_{n,n+1} = B^{ISL} * log_2(1 + (P_{ISL} * G_{n,n+1}) / σ²)
    """
    # ISL typically uses optical links or high-frequency RF
    if distance_km <= 0:
        return float('inf')
    
    # For optical ISL, use simplified model (very high capacity)
    if distance_km <= 5000:  # Typical ISL range
        return LEOCommParams.ISL_DATA_RATE_GBPS * 1000  # Convert to Mbps
    else:
        # Use RF ISL model for longer distances
        return calculate_leo_rate_mbps(
            tx_power_watt=LEOCommParams.ISL_TX_POWER_WATT,
            tx_gain_dbi=LEOCommParams.ISL_ANTENNA_GAIN_DBI,
            rx_gain_dbi=LEOCommParams.ISL_ANTENNA_GAIN_DBI,
            distance_km=distance_km,
            bandwidth_hz=100e6,  # 100 MHz for RF ISL
            frequency_ghz=60,    # V-band for ISL
            elevation_angle_deg=90  # Direct line-of-sight
        )
    
def calculate_gs_uplink_rate_mbps(distance_km, elevation_angle_deg):
    """
    Calculate Ground Station to LEO uplink rate.
    R_{g,n} = B^{GS} * log_2(1 + (P_{GS} * G_{g,s}) / σ²)
    """
    return calculate_leo_rate_mbps(
        tx_power_watt=LEOCommParams.GS_TX_POWER_WATT,
        tx_gain_dbi=LEOCommParams.GS_ANTENNA_GAIN_DBI,
        rx_gain_dbi=LEOCommParams.SATELLITE_ANTENNA_GAIN_DBI,
        distance_km=distance_km,
        bandwidth_hz=LEOCommParams.UPLINK_BANDWIDTH_MHZ * 1e6,
        frequency_ghz=LEOCommParams.UPLINK_FREQ_GHZ,
        elevation_angle_deg=elevation_angle_deg
    )

def get_modern_satellite_params():
    """
    Return updated satellite parameters based on recent papers and specifications.
    """
    return {
        'tx_power_watt': LEOCommParams.SATELLITE_TX_POWER_WATT,
        'antenna_gain_dbi': LEOCommParams.SATELLITE_ANTENNA_GAIN_DBI,
        'downlink_freq_ghz': LEOCommParams.DOWNLINK_FREQ_GHZ,
        'isl_data_rate_gbps': LEOCommParams.ISL_DATA_RATE_GBPS,
        'altitude_km': LEOCommParams.SATELLITE_ALTITUDE_KM
    }


def get_modern_user_params():
    """
    Return updated user terminal parameters.
    """
    return {
        'antenna_gain_dbi': LEOCommParams.USER_ANTENNA_GAIN_DBI,
        'bandwidth_hz': LEOCommParams.DOWNLINK_BANDWIDTH_MHZ * 1e6
    }


def get_modern_gs_params():
    """
    Return updated ground station parameters.
    """
    return {
        'tx_power_watt': LEOCommParams.GS_TX_POWER_WATT,
        'antenna_gain_dbi': LEOCommParams.GS_ANTENNA_GAIN_DBI,
        'bandwidth_hz': LEOCommParams.UPLINK_BANDWIDTH_MHZ * 1e6,
        'uplink_freq_ghz': LEOCommParams.UPLINK_FREQ_GHZ
    }

def generate_zipf_distribution(N, alpha):
    """Generates a Zipf probability distribution for N items."""
    if N <= 0: return []
    x = np.arange(1, N + 1)
    weights = x ** (-alpha)
    return weights / np.sum(weights) if np.sum(weights) > 0 else []

# create a class to represent a user
class User:
    def __init__(self, user_id, lat, lon, x, y, z, video_size_mb=6):
        self.user_id, self.lat, self.lon, self.x, self.y, self.z = user_id, lat, lon, x, y, z
        self.video_size_mb = video_size_mb

        # serving satellite information
        self.elevation = 0
        self.sat = None
        # Communication parameters for user terminals
        # self.antenna_gain_dbi = 4.0 # Typical for a user terminal
        # self.bandwidth_hz = 20 * 1e6 # 20 MHz bandwidth

        self.antenna_gain_dbi = 12.0       # Modern flat-panel phased arrays (up from 4 dBi)
        self.bandwidth_hz = 250 * 1e6      # 250 MHz bandwidth (up from 20 MHz)
        self.terminal_type = "flat_panel"  # Modern terminal type

    def __str__(self): 
        return f'User {self.user_id}'
    
    def __lt__(self, other): 
        return self.user_id < other.user_id
    
    # Other methods (generate_request, etc.) remain the same...
    # def generate_request(self, view_index=None, view_range_B=3):
    #     if self.sat is None:
    #         return False, "No satellite connection available"
        
    #     if view_index is None:
    #         view_index = np.random.randint(0, 1000)
    #     if view_range_B is None:
    #         view_range_B = np.random.randint(5, 20)
        
    #     return self.make_request(self.sat.sat_name, view_index, view_range_B)
    def generate_request(self, num_videos, views_per_video, zipf_alpha, view_range_B=3):
        """
        Generates a realistic request using a SINGLE zipf_alpha for both
        video and view angle popularity distributions.
        """
        if self.sat is None:
            return False, None

        # 1. Generate popularity distributions internally
        video_pop_dist = generate_zipf_distribution(num_videos, zipf_alpha)
        angle_pop_dist = generate_zipf_distribution(views_per_video, zipf_alpha)
        
        # 2. Select a video and view angle based on the distributions
        video_id = np.random.choice(np.arange(num_videos), p=video_pop_dist)
        center_angle = np.random.choice(np.arange(views_per_video), p=angle_pop_dist)
        
        # 3. Define the request range, clamped within the video's boundaries
        half_range = view_range_B // 2
        start_angle = max(0, center_angle - half_range)
        end_angle = min(views_per_video - 1, center_angle + half_range)

        # 4. Convert angle range to global view ID range
        h = video_id * views_per_video + start_angle
        l = video_id * views_per_video + end_angle
        
        request_data = {'h': h, 'l': l}
        
        return True, request_data


    def make_request(self, satellite_name, view_index, view_range_B):
        if self.sat is None:
            return False, "No satellite connection available"
        
        half_range = view_range_B // 2
        start_view = view_index - half_range
        end_view = view_index + half_range
        
        total_views = view_range_B
        total_data_size_mb = total_views * self.video_size_mb
        
        request_data = {
            'satellite_name': satellite_name,
            'center_view': view_index,
            'view_range': view_range_B,
            'start_view': start_view,
            'end_view': end_view,
            'user_id': self.user_id,
            'video_size_per_view_mb': self.video_size_mb,
            'total_data_size_mb': total_data_size_mb
        }
        
        return self.sat.handle_user_request(self.user_id, request_data)

# create a class to represent a satellite
class Satellite:
    def __init__(self, sat_name, sat_csv, storage_constraint_Z=100, total_views=360, view_size_mb=6):
        self.sat_name = sat_name
        self.lat, self.lon, self.alt = sat_csv['lat'], sat_csv['lon'], sat_csv['alt']
        self.x, self.y, self.z = sat_csv['x'], sat_csv['y'], sat_csv['z']
        # self.min_elevation_deg = 25
        self.serving_users = []
        self.storage_constraint_Z = storage_constraint_Z
        self.view_size_mb = view_size_mb
        self.cache_state = set()
        self.neighbor_caches = set()
        # self.tx_power_watt = 10.0 
        # self.antenna_gain_dbi = 45.0
        # self.isl_data_rate_gbps = 100.0
        # self.downlink_freq_ghz = 18.0
        self.tx_power_watt = 15.0          # Increased from 10W (Starlink v2 specs)
        self.antenna_gain_dbi = 48.0       # Improved phased arrays (up from 45 dBi)
        self.isl_data_rate_gbps = 100.0    # Optical ISL capacity
        self.downlink_freq_ghz = 14.0      # Ku-band (more common than 18 GHz)
        self.uplink_freq_ghz = 30.0        # Ka-band uplink
        self.altitude_km = 550             # Typical LEO altitude
        self.min_elevation_deg = 25
        
    def __str__(self):
        return f'Satellite {self.sat_name}, serving users {self.serving_users}, cached views: {len(self.cache_state)}'
    
    def __lt__(self, other):
        return self.sat_name < other.sat_name

    def is_view_cached(self, view_index):
        return view_index in self.cache_state

    def cache_view(self, view_index):
        if len(self.cache_state) < self.storage_constraint_Z:
            self.cache_state.add(view_index)
            return True
        return False

    def evict_view(self, view_index):
        if view_index in self.cache_state:
            self.cache_state.remove(view_index)
            return True
        return False
    
    def get_cache_utilization(self):
        return len(self.cache_state) / self.storage_constraint_Z if self.storage_constraint_Z > 0 else 0

    def get_storage_used_mb(self):
        # Assuming view_size_mb is consistent, needs to be passed in or set globally
        return len(self.cache_state) * self.view_size_mb # Using global video_size_per_view_mb

    def get_storage_capacity_mb(self):
        return self.storage_constraint_Z * self.view_size_mb
    
    def calculate_elevation_angle(self, time, user):
        """Calculate elevation angle from user to satellite"""
        sat_lat = self.lat[time]
        sat_lon = self.lon[time]
        sat_alt = self.alt[time]
        
        # Convert to radians
        sat_lat_rad = np.radians(sat_lat)
        sat_lon_rad = np.radians(sat_lon)
        user_lat_rad = np.radians(user.lat)
        user_lon_rad = np.radians(user.lon)
        
        # Calculate great circle distance
        delta_lon = sat_lon_rad - user_lon_rad
        cos_central_angle = (np.sin(user_lat_rad) * np.sin(sat_lat_rad) + 
                            np.cos(user_lat_rad) * np.cos(sat_lat_rad) * np.cos(delta_lon))
        cos_central_angle = np.clip(cos_central_angle, -1.0, 1.0)
        central_angle = np.arccos(cos_central_angle)
        
        # Calculate elevation angle
        R = EARTH_RADIUS
        h = sat_alt
        cos_elevation = (np.sin(central_angle) * R) / np.sqrt(R**2 + h**2 + 2*R*h*np.cos(central_angle))
        cos_elevation = np.clip(cos_elevation, 0.0, 1.0)
        elevation_angle = np.pi/2 - np.arccos(cos_elevation)
        
        return np.degrees(elevation_angle)

    def connect_user(self, time, user):
        """
        Checks if a user is within the satellite's coverage area (A_n(t)) based on the
        model from the research report.
        """
        # Convert degrees to radians for calculations
        theta_min_rad = np.radians(self.min_elevation_deg)
        h = self.alt[time]
        cos_psi = (EARTH_RADIUS / (EARTH_RADIUS + h)) * np.cos(theta_min_rad)
        if cos_psi > 1.0: 
            cos_psi = 1.0
        psi = np.arccos(cos_psi) - theta_min_rad

        # Get satellite and user positions in radians
        lat_sat_rad = np.radians(self.lat[time])
        lon_sat_rad = np.radians(self.lon[time])
        lat_user_rad = np.radians(user.lat)
        lon_user_rad = np.radians(user.lon)
        
        # Calculate the central angle between satellite sub-point and user
        # Using the spherical law of cosines
        delta_lon = lon_user_rad - lon_sat_rad
        cos_central_angle = (np.sin(lat_sat_rad) * np.sin(lat_user_rad) + np.cos(lat_sat_rad) * np.cos(lat_user_rad) * np.cos(delta_lon))
        if cos_central_angle > 1.0: 
            cos_central_angle = 1.0
        if cos_central_angle < -1.0: 
            cos_central_angle = -1.0
        central_angle = np.arccos(cos_central_angle)
        
        # The user is covered if their central angle is within the coverage cone
        return central_angle <= psi

    def distance_to_user(self, time, user):
        """
        Calculates the direct 3D Euclidean distance to a user.
        """
        sx, sy, sz = self.x[time], self.y[time], self.z[time] # Get satellite's Cartesian coordinates at the given time
        ux, uy, uz = user.x, user.y, user.z  # Get user's Cartesian coordinates
        
        distance = np.sqrt((sx - ux)**2 + (sy - uy)**2 + (sz - uz)**2) # Calculate Euclidean distance
        return distance
    
    def distance_to_satellite(self, other_sat, time):
        """
        Calculates the direct 3D Euclidean distance to another satellite.
        """
        # Ensure the other satellite has data for the current timeslot
        if time >= len(other_sat.x):
            return -1 # Return invalid distance

        # Get coordinates for self and the other satellite
        sx1, sy1, sz1 = self.x[time], self.y[time], self.z[time]
        sx2, sy2, sz2 = other_sat.x[time], other_sat.y[time], other_sat.z[time]

        # Calculate Euclidean distance
        distance = np.sqrt((sx1 - sx2)**2 + (sy1 - sy2)**2 + (sz1 - sz2)**2)
        return distance

    def handle_user_request(self, user_id, request_data):
        # This method is not used by the new DP logic but kept for compatibility
        return True, "Request received by DP model"


# GroundStation class remains unchanged...
class GroundStation:
    def __init__(self, station_id, name, lat, lon, total_views=360, view_size_mb=6):
        self.station_id, self.name, self.lat, self.lon = station_id, name, float(lat), float(lon)
        self.total_views, self.view_size_mb = int(total_views), view_size_mb
        self.available_views = set(range(self.total_views))
        self.min_elevation_deg = 10.0
        # self.tx_power_watt = 50.0
        # self.antenna_gain_dbi = 30.0
        # self.bandwidth_hz = 250 * 1e6
        # self.uplink_freq_ghz = 30.0

        self.tx_power_watt = 100.0         # Higher power for better uplink
        self.antenna_gain_dbi = 65.0       # Large parabolic antennas (up from 30 dBi)
        self.bandwidth_hz = 500 * 1e6      # 500 MHz uplink bandwidth
        self.uplink_freq_ghz = 30.0        # Ka-band uplink
        self.antenna_diameter_m = 11.3     # Large ground station antenna

        # Track requests and transmissions
        self.transmission_history = []
        self.connected_satellites = []
        
    def __str__(self):
        total_size_gb = (len(self.available_views) * self.view_size_mb) / 1024
        return f'Ground Station {self.station_id} ({self.name}) at ({self.lat}, {self.lon}) with {len(self.available_views)} views ({total_size_gb:.1f}GB)'
    
    # def get_uplink_rate_to_sat(self, sat, time):
    #     distance = self.calculate_distance_to_satellite(sat.lat[time], sat.lon[time], sat.alt[time])
    #     return calculate_rate_mbps(
    #         tx_power_watt=self.tx_power_watt, tx_gain_dbi=self.antenna_gain_dbi,
    #         rx_gain_dbi=sat.antenna_gain_dbi, distance_km=distance,
    #         bandwidth_hz=self.bandwidth_hz, frequency_ghz=self.uplink_freq_ghz
    #     )
    
    def get_uplink_rate_to_sat(self, sat, time):
        """Calculate uplink rate with elevation angle consideration"""
        distance = self.calculate_distance_to_satellite(sat.lat[time], sat.lon[time], sat.alt[time])
        
        # Calculate elevation angle for atmospheric effects
        elevation_angle = self.calculate_elevation_angle_to_sat(sat, time)
        
        return calculate_rate_mbps(
            tx_power_watt=self.tx_power_watt,
            tx_gain_dbi=self.antenna_gain_dbi,
            rx_gain_dbi=sat.antenna_gain_dbi,
            distance_km=distance,
            bandwidth_hz=self.bandwidth_hz,
            frequency_ghz=self.uplink_freq_ghz,
            elevation_angle_deg=elevation_angle
        )
    
    def calculate_elevation_angle_to_sat(self, sat, time):
        """Calculate elevation angle from ground station to satellite"""
        sat_lat = sat.lat[time]
        sat_lon = sat.lon[time]
        sat_alt = sat.alt[time]
        
        # Convert to radians
        sat_lat_rad = np.radians(sat_lat)
        sat_lon_rad = np.radians(sat_lon)
        gs_lat_rad = np.radians(self.lat)
        gs_lon_rad = np.radians(self.lon)
        
        # Calculate great circle distance
        delta_lon = sat_lon_rad - gs_lon_rad
        cos_central_angle = (np.sin(gs_lat_rad) * np.sin(sat_lat_rad) + 
                            np.cos(gs_lat_rad) * np.cos(sat_lat_rad) * np.cos(delta_lon))
        cos_central_angle = np.clip(cos_central_angle, -1.0, 1.0)
        central_angle = np.arccos(cos_central_angle)
        
        # Calculate elevation angle
        R = EARTH_RADIUS
        h = sat_alt
        cos_elevation = (np.sin(central_angle) * R) / np.sqrt(R**2 + h**2 + 2*R*h*np.cos(central_angle))
        cos_elevation = np.clip(cos_elevation, 0.0, 1.0)
        elevation_angle = np.pi/2 - np.arccos(cos_elevation)
        
        return np.degrees(elevation_angle)

    def has_view(self, view_index):
        """Check if ground station has a specific view (always True)"""
        return view_index in self.available_views
    
    def get_views_range(self, start_view, end_view):
        """Get a range of views from the ground station"""
        requested_views = list(range(start_view, end_view + 1))
        available_requested = [v for v in requested_views if v in self.available_views]
        return available_requested
    
    def transmit_to_satellite(self, satellite, requested_views):
        """Transmit requested views to a satellite"""
        if not isinstance(requested_views, list):
            requested_views = [requested_views]
        
        available_views = [v for v in requested_views if v in self.available_views]
        transmitted_data_size_mb = len(available_views) * self.view_size_mb
        
        # Record transmission
        transmission_record = {
            'satellite': satellite.sat_name,
            'requested_views': requested_views,
            'transmitted_views': available_views,
            'transmitted_data_size_mb': transmitted_data_size_mb,
            'timestamp': len(self.transmission_history)
        }
        self.transmission_history.append(transmission_record)
        
        # print(f"Ground Station {self.station_id} transmitting {len(available_views)} views ({transmitted_data_size_mb:.1f}MB) to satellite {satellite.sat_name}")
        
        return available_views
    
    def connect_satellite(self, satellite):
        """Establish connection with a satellite"""
        if satellite.sat_name not in self.connected_satellites:
            self.connected_satellites.append(satellite.sat_name)
    
    def disconnect_satellite(self, satellite):
        """Disconnect from a satellite"""
        if satellite.sat_name in self.connected_satellites:
            self.connected_satellites.remove(satellite.sat_name)
    
    def get_transmission_statistics(self):
        """Get statistics about transmissions from ground station"""
        total_transmissions = len(self.transmission_history)
        total_views_transmitted = sum(len(t['transmitted_views']) for t in self.transmission_history)
        total_data_transmitted_mb = sum(t['transmitted_data_size_mb'] for t in self.transmission_history)
        
        return {
            'station_id': self.station_id,
            'total_transmissions': total_transmissions,
            'total_views_transmitted': total_views_transmitted,
            'total_data_transmitted_mb': total_data_transmitted_mb,
            'total_data_transmitted_gb': total_data_transmitted_mb / 1024,
            'connected_satellites': len(self.connected_satellites)
        }
    
    def is_satellite_in_view(self, sat, time):
        """
        Correctly checks if a satellite is within the ground station's service cover
        using the same geometric logic as the satellite-to-user connection.
        """
        if time >= len(sat.lat):
            return False

        theta_min_rad = np.radians(self.min_elevation_deg)
        h = sat.alt[time]  # Satellite altitude

        # Calculate psi, the half-cone angle of coverage from the GS perspective
        # This formula is from your report for satellite coverage
        cos_psi = (EARTH_RADIUS / (EARTH_RADIUS + h)) * np.cos(theta_min_rad)
        if cos_psi > 1.0: cos_psi = 1.0
        psi = np.arccos(cos_psi) - theta_min_rad
        
        # Get satellite and GS positions in radians
        lat_sat_rad = np.radians(sat.lat[time])
        lon_sat_rad = np.radians(sat.lon[time])
        lat_gs_rad = np.radians(self.lat)
        lon_gs_rad = np.radians(self.lon)

        # Calculate the central angle between the GS and the satellite's sub-point
        delta_lon = lon_sat_rad - lon_gs_rad
        cos_central_angle = (np.sin(lat_gs_rad) * np.sin(lat_sat_rad) +
                             np.cos(lat_gs_rad) * np.cos(lat_sat_rad) * np.cos(delta_lon))
        
        # Clamp value to avoid floating point errors
        if cos_central_angle > 1.0: cos_central_angle = 1.0
        if cos_central_angle < -1.0: cos_central_angle = -1.0
        central_angle = np.arccos(cos_central_angle)

        # The satellite is in view if its central angle is within the coverage cone
        return central_angle <= psi

    def calculate_distance_to_satellite(self, sat_lat, sat_lon, sat_alt):
        """Calculate distance between ground station and satellite"""
        # This method is still useful for finding the *closest* among visible stations
        gs_lat_rad, gs_lon_rad = np.radians(float(self.lat)), np.radians(float(self.lon))
        sat_lat_rad, sat_lon_rad = np.radians(float(sat_lat)), np.radians(float(sat_lon))
        
        dlat, dlon = sat_lat_rad - gs_lat_rad, sat_lon_rad - gs_lon_rad

        a = np.sin(dlat/2)**2 + np.cos(gs_lat_rad) * np.cos(sat_lat_rad) * np.sin(dlon/2)**2
        angular_distance = 2 * np.arcsin(np.sqrt(a))
        
        surface_distance = EARTH_RADIUS * angular_distance
        return np.sqrt(surface_distance**2 + float(sat_alt)**2)
    
    def calculate_transmission_cost(self, distance, data_size_mb):
        """Calculate transmission cost based on distance and data size"""
        # Simple cost model: higher cost for longer distances and larger data
        base_cost = 10
        distance_factor = distance / 1000  # Convert to km and scale
        data_factor = data_size_mb / 100  # Scale data size
        return base_cost + distance_factor + data_factor
    

class LEOCommParams:
    # Satellite Parameters (based on Starlink v2, Telesat Lightspeed, OneWeb specifications)
    SATELLITE_ALTITUDE_KM = 550  # km (typical LEO altitude)
    SATELLITE_TX_POWER_WATT = 15.0  # W (increased from 10W based on recent specs)
    SATELLITE_ANTENNA_GAIN_DBI = 48.0  # dBi (improved phased array antennas)
    
    # User Terminal Parameters (based on modern flat-panel terminals)
    USER_ANTENNA_GAIN_DBI = 12.0  # dBi (improved from 4 dBi - modern phased arrays)
    USER_ANTENNA_DIAMETER_M = 0.3  # m (typical flat-panel terminal)
    
    # Ground Station Parameters (based on recent Gateway specifications)
    GS_TX_POWER_WATT = 100.0  # W (increased for better uplink)
    GS_ANTENNA_GAIN_DBI = 65.0  # dBi (large parabolic antennas, 11.3m diameter)
    GS_ANTENNA_DIAMETER_M = 11.3  # m
    
    # ISL Parameters (based on recent optical/RF ISL specifications)
    ISL_TX_POWER_WATT = 5.0  # W
    ISL_ANTENNA_GAIN_DBI = 35.0  # dBi
    ISL_DATA_RATE_GBPS = 100.0  # Gbps (optical ISL)
    
    # Frequency and Bandwidth Parameters (based on recent allocations)
    # Ku-band downlink (12-18 GHz) - widely used
    DOWNLINK_FREQ_GHZ = 14.0  # GHz (Ku-band)
    DOWNLINK_BANDWIDTH_MHZ = 250  # MHz (wider bandwidth for higher rates)
    
    # Ka-band uplink (27-40 GHz) - for higher capacity
    UPLINK_FREQ_GHZ = 30.0  # GHz (Ka-band)
    UPLINK_BANDWIDTH_MHZ = 500  # MHz
    
    # Noise Parameters
    NOISE_TEMP_K = 290  # K (system noise temperature)
    NOISE_FIGURE_DB = 3.0  # dB (typical for modern receivers)
