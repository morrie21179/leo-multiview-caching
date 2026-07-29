# container_RFP.py

import numpy as np
import pandas as pd

EARTH_RADIUS = 6371 # in km
BOLTZMANN_K = 1.38e-23
NOISE_TEMP_K = 290
SPEED_OF_LIGHT_KM_S = 299792.458 # Speed of light in km/s (more accurate)
SPEED_OF_LIGHT = 299792458  # m/s

NUM_VIDEOS = 2500
VIEWS_PER_VIDEO = 16

# Updated LEO Communication Parameters Class
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


def generate_zipf_distribution(N, alpha):
    """Generates a Zipf probability distribution for N items."""
    if N <= 0: return []
    x = np.arange(1, N + 1)
    weights = x ** (-alpha)
    return weights / np.sum(weights) if np.sum(weights) > 0 else []


class User:
    def __init__(self, user_id, lat, lon, x, y, z, video_size_mb=6):
        self.user_id, self.lat, self.lon, self.x, self.y, self.z = user_id, lat, lon, x, y, z
        self.video_size_mb = video_size_mb
        self.elevation = 0
        self.sat = None
        
        # Updated parameters for modern user terminals
        self.antenna_gain_dbi = 12.0       # Modern flat-panel phased arrays (up from 4 dBi)
        self.bandwidth_hz = 250 * 1e6      # 250 MHz bandwidth (up from 20 MHz)
        self.terminal_type = "flat_panel"  # Modern terminal type

    def __str__(self):
        return f'User {self.user_id}'

    def __lt__(self, other):
        return self.user_id < other.user_id

    def generate_request(self, num_videos, views_per_video, zipf_alpha, view_range_B=3):
        if self.sat is None:
            return False, None
        video_pop_dist = generate_zipf_distribution(num_videos, zipf_alpha)
        angle_pop_dist = generate_zipf_distribution(views_per_video, zipf_alpha)
        video_id = np.random.choice(np.arange(num_videos), p=video_pop_dist)
        center_angle = np.random.choice(np.arange(views_per_video), p=angle_pop_dist)
        half_range = view_range_B // 2
        start_angle = max(0, center_angle - half_range)
        end_angle = min(views_per_video - 1, center_angle + half_range)
        h = video_id * views_per_video + start_angle
        l = video_id * views_per_video + end_angle
        request_data = {'h': h, 'l': l}
        return True, request_data


class Satellite:
    def __init__(self, sat_name, sat_csv, storage_constraint_Z=100, total_views=360, view_size_mb=60, region_id=None):
        self.sat_name = sat_name
        self.region_id = region_id
        self.lat, self.lon, self.alt = sat_csv['lat'], sat_csv['lon'], sat_csv['alt']
        self.x, self.y, self.z = sat_csv['x'], sat_csv['y'], sat_csv['z']
        self.min_elevation_deg = 25
        self.serving_users = []
        self.storage_constraint_Z = storage_constraint_Z
        self.view_size_mb = view_size_mb
        self.cache_state = set()
        self.neighbor_caches = set()
        
        # Updated parameters based on modern LEO satellites
        self.tx_power_watt = 15.0          # Increased from 10W (Starlink v2 specs)
        self.antenna_gain_dbi = 48.0       # Improved phased arrays (up from 45 dBi)
        self.isl_data_rate_gbps = 100.0    # Optical ISL capacity
        self.downlink_freq_ghz = 14.0      # Ku-band (more common than 18 GHz)
        self.uplink_freq_ghz = 30.0        # Ka-band uplink
        self.altitude_km = 550             # Typical LEO altitude
        
        # RFP specific attributes
        self.region_features = np.zeros(VIEWS_PER_VIDEO)  # D-dimensional vector
        self.request_history = {}  # To store request counts for feature prediction
        
        # Additional tracking for caching algorithms
        self.last_access_time = {}
        self.access_frequency = {}

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
        theta_min_rad = np.radians(self.min_elevation_deg)
        h = self.alt[time]
        cos_psi = (EARTH_RADIUS / (EARTH_RADIUS + h)) * np.cos(theta_min_rad)
        if cos_psi > 1.0: cos_psi = 1.0
        psi = np.arccos(cos_psi) - theta_min_rad
        lat_sat_rad = np.radians(self.lat[time])
        lon_sat_rad = np.radians(self.lon[time])
        lat_user_rad = np.radians(user.lat)
        lon_user_rad = np.radians(user.lon)
        delta_lon = lon_user_rad - lon_sat_rad
        cos_central_angle = (np.sin(lat_sat_rad) * np.sin(lat_user_rad) + np.cos(lat_sat_rad) * np.cos(lat_user_rad) * np.cos(delta_lon))
        if cos_central_angle > 1.0: cos_central_angle = 1.0
        if cos_central_angle < -1.0: cos_central_angle = -1.0
        central_angle = np.arccos(cos_central_angle)
        return central_angle <= psi

    def distance_to_user(self, time, user):
        sx, sy, sz = self.x[time], self.y[time], self.z[time]
        ux, uy, uz = user.x, user.y, user.z
        return np.sqrt((sx - ux)**2 + (sy - uy)**2 + (sz - uz)**2)

    def distance_to_satellite(self, other_sat, time):
        if time >= len(other_sat.x): return -1
        sx1, sy1, sz1 = self.x[time], self.y[time], self.z[time]
        sx2, sy2, sz2 = other_sat.x[time], other_sat.y[time], other_sat.z[time]
        return np.sqrt((sx1 - sx2)**2 + (sy1 - sy2)**2 + (sz1 - sz2)**2)


class GroundStation:
    def __init__(self, station_id, name, lat, lon, total_views=360, view_size_mb=60):
        self.station_id, self.name, self.lat, self.lon = station_id, name, float(lat), float(lon)
        self.total_views, self.view_size_mb = int(total_views), view_size_mb
        self.available_views = set(range(self.total_views))
        self.min_elevation_deg = 10.0
        
        # Updated parameters for modern ground stations
        self.tx_power_watt = 100.0         # Higher power for better uplink
        self.antenna_gain_dbi = 65.0       # Large parabolic antennas (up from 30 dBi)
        self.bandwidth_hz = 500 * 1e6      # 500 MHz uplink bandwidth
        self.uplink_freq_ghz = 30.0        # Ka-band uplink
        self.antenna_diameter_m = 11.3     # Large ground station antenna

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

    def is_satellite_in_view(self, sat, time):
        if time >= len(sat.lat): return False
        theta_min_rad = np.radians(self.min_elevation_deg)
        h = sat.alt[time]
        cos_psi = (EARTH_RADIUS / (EARTH_RADIUS + h)) * np.cos(theta_min_rad)
        if cos_psi > 1.0: cos_psi = 1.0
        psi = np.arccos(cos_psi) - theta_min_rad
        lat_sat_rad, lon_sat_rad = np.radians(sat.lat[time]), np.radians(sat.lon[time])
        lat_gs_rad, lon_gs_rad = np.radians(self.lat), np.radians(self.lon)
        delta_lon = lon_sat_rad - lon_gs_rad
        cos_central_angle = (np.sin(lat_gs_rad) * np.sin(lat_sat_rad) + np.cos(lat_gs_rad) * np.cos(lat_sat_rad) * np.cos(delta_lon))
        if cos_central_angle > 1.0: cos_central_angle = 1.0
        if cos_central_angle < -1.0: cos_central_angle = -1.0
        central_angle = np.arccos(cos_central_angle)
        return central_angle <= psi

    def calculate_distance_to_satellite(self, sat_lat, sat_lon, sat_alt):
        gs_lat_rad, gs_lon_rad = np.radians(float(self.lat)), np.radians(float(self.lon))
        sat_lat_rad, sat_lon_rad = np.radians(float(sat_lat)), np.radians(float(sat_lon))
        dlat, dlon = sat_lat_rad - gs_lat_rad, sat_lon_rad - gs_lon_rad
        a = np.sin(dlat/2)**2 + np.cos(gs_lat_rad) * np.cos(sat_lat_rad) * np.sin(dlon/2)**2
        angular_distance = 2 * np.arcsin(np.sqrt(a))
        surface_distance = EARTH_RADIUS * angular_distance
        return np.sqrt(surface_distance**2 + float(sat_alt)**2)
