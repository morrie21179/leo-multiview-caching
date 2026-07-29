import os
import random
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

from container_0617 import generate_zipf_distribution, calculate_rate_mbps, SPEED_OF_LIGHT_KM_S


video_size_per_view_mb = 60

class ESMMatching:
    """Exchange-Stable Matching for LEO Satellite Caching"""
    
    def __init__(self, satellites, ground_stations, users, num_videos, views_per_video, zipf_alpha):
        self.satellites = satellites
        self.ground_stations = ground_stations
        self.users = users
        self.num_videos = num_videos
        self.views_per_video = views_per_video
        self.zipf_alpha = zipf_alpha
        self.total_views = num_videos * views_per_video
        # Pre-compute global popularity distribution
        self.global_popularity = generate_zipf_distribution(self.total_views, zipf_alpha)
        
    def calculate_content_access_delay(self, user_id, view_id, satellite, timeslot):
        """Calculate Content Access Delay (CAD) for a specific user-content pair"""
        user = next((u for u in self.users if u.user_id == user_id), None)
        if not user or not satellite:
            return float('inf')
            
        # Find the shortest path to access the content
        min_delay = float('inf')
        
        # Case 1: Content is cached locally in the serving satellite
        if view_id in satellite.cache_state:
            distance = satellite.distance_to_user(timeslot, user)
            propagation_delay = distance / SPEED_OF_LIGHT_KM_S
            min_delay = min(min_delay, propagation_delay)
        
        # Case 2: Content is in neighbor satellites (ISL)
        for neighbor_sat in self.satellites.values():
            if neighbor_sat != satellite and view_id in neighbor_sat.cache_state:
                # Calculate ISL delay + downlink delay
                isl_distance = satellite.distance_to_satellite(neighbor_sat, timeslot)
                if isl_distance > 0:
                    isl_delay = isl_distance / SPEED_OF_LIGHT_KM_S
                    downlink_distance = satellite.distance_to_user(timeslot, user)
                    downlink_delay = downlink_distance / SPEED_OF_LIGHT_KM_S
                    total_delay = isl_delay + downlink_delay
                    min_delay = min(min_delay, total_delay)
        
        # Case 3: Content from ground station
        for gs in self.ground_stations:
            if gs.has_view(view_id) and gs.is_satellite_in_view(satellite, timeslot):
                gs_to_sat_distance = gs.calculate_distance_to_satellite(
                    satellite.lat.iloc[timeslot], satellite.lon.iloc[timeslot], satellite.alt.iloc[timeslot])
                gs_delay = gs_to_sat_distance / SPEED_OF_LIGHT_KM_S
                downlink_distance = satellite.distance_to_user(timeslot, user)
                downlink_delay = downlink_distance / SPEED_OF_LIGHT_KM_S
                total_delay = gs_delay + downlink_delay
                min_delay = min(min_delay, total_delay)
        
        return min_delay if min_delay != float('inf') else 1000.0  # High penalty for unavailable content
    
    def calculate_satellite_cost(self, satellite, content_set, timeslot):
        """Calculate the cost for a satellite caching a specific set of contents - SIMPLIFIED"""
        total_cost = 0.0
        
        if not satellite.serving_users:
            return 0.0
        
        # Sample only top popular content (top 1% or 100 items max)
        num_samples = min(100, max(1, self.total_views // 100))
        top_popular_indices = np.argsort(self.global_popularity)[-num_samples:]
        
        for user_id in satellite.serving_users:
            user = next((u for u in self.users if u.user_id == user_id), None)
            if not user:
                continue
                
            # Only check the most popular content
            for view_id in top_popular_indices:
                request_prob = self.global_popularity[view_id]
                cad = self.calculate_content_access_delay(user_id, view_id, satellite, timeslot)
                total_cost += request_prob * cad * video_size_per_view_mb
                
        return total_cost
    
    def calculate_content_cost(self, view_id, satellite_set, timeslot):
        """Calculate the cost for a content being cached in a specific set of satellites"""
        total_cost = 0.0
        
        # For all users in the system
        for user in self.users:
            if user.sat is None:
                continue
                
            # Check if the user's serving satellite is in the satellite set
            if user.sat.sat_name in [sat.sat_name for sat in satellite_set]:
                request_prob = self.global_popularity[view_id] if view_id < len(self.global_popularity) else 0
                
                # Find the best serving satellite from the set
                min_cad = float('inf')
                for sat in satellite_set:
                    if sat.sat_name == user.sat.sat_name:
                        cad = self.calculate_content_access_delay(user.user_id, view_id, sat, timeslot)
                        min_cad = min(min_cad, cad)
                
                content_size_weight = video_size_per_view_mb
                total_cost += request_prob * min_cad * content_size_weight
        
        return total_cost
    
    def find_swap_blocking_pairs(self, matching, timeslot):
        """Find swap-blocking pairs for exchange-stable matching - SIMPLIFIED"""
        blocking_pairs = []
        
        satellite_list = list(self.satellites.values())
        
        # MAJOR SIMPLIFICATION: Only check first few satellites to prevent infinite loops
        # max_sats_to_check = min(5, len(satellite_list))
        max_sats_to_check = len(satellite_list)
        satellite_list = satellite_list[:max_sats_to_check]
        
        # Check only adjacent pairs for swaps
        for i in range(len(satellite_list)):
            for j in range(i + 1, min(i + 3, len(satellite_list))):  # Only check 2 neighbors
                sat1, sat2 = satellite_list[i], satellite_list[j]
                
                # cache1 = list(sat1.cache_state)[:5]  # Only check first 5 contents
                # cache2 = list(sat2.cache_state)[:5]  # Only check first 5 contents
                cache1 = list(sat1.cache_state)  # check all contents
                cache2 = list(sat2.cache_state)  # check all contents

                
                # Try swapping contents between satellites (limited)
                for content1 in cache1:
                    for content2 in cache2:
                        if content1 != content2:
                            if self.is_beneficial_swap(sat1, sat2, content1, content2, timeslot):
                                blocking_pairs.append((sat1, sat2, content1, content2))
                                if len(blocking_pairs) >= 5:  # Limit number of swaps (10%)
                                    return blocking_pairs
        
        return blocking_pairs
    
    def is_beneficial_swap(self, sat1, sat2, content1, content2, timeslot):
        """Check if swapping content1 from sat1 with content2 from sat2 is beneficial - SIMPLIFIED"""
        # Simple heuristic: swap if content2 is more popular than content1
        if content1 >= len(self.global_popularity) or content2 >= len(self.global_popularity):
            return False
            
        pop1 = self.global_popularity[content1]
        pop2 = self.global_popularity[content2]
        
        # Simple rule: swap if we're getting more popular content
        return pop2 > pop1 * 1.1  # 10% improvement threshold
    
    def is_beneficial_swap_with_hole(self, sat, old_content, new_content, timeslot):
        """Check if swapping old_content with new_content (from hole) is beneficial"""
        current_cost = self.calculate_satellite_cost(sat, sat.cache_state, timeslot)
        
        # Create temporary cache state
        temp_cache = (sat.cache_state - {old_content}) | {new_content}
        
        # Check capacity constraint
        if len(temp_cache) > sat.storage_constraint_Z:
            return False
        
        # Calculate cost after swap
        sat_temp_cache = sat.cache_state.copy()
        sat.cache_state = temp_cache
        
        new_cost = self.calculate_satellite_cost(sat, temp_cache, timeslot)
        
        # Restore original cache state
        sat.cache_state = sat_temp_cache
        
        return new_cost < current_cost
    
    def perform_swap(self, sat1, sat2, content1, content2):
        """Perform the actual swap operation"""
        if sat2 is None:  # Swap with hole
            sat1.cache_state.discard(content1)
            sat1.cache_state.add(content2)
        else:  # Swap between two satellites
            sat1.cache_state.discard(content1)
            sat1.cache_state.add(content2)
            sat2.cache_state.discard(content2)
            sat2.cache_state.add(content1)
    
    def exchange_stable_matching(self, timeslot, max_iterations=20):
        """Main ESM algorithm"""
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            
            # Find swap-blocking pairs
            blocking_pairs = self.find_swap_blocking_pairs(None, timeslot)
            
            if not blocking_pairs:
                print(f"ESM converged after {iterations} iterations")
                break
            
            # Perform the first beneficial swap found
            sat1, sat2, content1, content2 = blocking_pairs[0]
            self.perform_swap(sat1, sat2, content1, content2)
            
        #     print(f"Iteration {iterations}: Performed swap between {sat1.sat_name} and {sat2.sat_name if sat2 else 'hole'}")
        
        # if iterations >= max_iterations:
        #     print(f"ESM reached maximum iterations ({max_iterations})")
        
        return iterations
