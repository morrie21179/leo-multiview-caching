import pandas as pd
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# Set matplotlib backend to avoid segmentation fault
import matplotlib
matplotlib.use('Agg')
plt.ioff()  # Turn off interactive plotting

def load_satellite_data():
    """Load satellite data using the same structure as main.py"""
    satellite_table = {}
    
    # Check if the data directory exists
    data_dir = 'data/starlink/satellite_trace'
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found")
        return None
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            try:
                satellite = pd.read_csv(f'{data_dir}/{file}')
                satellite_name = file.split('_')[0]
                
                # Ensure required columns exist
                required_cols = ['time', 'lat', 'lon', 'x', 'y', 'z', 'alt']
                if all(col in satellite.columns for col in required_cols):
                    satellite_table[satellite_name] = satellite
                    print(f"Loaded satellite {satellite_name} with {len(satellite)} records")
                else:
                    print(f"Missing required columns in {file}")
                    
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    if not satellite_table:
        print("No valid satellite data files found")
        return None
        
    return satellite_table

def create_satellite_animation(satellite_table, output_file='satellite_animation.gif', max_timesteps=100):
    """Create animation showing satellite positions over time"""
    
    if not satellite_table:
        print("No satellite data provided")
        return None
    
    # Set up the figure and axis
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.6)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.6)
    ax.set_global()
    
    # Get satellite names and determine time range
    satellite_names = list(satellite_table.keys())
    
    # Find common time range across all satellites
    min_time_steps = min(len(satellite_table[sat_name]) for sat_name in satellite_names)
    timesteps = min(min_time_steps, max_timesteps)
    
    print(f"Creating animation with {len(satellite_names)} satellites over {timesteps} timesteps")
    
    # Create color map for different satellites
    colors = plt.cm.tab20(np.linspace(0, 1, len(satellite_names)))
    sat_colors = dict(zip(satellite_names, colors))
    
    def animate(frame):
        ax.clear()
        
        # Re-add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.6)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.6)
        ax.set_global()
        
        # Plot current satellite positions
        for sat_name in satellite_names:
            sat_data = satellite_table[sat_name]
            
            if frame < len(sat_data):
                # Current position
                current_lat = sat_data['lat'].iloc[frame]
                current_lon = sat_data['lon'].iloc[frame]
                current_time = sat_data['time'].iloc[frame] if 'time' in sat_data.columns else frame
                
                # Plot current satellite position
                ax.plot(current_lon, current_lat, 'o', 
                       color=sat_colors[sat_name], 
                       markersize=8, 
                       transform=ccrs.PlateCarree(),
                       markeredgecolor='black',
                       markeredgewidth=0.5,
                       label=f'{sat_name}')
                
                # Add satellite trace (previous positions)
                if frame > 0:
                    trace_end = min(frame + 1, len(sat_data))
                    trace_start = max(0, frame - 10)  # Show last 10 positions
                    
                    trace_lats = sat_data['lat'].iloc[trace_start:trace_end].values
                    trace_lons = sat_data['lon'].iloc[trace_start:trace_end].values
                    
                    # Plot trace with fading effect
                    ax.plot(trace_lons, trace_lats, '-', 
                           color=sat_colors[sat_name], 
                           alpha=0.4, 
                           linewidth=2, 
                           transform=ccrs.PlateCarree())
        
        # Set title with current time/frame
        if 'time' in satellite_table[satellite_names[0]].columns:
            current_time = satellite_table[satellite_names[0]]['time'].iloc[frame]
            ax.set_title(f'Satellite Positions at Time {current_time}', fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Satellite Positions - Frame {frame}', fontsize=14, fontweight='bold')
        
        # Add legend
        if len(satellite_names) <= 10:  # Only show legend if not too many satellites
            ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                     fontsize=8, framealpha=0.8)
        
        # Add grid
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                    alpha=0.3, linewidth=0.5)
        
        return []
    
    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, animate, frames=timesteps, 
                        interval=300, blit=False, repeat=True)
    # Save as GIF
    print(f"Saving animation to {output_file}...")
    try:
        # Use imagemagick writer as fallback if pillow fails
        try:
            anim.save(output_file, writer='pillow', fps=3, dpi=100)
            print(f"Animation saved successfully as {output_file}")
        except Exception as pillow_error:
            print(f"Pillow writer failed: {pillow_error}")
            print("Trying imagemagick writer...")
            anim.save(output_file, writer='imagemagick', fps=3, dpi=100)
            print(f"Animation saved successfully as {output_file} using imagemagick")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Skipping animation save - creating static plot instead")
    
    # Close figure to prevent memory leaks and segmentation faults
    plt.close(fig)
    
    return anim

def create_satellite_trace_plot(satellite_table, output_file='satellite_traces.png'):
    """Create a static plot showing complete satellite traces"""
    
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.6)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.6)
    ax.set_global()
    
    satellite_names = list(satellite_table.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(satellite_names)))
    sat_colors = dict(zip(satellite_names, colors))
    
    for sat_name in satellite_names:
        sat_data = satellite_table[sat_name]
        
        # Plot complete trace
        ax.plot(sat_data['lon'], sat_data['lat'], '-', 
               color=sat_colors[sat_name], 
               alpha=0.7, 
               linewidth=1, 
               transform=ccrs.PlateCarree(),
               label=f'{sat_name}')
        
        # Mark start and end points
        ax.plot(sat_data['lon'].iloc[0], sat_data['lat'].iloc[0], 'o', 
               color=sat_colors[sat_name], markersize=6, 
               transform=ccrs.PlateCarree(), markeredgecolor='green', markeredgewidth=2)
        ax.plot(sat_data['lon'].iloc[-1], sat_data['lat'].iloc[-1], 's', 
               color=sat_colors[sat_name], markersize=6, 
               transform=ccrs.PlateCarree(), markeredgecolor='red', markeredgewidth=2)
    
    ax.set_title('Complete Satellite Traces\n(○ = Start, ■ = End)', fontsize=14, fontweight='bold')
    ax.set_title('Complete Satellite Traces\n(○ = Start, ■ = End)', fontsize=14, fontweight='bold')
    
    # Add legend if not too many satellites
    if len(satellite_names) <= 10:
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                 fontsize=8, framealpha=0.8)
    
    # Add grid
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                alpha=0.3, linewidth=0.5)
    
    # Save the plot
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Static trace plot saved as {output_file}")
    
    # Close figure to prevent memory leaks
    plt.close(fig)

def main():
    try:
        # Load satellite data using the same structure as main.py
        satellite_table = load_satellite_data()
        
        if satellite_table is not None:
            print(f"Loaded {len(satellite_table)} satellites")
            
            # Print summary of loaded data
            for sat_name, sat_data in satellite_table.items():
                print(f"  {sat_name}: {len(sat_data)} position records")
                if 'time' in sat_data.columns:
                    print(f"    Time range: {sat_data['time'].min()} to {sat_data['time'].max()}")
            
            # Create static trace plot first (safer)
            print("Creating static trace plot...")
            create_satellite_trace_plot(satellite_table)
            
            # Create animation with error handling
            print("Creating animation...")
            create_satellite_animation(satellite_table)
        else:
            print("Failed to load satellite data")
            print("Make sure the 'data/starlink/satellite_trace' directory exists and contains CSV files")
            print("Available files in current directory:", os.listdir('.'))
    except Exception as e:
        print(f"Unexpected error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure all matplotlib resources are cleaned up
        plt.close('all')

if __name__ == "__main__":
    main()