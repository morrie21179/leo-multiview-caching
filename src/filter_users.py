import pandas as pd

def filter_users_for_specific_regions(input_file='data/users.csv', output_file='data/users_filtered.csv'):
    """
    Reads a user CSV file and filters it to keep only users in the specific regions
    highlighted in the provided image (North America, Europe, East Asia, Oceania).

    Args:
        input_file (str): The path to the original user data file.
        output_file (str): The path to save the new, filtered user data.
    """
    try:
        # Read the original user data
        df = pd.read_csv(input_file)
        original_count = len(df)
        print(f"Original number of users: {original_count}")

        # Define specific bounding boxes based on the highlighted image
        # Format: (min_lat, max_lat, min_lon, max_lon)
        regions = {
            'NorthAmerica': (25, 55, -125, -70),
            'Europe': (40, 60, -10, 30),
            'EastAsia': (22, 45, 105, 145),
            'Oceania': (-45, -12, 113, 178) # Australia and New Zealand
        }

        print("\nFiltering to keep users only in these regions:")
        for region, (min_lat, max_lat, min_lon, max_lon) in regions.items():
            print(f"- {region}: Lat({min_lat} to {max_lat}), Lon({min_lon} to {max_lon})")

        # Create a boolean mask for each region
        conditions = []
        for region, (min_lat, max_lat, min_lon, max_lon) in regions.items():
            condition = (
                (df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
                (df['lon'] >= min_lon) & (df['lon'] <= max_lon)
            )
            conditions.append(condition)

        # Combine all conditions with a logical OR
        # A user is kept if they fall into ANY of the defined regions
        final_condition = pd.concat(conditions, axis=1).any(axis=1)

        # Apply the filter to the DataFrame
        filtered_df = df[final_condition]
        filtered_count = len(filtered_df)

        # Save the new filtered data, overwriting the old file if it exists
        filtered_df.to_csv(output_file, index=False)
        
        print(f"\nFiltered number of users: {filtered_count}")
        print(f"Removed {original_count - filtered_count} users.")
        print(f"New user file has been saved as '{output_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please make sure it's in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the filtering function
if __name__ == '__main__':
    filter_users_for_specific_regions()
