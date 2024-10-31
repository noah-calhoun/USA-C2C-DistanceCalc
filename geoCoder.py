import json
import sqlite3
from geopy.distance import geodesic
from geopy.geocoders import Nominatim  # Added missing import
import time  # Added missing import
from tqdm import tqdm
import pandas as pd


# This is un-used, as you will run into a rate limit 
# and its a bit abusive of this API for our purposeses

class DistanceCalculator:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="my_distance_calculator")
        self.cache_conn = sqlite3.connect('geocode_cache.db')
        self.setup_cache()
        
    def setup_cache(self):
        """Create cache table if it doesn't exist"""
        cursor = self.cache_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS geocode_cache
            (location TEXT PRIMARY KEY, coordinates TEXT)
        ''')
        self.cache_conn.commit()
        
    def get_cached_coordinates(self, location):
        """Get coordinates from cache"""
        cursor = self.cache_conn.cursor()
        cursor.execute('SELECT coordinates FROM geocode_cache WHERE location = ?', (location,))
        result = cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None
    
    def cache_coordinates(self, location, coordinates):
        """Store coordinates in cache"""
        cursor = self.cache_conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO geocode_cache (location, coordinates) VALUES (?, ?)',
            (location, json.dumps(coordinates))
        )
        self.cache_conn.commit()
    
    def get_coordinates(self, city, state):
        """Get coordinates for a city, state pair"""
        location = f"{city}, {state}, USA"
        
        # Check cache first
        cached_coords = self.get_cached_coordinates(location)
        if cached_coords:
            return cached_coords
        
        try:
            # If not in cache, geocode and store
            loc = self.geolocator.geocode(location)
            if loc:
                coords = (loc.latitude, loc.longitude)
                self.cache_coordinates(location, coords)
                time.sleep(1)  # Respect API limits
                return coords
        except Exception as e:
            print(f"Error geocoding {location}: {e}")
        return None
    
    def calculate_distances(self, input_file, output_file):
        """
        Calculate distances between US cities from an Excel file
        
        Excel columns:
        - Column B: First city name
        - Column C: First state name
        - Column D: Second city name
        - Column E: Second state name
        - Column J: Distance in miles (output)
        """
        print("Reading input file...")
        # Read specific columns using their Excel column letters
        df = pd.read_excel(input_file, usecols=['w_city', 'w_state', 't_city', 't_state'])
        
        # Rename columns to work with existing code
        df.columns = ['w_city', 'w_state', 't_city', 't_state']
        
        distances = []
        print("Calculating distances...")
        
        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Get coordinates for both cities
                coords1 = self.get_coordinates(row['w_city'], row['w_state'])
                coords2 = self.get_coordinates(row['t_city'], row['t_state'])
                
                # Calculate distance
                if coords1 and coords2:
                    # Convert kilometers to miles
                    distance = geodesic(coords1, coords2).miles
                else:
                    distance = None
                    
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                distance = None
                
            distances.append(distance)
        
        # Create new dataframe with original data and distances in a new column
        result_df = pd.read_excel(input_file)  # Read original file again to preserve all columns
        result_df = result_df.head(10)  # Limit to the first 10 rows
        result_df['distance_miles'] = distances  # Add new column for distances
        
        # Save results
        print("Saving results...")
        result_df.to_excel(output_file, index=False)
        print("Processing complete!")
        
        # Print summary
        successful = sum(1 for d in distances if d is not None)
        failed = sum(1 for d in distances if d is None)
        print(f"\nProcessing Summary:")
        print(f"Successfully calculated: {successful} distances")
        print(f"Failed to calculate: {failed} distances")

# Example usage
if __name__ == "__main__":
    calculator = DistanceCalculator()
    calculator.calculate_distances(
        input_file="inputFile.xlsx",
        output_file="output_distances.xlsx"
    )