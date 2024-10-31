import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import time
from datetime import datetime
import sys

class CityDistanceCalculator:
    def __init__(self, county_distances_file=None, county_zip_file=None):
        self.distances = None
        self.city_to_index = None
        self.city_to_county = None
        self.county_distances = None
        self.start_time = time.time()
        
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        self.log_file = f"logs/distance_calculator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        if county_distances_file and county_zip_file:
            self.load_data(county_distances_file, county_zip_file)
    
    def log(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"{timestamp}: {message}"
        print(full_message)
        with open(self.log_file, 'a') as f:
            f.write(full_message + '\n')
    
    def load_data(self, county_distances_file, county_zip_file):
        try:
            self.log("Loading county distances...")
            self.county_distances = pd.read_csv(county_distances_file)
            
            # Format county codes to 5 digits
            self.county_distances['county1'] = self.county_distances['county1'].astype(str).str.zfill(5)
            self.county_distances['county2'] = self.county_distances['county2'].astype(str).str.zfill(5)
            
            self.log(f"Loaded {len(self.county_distances):,} county distance records")
            
            self.log("\nLoading and processing COUNTY_ZIP data...")
            county_zip_df = pd.read_csv(county_zip_file)
            county_zip_df['COUNTY'] = county_zip_df['COUNTY'].astype(str).str.zfill(5)
            
            # Standardize city and state names
            county_zip_df['USPS_ZIP_PREF_CITY'] = county_zip_df['USPS_ZIP_PREF_CITY'].str.strip().str.title()
            county_zip_df['USPS_ZIP_PREF_STATE'] = county_zip_df['USPS_ZIP_PREF_STATE'].str.strip().str.upper()
            
            # Create city-county mapping
            city_county_groups = county_zip_df.groupby(['USPS_ZIP_PREF_CITY', 'USPS_ZIP_PREF_STATE'])['COUNTY'].agg(list).reset_index()
            
            self.city_to_county = {}
            for _, row in city_county_groups.iterrows():
                city_key = f"{row['USPS_ZIP_PREF_CITY'].lower()}, {row['USPS_ZIP_PREF_STATE'].lower()}"
                most_common_county = max(set(row['COUNTY']), key=row['COUNTY'].count)
                self.city_to_county[city_key] = most_common_county
            
            self.city_to_index = {city: idx for idx, city in enumerate(self.city_to_county.keys())}
            
            # Initialize distance matrix
            n = len(self.city_to_index)
            self.distances = np.full((n, n), np.nan)
            
            self.log(f"\nFound {n:,} unique cities")
            self._populate_distance_matrix()
            
        except Exception as e:
            self.log(f"ERROR during data loading: {str(e)}")
            raise
    
    def save_checkpoint(self, i):
        """Save a checkpoint of the current progress"""
        checkpoint_file = f"checkpoints/matrix_checkpoint_{i}.pkl"
        Path('checkpoints').mkdir(exist_ok=True)
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'matrix': self.distances,
                'city_mapping': self.city_to_index,
                'last_processed_index': i
            }, f)
        self.log(f"Saved checkpoint at city {i:,}")
    
    def _populate_distance_matrix(self):
        """Populate the distance matrix using county distances"""
        try:
            self.log("Creating county distance lookup...")
            county_distance_lookup = {}
            for _, row in self.county_distances.iterrows():
                county_distance_lookup[(row['county1'], row['county2'])] = row['mi_to_county']
                county_distance_lookup[(row['county2'], row['county1'])] = row['mi_to_county']
            
            self.log(f"Created lookup for {len(county_distance_lookup):,} county pairs")
            
            # Populate matrix
            cities = list(self.city_to_index.keys())
            pairs_with_distance = 0
            total_pairs = 0
            last_checkpoint_time = time.time()
            
            for i, city1 in enumerate(tqdm(cities, desc="Processing cities")):
                county1 = self.city_to_county[city1]
                
                for j in range(i, len(cities)):
                    city2 = cities[j]
                    county2 = self.city_to_county[city2]
                    total_pairs += 1
                    
                    if i == j:
                        self.distances[i, j] = 0
                        pairs_with_distance += 1
                    else:
                        distance = county_distance_lookup.get((county1, county2))
                        if distance is not None:
                            self.distances[i, j] = distance
                            self.distances[j, i] = distance
                            pairs_with_distance += 1
                
                # Progress update and checkpoint every hour
                current_time = time.time()
                if current_time - last_checkpoint_time >= 3600:  # 1 hour
                    elapsed_hours = (current_time - self.start_time) / 3600
                    progress_percent = (i + 1) / len(cities) * 100
                    
                    self.log(f"\nProgress update at {datetime.now()}:")
                    self.log(f"Processed {i + 1:,} of {len(cities):,} cities ({progress_percent:.1f}%)")
                    self.log(f"Found distances for {pairs_with_distance:,} of {total_pairs:,} pairs")
                    self.log(f"Elapsed time: {elapsed_hours:.1f} hours")
                    
                    # Save checkpoint
                    self.save_checkpoint(i)
                    last_checkpoint_time = current_time
            
            self.log("\nMatrix population completed successfully!")
            
        except Exception as e:
            self.log(f"ERROR during matrix population: {str(e)}")
            # Save emergency checkpoint
            self.save_checkpoint(-1)
            raise
    
    def save_matrix(self, output_file):
        """Save the final matrix and mappings"""
        try:
            with open(output_file, 'wb') as f:
                pickle.dump({
                    'matrix': self.distances,
                    'city_mapping': self.city_to_index
                }, f)
            self.log(f"Matrix saved to {output_file}")
        except Exception as e:
            self.log(f"ERROR saving matrix: {str(e)}")
            raise
    
    def get_city_coverage(self):
        """Get statistics about city coverage"""
        total_cities = len(self.city_to_index)
        total_counties = len(set(self.city_to_county.values()))
        valid_distances = np.count_nonzero(~np.isnan(np.triu(self.distances, k=1)))
        
        return {
            'total_cities': total_cities,
            'total_counties': total_counties,
            'total_city_pairs': valid_distances,
            'matrix_size': self.distances.shape,
        }

if __name__ == "__main__":
    try:
        print(f"Starting process at {datetime.now()}")
        
        calculator = CityDistanceCalculator(
            county_distances_file="sf2010.csv",
            county_zip_file="COUNTY_ZIP.csv"
        )
        
        # Print coverage statistics
        coverage = calculator.get_city_coverage()
        calculator.log("\nCoverage Statistics:")
        calculator.log(f"Total unique cities: {coverage['total_cities']:,}")
        calculator.log(f"Total counties: {coverage['total_counties']:,}")
        calculator.log(f"Total city pairs with distances: {coverage['total_city_pairs']:,}")
        calculator.log(f"Matrix size: {coverage['matrix_size']}")
        
        # Save the final matrix
        calculator.save_matrix("city_distances.pkl")
        
        elapsed_time = (time.time() - calculator.start_time) / 3600
        calculator.log(f"\nProcess completed successfully in {elapsed_time:.1f} hours")
        
    except Exception as e:
        print(f"ERROR: Process failed with error: {str(e)}")
        sys.exit(1)