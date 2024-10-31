import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

# Dictionary to convert full state names to abbreviations
STATE_MAPPING = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
    'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms', 'missouri': 'mo',
    'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh', 'new jersey': 'nj',
    'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh',
    'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
    'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt',
    'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy',
    'district of columbia': 'dc'
}

def load_distance_matrix(matrix_file):
    """Load the pre-calculated distance matrix and city mappings"""
    with open(matrix_file, 'rb') as f:
        data = pickle.load(f)
    return data['matrix'], data['city_mapping']

def standardize_city_key(city, state):
    """Standardize city and state format to match the mapping"""
    if type(city) != str or type(state) != str:
        return ""
    city = city.strip().lower()
    state = state.strip().lower()
    
    # Convert full state name to abbreviation
    state_abbrev = STATE_MAPPING.get(state, state)
    
    return f"{city}, {state_abbrev}"

def get_city_distance(city1, state1, city2, state2, distance_matrix, city_mapping):
    """Look up the distance between two cities"""
    # Format city keys to match the stored format
    city1_key = standardize_city_key(city1, state1)
    city2_key = standardize_city_key(city2, state2)
    
    try:
        # Get matrix indices
        idx1 = city_mapping[city1_key]
        idx2 = city_mapping[city2_key]
        
        # Return distance
        return distance_matrix[idx1, idx2]
    except KeyError as e:
        # For debugging, print which city wasn't found
        if city1_key not in city_mapping:
            print(f"City not found in mapping: {city1_key}")
        if city2_key not in city_mapping:
            print(f"City not found in mapping: {city2_key}")
        return np.nan

def process_excel_file(input_file, output_file, matrix_file):
    """Process the Excel file and add distances"""
    # Load the distance matrix and mappings
    print("Loading distance matrix...")
    distance_matrix, city_mapping = load_distance_matrix(matrix_file)
    
    # Print some sample mappings for verification
    print("\nSample city mappings:")
    sample_cities = list(city_mapping.items())[:5]
    for city, idx in sample_cities:
        print(f"{city} => {idx}")
    
    # Read Excel file
    print("\nReading Excel file...")
    df = pd.read_excel(input_file)
    
    # Create a new column for distances
    print("Calculating distances...")
    distances = []
    not_found = 0
    not_found_pairs = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        distance = get_city_distance(
            row['w_city'],
            row['w_state'],
            row['t_city'],
            row['t_state'],
            distance_matrix,
            city_mapping
        )
        
        if np.isnan(distance):
            not_found += 1
            not_found_pairs.append((
                standardize_city_key(row['w_city'], row['w_state']),
                standardize_city_key(row['t_city'], row['t_state'])
            ))
        
        distances.append(distance)
    
    # Add distances to dataframe
    df['Distance in miles'] = distances
    
    # Print statistics
    total_rows = len(df)
    found_rows = total_rows - not_found
    print(f"\nProcessing complete:")
    print(f"Total pairs processed: {total_rows:,}")
    print(f"Distances found: {found_rows:,} ({found_rows/total_rows*100:.1f}%)")
    print(f"Distances not found: {not_found:,} ({not_found/total_rows*100:.1f}%)")
    
    # Print some examples of city pairs that weren't found
    if not_found_pairs:
        print("\nSample of city pairs not found (first 5):")
        for pair in not_found_pairs[:5]:
            print(f"{pair[0]} <-> {pair[1]}")
    
    # Save results
    print(f"\nSaving results to {output_file}")
    df.to_excel(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    # Replace these with your actual file names
    INPUT_FILE = "inputFile.xlsx"  # Your input Excel file
    OUTPUT_FILE = "city_pairs_with_distances.xlsx"  # Where to save the results
    MATRIX_FILE = "city_distances.pkl"  # Your saved distance matrix
    
    process_excel_file(INPUT_FILE, OUTPUT_FILE, MATRIX_FILE)

