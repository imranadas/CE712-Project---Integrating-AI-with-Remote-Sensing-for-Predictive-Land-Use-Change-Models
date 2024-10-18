import rasterio
import numpy as np
import os
import pandas as pd

# Load the CSV with filenames
csv_path = 'Data\RAW\Jaipur_Filenames.csv'  # Update with actual path
csv_data = pd.read_csv(csv_path)

# Global variable to store the minimum dimensions
min_height, min_width = np.inf, np.inf

# Function to find the minimum dimensions across all GeoTIFF files
def find_minimum_dimensions(input_dir, csv_data):
    global min_height, min_width
    for idx, row in csv_data.iterrows():
        file_name = row['File Name']
        file_path = os.path.join(input_dir, file_name)
        
        with rasterio.open(file_path) as src:
            height, width = src.height, src.width
            min_height = min(min_height, height)
            min_width = min(min_width, width)
            print(f"File {file_name}: height={height}, width={width}")
    
    print(f"Minimum dimensions found: height={min_height}, width={min_width}")

# Function to crop the data to the minimum dimensions
def crop_to_minimum_dimensions(data, min_height, min_width):
    cropped_data = data[:, :min_height, :min_width]  # Crop all bands to the smallest dimensions
    return cropped_data

# Function to replace NaN values with the mean of the respective band
def replace_nan_with_mean(data):
    for i in range(data.shape[0]):
        band = data[i, :, :]
        mean_value = np.nanmean(band)  # Calculate the mean of the band, ignoring NaNs
        band[np.isnan(band)] = mean_value  # Replace NaN values with the mean
        data[i, :, :] = band  # Update the data with the filled values
    return data

# Function to normalize data between 0 and 1
def normalize_band(band):
    min_val = np.nanmin(band)
    max_val = np.nanmax(band)
    return (band - min_val) / (max_val - min_val)

# Function to preprocess each GeoTIFF file
def preprocess_geotiff(file_path):
    with rasterio.open(file_path) as src:
        data = src.read().astype(np.float32)  # Read all bands as float32 for consistency
        profile = src.profile
    
    # Crop to the minimum dimensions
    data = crop_to_minimum_dimensions(data, min_height, min_width)
    
    # Replace NaN values with the mean for each band
    data = replace_nan_with_mean(data)

    # Normalize only the first 7 Landsat bands
    for i in range(7):
        data[i, :, :] = normalize_band(data[i, :, :])

    # The last 3 bands (DEM, ERA5 climate) remain unchanged
    return data

# Directory containing the GeoTIFF files
input_dir = 'Data\RAW\Jaipur'
output_dir = 'Data\Pre-Processed\Jaipur'
os.makedirs(output_dir, exist_ok=True)

# Step 1: Find the minimum dimensions across all files
find_minimum_dimensions(input_dir, csv_data)

# Step 2: Process each file and crop to minimum dimensions
for idx, row in csv_data.iterrows():
    file_name = row['File Name']
    date_part = file_name.split('_')[4]  # Extract the date part (YYYY-MM-DD)
    
    file_path = os.path.join(input_dir, file_name)
    print(f"Processing {file_path}...")
    
    # Preprocess the file
    processed_data = preprocess_geotiff(file_path)
    
    # Save the processed data as .npy file, using the date for the filename
    output_file = os.path.join(output_dir, f"{date_part}.npy")
    np.save(output_file, processed_data)
    
    print(f"Saved processed data to {output_file}")

print("All files processed successfully.")