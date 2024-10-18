import numpy as np
import os
import pandas as pd

# Directory containing the processed .npy files
npy_dir = 'Data\Pre-Processed\Jaipur'

# CSV file to store the statistics
output_csv_path = 'Data\Pre-Processed\Jaipur_npy_stats.csv'

# List to hold statistics for all files
statistics_data = []

# Function to calculate and return statistics for each band
def calculate_band_statistics(band_data):
    band_mean = np.nanmean(band_data)
    band_std = np.nanstd(band_data)
    band_min = np.nanmin(band_data)
    band_max = np.nanmax(band_data)
    return band_mean, band_std, band_min, band_max

# Process each .npy file in the directory
for file_name in os.listdir(npy_dir):
    if file_name.endswith('.npy'):
        file_path = os.path.join(npy_dir, file_name)
        print(f"Processing {file_path}...")
        
        # Load the .npy file
        data = np.load(file_path)
        
        # Extract file date (assuming filename format includes date, e.g., '2013-04-19.npy')
        date_part = file_name.split('.')[0]  # Get the date part (e.g., '2013-04-19')

        # Get the dimensions of the data (assuming shape is [bands, height, width])
        num_bands, height, width = data.shape
        
        # Initialize a dictionary to store stats for the current file
        file_stats = {'File Name': file_name, 'Date': date_part, 'Height': height, 'Width': width}
        
        # Loop through each band and calculate statistics
        for band_idx in range(num_bands):
            band_data = data[band_idx, :, :]
            band_mean, band_std, band_min, band_max = calculate_band_statistics(band_data)
            
            # Save the statistics for the current band
            file_stats[f'Band_{band_idx+1}_Mean'] = band_mean
            file_stats[f'Band_{band_idx+1}_Std'] = band_std
            file_stats[f'Band_{band_idx+1}_Min'] = band_min
            file_stats[f'Band_{band_idx+1}_Max'] = band_max
        
        # Append the stats to the list
        statistics_data.append(file_stats)

# Convert the list of dictionaries to a DataFrame
df_statistics = pd.DataFrame(statistics_data)

# Save the DataFrame to a CSV file
df_statistics.to_csv(output_csv_path, index=False)

print(f"Statistics saved to {output_csv_path}")
