import numpy as np
import os

def calculate_indices(geotiff_data):
    # Extract relevant bands (assuming shape is [height, width, bands])
    B3 = geotiff_data[:, :, 2]  # Green (SR_B3)
    B4 = geotiff_data[:, :, 3]  # Red (SR_B4)
    B5 = geotiff_data[:, :, 4]  # NIR (SR_B5)
    B6 = geotiff_data[:, :, 5]  # SWIR (SR_B6)

    # Calculate NDVI: (NIR - Red) / (NIR + Red)
    ndvi = (B5 - B4) / (B5 + B4 + 1e-10)  # Adding small value to avoid division by zero

    # Calculate NDWI: (Green - NIR) / (Green + NIR)
    ndwi = (B3 - B5) / (B3 + B5 + 1e-10)

    # Calculate Built-up Index: (SWIR - NIR) / (SWIR + NIR)
    built_up = (B6 - B5) / (B6 + B5 + 1e-10)

    return ndvi, ndwi, built_up

def label_land_use(ndvi, ndwi, built_up):
    # Initialize a label map with zeros (2D array)
    labels = np.zeros(ndvi.shape, dtype=np.uint8)  # Using uint8 for memory efficiency

    # Label water bodies (NDWI > 0.3)
    water_mask = ndwi > 0.3
    labels[water_mask] = 1  # Label water as 1

    # Label vegetation (NDVI > 0.3)
    vegetation_mask = ndvi > 0.3
    labels[vegetation_mask] = 2  # Label vegetation as 2

    # Label urban areas (Built-up > 0.1)
    urban_mask = built_up > 0.1
    labels[urban_mask] = 3  # Label urban areas as 3

    return labels

def automate_labeling(data_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Loop through each .npy file in the directory
    for file_name in sorted(os.listdir(data_dir)):
        if file_name.endswith('.npy'):
            print(f"Processing {file_name}...")

            # Load geotiff data from .npy file
            geotiff_data = np.load(os.path.join(data_dir, file_name))

            # Check if the data is 3D (height, width, bands)
            if geotiff_data.ndim != 3:
                print(f"Skipping {file_name}: Expected 3D array, got {geotiff_data.ndim}D.")
                continue

            # Calculate indices
            ndvi, ndwi, built_up = calculate_indices(geotiff_data)

            # Generate land-use labels
            labels = label_land_use(ndvi, ndwi, built_up)

            # Save the labels as .npy
            save_path = os.path.join(save_dir, f"labels_{file_name}")
            np.save(save_path, labels)
            print(f"Labels saved to {save_path}")

# Paths
data_dir = 'DataSet\Data'  # Directory containing the .npy geotiff files
save_dir = 'DataSet\Labels'  # Directory to save the label files

# Run labeling process
automate_labeling(data_dir, save_dir)
