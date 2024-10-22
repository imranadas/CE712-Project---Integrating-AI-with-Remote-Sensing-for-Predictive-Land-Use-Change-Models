import os
import numpy as np
import rasterio

def get_highest_precision_dtype(dtypes):
    # Order of precision: float64 > float32 > int64 > uint64 > int32 > uint32 > int16 > uint16 > int8 > uint8
    dtype_order = [np.float64, np.float32, np.int64, np.uint64, np.int32, np.uint32, np.int16, np.uint16, np.int8, np.uint8]
    for dtype in dtype_order:
        if any(np.issubdtype(t, dtype) for t in dtypes):
            return dtype
    return np.float64  # Default to float64 if no match found

def convert_geotiff_to_npy(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.npy')

            with rasterio.open(input_path) as src:
                # Read all bands
                data = src.read()
                
                # Get the highest precision dtype among all bands
                band_dtypes = [band.dtype for band in data]
                highest_precision_dtype = get_highest_precision_dtype(band_dtypes)

                # Convert all bands to the highest precision dtype
                data = data.astype(highest_precision_dtype)

                # Save as NumPy array
                np.save(output_path, data)

            print(f"Converted {filename} to {os.path.basename(output_path)} with dtype {highest_precision_dtype}")

# Usage
input_folder = 'Dataset_RAW'
output_folder = 'DataSet'
convert_geotiff_to_npy(input_folder, output_folder)