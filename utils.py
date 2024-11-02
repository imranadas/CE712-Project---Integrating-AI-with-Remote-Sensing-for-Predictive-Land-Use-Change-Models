import os
import pickle
import numpy as np
from datetime import datetime

def read_metadata(mtl_file):
    """
    Read Landsat 8 metadata from MTL file.
    
    Args:
        mtl_file: Path to the MTL metadata file
        
    Returns:
        Dictionary containing relevant metadata parameters
    """
    metadata = {}
    
    try:
        with open(mtl_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('SOLAR_ELEVATION_ANGLE'):
                    metadata['SOLAR_ELEVATION_ANGLE'] = float(line.split('=')[1].strip())
                elif line.startswith('EARTH_SUN_DISTANCE'):
                    metadata['EARTH_SUN_DISTANCE'] = float(line.split('=')[1].strip())
                elif line.startswith('RADIANCE_MULT_BAND'):
                    band = line.split('_')[3]
                    metadata[f'ML_B{band}'] = float(line.split('=')[1].strip())
                elif line.startswith('RADIANCE_ADD_BAND'):
                    band = line.split('_')[3]
                    metadata[f'AL_B{band}'] = float(line.split('=')[1].strip())
    except Exception as e:
        print(f"Error reading metadata file: {str(e)}")
        return None
        
    return metadata

def calculate_spectral_reflectance(band_data, ml, al, solar_elevation_angle, earth_sun_distance=1):
    """
    Calculate surface reflectance from Landsat 8 DN values using radiometric coefficients
    and solar geometry.
    
    Args:
        band_data: Raw DN values from Landsat 8 band
        ml: Multiplicative scaling factor from metadata
        al: Additive scaling factor from metadata
        solar_elevation_angle: Sun elevation angle in degrees
        earth_sun_distance: Earth-Sun distance in astronomical units (default=1)
        
    Returns:
        Surface reflectance values
    """
    # Convert solar elevation angle to radians
    solar_zenith = (90 - solar_elevation_angle) * np.pi / 180
    
    # Calculate TOA Radiance
    l_lambda = ml * band_data.astype(float) + al
    
    # Calculate TOA Reflectance with correction for solar angle
    # π * L_λ * d^2 / (ESUN_λ * cos(θ_sz))
    # where ESUN_λ values are band-specific solar irradiance values
    # For Landsat 8, we can use the provided TOA reflectance coefficients
    # and just correct for solar angle
    p_lambda = l_lambda / np.cos(solar_zenith)
    
    # Clip values to valid range [0,1]
    return np.clip(p_lambda, 0, 1)

def process_landsat_spectral_data(file_path, metadata):
    """
    Process Landsat 8 data to surface reflectance before calculating indices.
    
    Args:
        file_path: Path to the Landsat data file
        metadata: Dictionary containing:
                 - ML and AL values for each band
                 - solar_elevation_angle
                 - earth_sun_distance (optional)
    
    Returns:
        List containing processed surface reflectance bands and other data
    """
    data = np.load(file_path, allow_pickle=True)
    
    # Extract solar parameters
    solar_elevation = metadata.get('SOLAR_ELEVATION_ANGLE', 45.0)  # Default 45 degrees
    earth_sun_dist = metadata.get('EARTH_SUN_DISTANCE', 1.0)  # Default 1 AU
    
    # Define coefficients for each band
    band_coeffs = {
        'B1': {'ml': metadata.get('ML_B1', 0.00002), 'al': metadata.get('AL_B1', -0.1)},
        'B2': {'ml': metadata.get('ML_B2', 0.00002), 'al': metadata.get('AL_B2', -0.1)},
        'B3': {'ml': metadata.get('ML_B3', 0.00002), 'al': metadata.get('AL_B3', -0.1)},
        'B4': {'ml': metadata.get('ML_B4', 0.00002), 'al': metadata.get('AL_B4', -0.1)},
        'B5': {'ml': metadata.get('ML_B5', 0.00002), 'al': metadata.get('AL_B5', -0.1)},
        'B6': {'ml': metadata.get('ML_B6', 0.00002), 'al': metadata.get('AL_B6', -0.1)},
        'B7': {'ml': metadata.get('ML_B7', 0.00002), 'al': metadata.get('AL_B7', -0.1)},
    }
    
    # Process each band to surface reflectance
    processed_bands = []
    for i, band in enumerate(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']):
        coeffs = band_coeffs[band]
        processed = calculate_spectral_reflectance(
            data[i],
            ml=coeffs['ml'],
            al=coeffs['al'],
            solar_elevation_angle=solar_elevation,
            earth_sun_distance=earth_sun_dist
        )
        processed_bands.append(processed)
    
    # Add remaining data (DEM, air_temp, precipitation)
    processed_bands.extend(data[7:])
    
    return processed_bands

def calculate_spectral_indices(band_1, band_2, band_3, band_4, band_5, band_6, band_7):
    epsilon = 1e-10
    
    ultrablue = band_1
    blue = band_2
    green = band_3
    red = band_4
    nir = band_5
    swir_1 = band_6
    swir_2 = band_7
    
    # NDVI calculation
    NDVI = np.divide(nir - red, nir + red + epsilon, where=nir + red > epsilon)
    
    # NDWI calculation
    NDWI = np.divide(green - nir, green + nir + epsilon, where= green + nir > epsilon)
    
    # NDBI calculation
    NDBI = np.divide(swir_1 - nir, swir_1 + nir + epsilon, where=swir_1 + nir > epsilon)
    
    # EBBI calculation
    EBBI = np.divide((swir_1 - nir), (10 * np.sqrt((swir_1 + swir_2) + epsilon)), where=(swir_1 + swir_2 > epsilon))
    
    NDVI = np.clip(NDVI, -1, 1)
    NDBI = np.clip(NDBI, -1, 1)
    NDWI = np.clip(NDWI, -1, 1)
    
    return NDVI, NDBI, NDWI, EBBI

def classify_land_use(NDVI, NDBI, NDWI, EBBI):
    """
    Classify land use into 5 categories:
    0: Barren Land
    1: Dense Vegetation
    2: Moderate Vegetation
    3: Urban Areas
    4: Water Bodies
    """
    # Initialize with zeros (barren land) and use uint8 for memory efficiency
    classification = np.zeros_like(NDVI, dtype=np.uint8)
    
    # Create boolean masks for each class
    # Note: Order of masks is important due to potential overlaps
    masks = {
        'water': (NDWI > 0.00) & (NDVI < 0.25),
        'dense_veg': (NDVI > 0.65) & (NDWI < 0.10) & (NDBI < -0.10),
        'mod_veg': (NDVI > 0.30) & (NDVI <= 0.65) & (NDWI < 0.10),
        'urban': ((NDBI > 0.00) & (NDVI < 0.20) & (EBBI > -0.10)) | 
                ((NDBI > 0.00) & (NDVI < 0.30))
    }
    
    # Everything starts as barren land (class 0)
    # Then apply masks in priority order
    classification = np.where(masks['water'], 4, classification)
    classification = np.where(masks['dense_veg'] & ~masks['water'], 1, classification)
    classification = np.where(masks['mod_veg'] & ~masks['water'] & ~masks['dense_veg'], 
                            2, classification)
    classification = np.where(masks['urban'] & ~masks['water'] & ~masks['dense_veg'] & 
                            ~masks['mod_veg'], 3, classification)
    
    return classification

def process_landsat_data(file_path):
    # Get metadata
    mtl_file = file_path.replace('.npy', '_MTL.txt')
    metadata = read_metadata(mtl_file)
    if metadata is None:
        print(f"Warning: Using default values for {file_path}")
        metadata = {}
    
    # Process to surface reflectance
    processed_data = process_landsat_spectral_data(file_path, metadata)
    
    date_str = os.path.basename(file_path).split('.')[0]
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        date = datetime.strptime(date_str, '%Y%m%d')
    
    # Calculate indices using surface reflectance values
    indices = calculate_spectral_indices(*processed_data[:7])
    NDVI, NDBI, NDWI, EBBI = indices
    
    # Calculate classification
    classification = classify_land_use(NDVI, NDBI, NDWI, EBBI)
    
    return {
        'date': date,
        'date_str': date_str,
        'classification': classification,
        'NDVI': NDVI,
        'NDBI': NDBI,
        'NDWI': NDWI,
        'EBBI': EBBI,
        'DEM': processed_data[7],
        'air_temp': processed_data[8],
        'precipitation': processed_data[9]
    }

def save_processed_data(processed_data_list, output_dir='processed_data'):
    """Save processed data to pickle files for faster loading"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual date files
    for data in processed_data_list:
        date_str = data['date_str']
        filename = f"{date_str}_processed.pkl"
        filepath = os.path.join(output_dir, filename)
        
        # Select only necessary data to save
        save_data = {
            'date': data['date'],
            'date_str': date_str,
            'classification': data['classification'],
            'NDVI': data['NDVI'],
            'NDBI': data['NDBI'],
            'NDWI': data['NDWI'],
            'EBBI': data['EBBI'],
            'DEM' : data['DEM'],
            'air_temp': data['air_temp'],
            'precipitation': data['precipitation']
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
            
def load_processed_data(processed_dir='processed_data'):
    """Load processed data from pickle files"""
    if not os.path.exists(processed_dir):
        return None
        
    processed_data_list = []
    for filename in sorted(os.listdir(processed_dir)):
        if filename.endswith('_processed.pkl'):
            filepath = os.path.join(processed_dir, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                processed_data_list.append(data)
    
    return sorted(processed_data_list, key=lambda x: x['date'])

def process_and_save_data(data_dir, processed_dir='processed_data'):
    """Process Landsat data and save if not already processed"""
    # Check if processed data exists
    if os.path.exists(processed_dir) and os.listdir(processed_dir):
        print("Loading pre-processed data...")
        return load_processed_data(processed_dir)
    
    print("Processing new data...")
    processed_data_list = []
    npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    
    for npy_file in npy_files:
        try:
            file_path = os.path.join(data_dir, npy_file)
            data = process_landsat_data(file_path)
            processed_data_list.append(data)
        except Exception as e:
            print(f"Error processing {npy_file}: {str(e)}")
            continue
    
    # Save processed data
    save_processed_data(processed_data_list, processed_dir)
    
    return sorted(processed_data_list, key=lambda x: x['date'])
