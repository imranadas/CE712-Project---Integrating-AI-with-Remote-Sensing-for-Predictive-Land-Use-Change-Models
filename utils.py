import os
import pickle
import numpy as np
from datetime import datetime

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
    EBBI = (swir_1 - nir) / (10 * np.sqrt((swir_1 + nir) + epsilon)) * (swir_2 + epsilon)
    
    NDVI = np.clip(NDVI, -1, 1)
    NDBI = np.clip(NDBI, -1, 1)
    NDWI = np.clip(NDWI, -1, 1)
    
    return NDVI, NDBI, NDWI, EBBI

def classify_land_use(NDVI, NDBI, NDWI, EBBI):
    """
    Classify land use into 5 categories based on spectral indices.
    
    Parameters:
    -----------
    NDVI: numpy array
        Normalized Difference Vegetation Index
    NDBI: numpy array
        Normalized Difference Built-up Index
    NDWI: numpy array
        Normalized Difference Water Index
    EBBI: numpy array
        Enhanced Built-Up and Bareness Index
    
    Returns:
    --------
    numpy array
        Classification array with values:
        0: Barren Land
        1: Dense Vegetation
        2: Moderate Vegetation
        3: Urban Areas
        4: Water Bodies
    """
    # Initialize output array
    classification = np.zeros_like(NDVI, dtype=np.uint8)
    
    # Define masks for each land use class
    masks = {
        # Water bodies: High NDWI, low NDVI
        'water': (NDWI > 0.3) & (NDVI < 0.2),
        
        # Dense vegetation: High NDVI, negative NDBI
        'dense_veg': (NDVI > 0.5) & (NDBI < 0) & (NDWI < 0.3),
        
        # Moderate vegetation: Medium NDVI, low EBBI
        'mod_veg': (NDVI >= 0.25) & (NDVI <= 0.5) & (EBBI < 0) & (NDWI < 0.3),
        
        # Urban areas: High NDBI or high EBBI with low vegetation
        'urban': ((NDBI > 0) & (NDVI < 0.25)) | 
                ((EBBI > 0) & (NDVI < 0.25) & (NDWI < 0.3)),
        
        # Barren land: Low NDVI, moderate EBBI, not water
        'barren': (NDVI < 0.25) & (EBBI > -0.25) & (EBBI < 0.25) & (NDWI < 0.3)
    }
    
    # Apply masks in order of priority
    # Water bodies take precedence
    classification = np.where(masks['water'], 4, classification)
    
    # Dense vegetation in non-water areas
    classification = np.where(masks['dense_veg'] & ~masks['water'], 
                            1, classification)
    
    # Moderate vegetation in areas not already classified
    classification = np.where(masks['mod_veg'] & 
                            ~masks['water'] & 
                            ~masks['dense_veg'], 
                            2, classification)
    
    # Urban areas in remaining areas
    classification = np.where(masks['urban'] & 
                            ~masks['water'] & 
                            ~masks['dense_veg'] & 
                            ~masks['mod_veg'], 
                            3, classification)
    
    # Barren land in any remaining unclassified areas
    classification = np.where((classification == 0) & masks['barren'], 
                            0, classification)
    
    return classification

def process_landsat_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    
    date_str = os.path.basename(file_path).split('.')[0]
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        date = datetime.strptime(date_str, '%Y%m%d')
    
    SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7, DEM, air_temp, precipitation = data[:10]
    
    # Calculate indices
    indices = calculate_spectral_indices(SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7)
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
        'air_temp': air_temp,
        'precipitation': precipitation
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