# Land Use Change Prediction System

A comprehensive system for processing satellite imagery data, analyzing land use patterns, and predicting future land use changes using machine learning.

## Overview

This system processes Landsat 8 satellite imagery along with environmental data (elevation, temperature, precipitation) to analyze and predict land use changes over time. It includes data collection from Google Earth Engine, processing pipelines, visualization tools, and a machine learning prediction model.

## Components

1. **Data Collection** (`GEE_Cloud.py`)
   - Fetches Landsat 8 imagery from Google Earth Engine
   - Collects SRTM elevation data
   - Retrieves ERA5 climate data
   - Handles cloud masking and data interpolation

2. **Data Processing**
   - `GeoTif_NPY.py`: Converts GeoTIFF files to NumPy arrays
   - `utils.py`: Processes satellite data and calculates spectral indices
   - Implements land use classification

3. **Visualization** (`views.py`)
   - Interactive visualization tools for satellite bands
   - Spectral indices visualization
   - Land use classification viewer
   - Time series analysis tools

4. **Prediction Model** (`model.py`)
   - Machine learning model for land use prediction
   - Temporal and spatial feature engineering
   - Future state prediction capabilities

5. **Analysis Tools**
   - Time series analysis
   - Land use change detection
   - Prediction confidence visualization

## Requirements

### System Requirements
- Python 3.10+
- 16GB RAM minimum (32GB recommended)
- GPU recommended for faster model training
- Storage space: Minimum 50GB for data storage

### Installation

1. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up Google Earth Engine authentication:
```bash
earthengine authenticate
```

## Usage

### 1. Data Collection

```python
python GEE_Cloud.py
```
This will:
- Connect to Google Earth Engine
- Download Landsat 8 imagery
- Collect climate and elevation data
- Save combined data to Google Cloud Storage

### 2. Data Processing

```python
python GeoTif_NPY.py
```
Converts downloaded GeoTIFF files to NumPy arrays for faster processing.

### 3. Running Analysis

Use the Jupyter notebook `code.ipynb` to:
- Process the collected data
- Visualize results
- Train the prediction model
- Generate future predictions

## File Structure

```
├── GEE_Cloud.py           # Google Earth Engine data collection
├── GeoTif_NPY.py         # GeoTIFF to NumPy converter
├── code.ipynb            # Main analysis notebook
├── model.py             # Prediction model implementation
├── utils.py             # Utility functions and data processing
├── views.py             # Visualization components
├── DataSet/            # Processed NumPy data
├── Dataset_RAW/        # Raw GeoTIFF files
└── Processed_DataSet/  # Final processed data
```

## Methodology

### Data Processing Pipeline
1. Cloud masking and interpolation of Landsat data
2. Calculation of spectral indices (NDVI, NDBI, NDWI, EBBI)
3. Integration with climate and elevation data
4. Land use classification using rule-based system

### Land Use Classification
Classifies areas into:
- Barren Land
- Dense Vegetation
- Moderate Vegetation
- Urban Areas
- Water Bodies

### Prediction Model
Uses Random Forest with:
- Spatial smoothing
- Temporal window analysis
- Feature engineering from multiple data sources
- Confidence score calculation

## Notes

- Ensure sufficient storage space for satellite imagery
- GPU acceleration recommended for model training
- Regular backup of processed data recommended

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request