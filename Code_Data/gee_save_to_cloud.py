import ee

# Initialize the Earth Engine library
print("Initializing Earth Engine...")
ee.Initialize()

# Define the new region of interest (ROI) with all four corners explicitly set
roi = ee.Geometry.Polygon([
    [29.152500860315314, 31.774731399017288],  # Bottom-left corner
    [33.135044805627814, 31.774731399017288],  # Bottom-right corner
    [33.135044805627814, 29.38202520194163],   # Top-right corner
    [29.152500860315314, 29.38202520194163],   # Top-left corner
    [29.152500860315314, 31.774731399017288]   # Close the polygon by repeating the first point
])
print("Region of interest defined:", roi.getInfo())

# Time range and interval
start_date = '2013-03-18'
end_date = '2020-07-09'
interval = 16  # Landsat is captured every 16 days
print(f"Time range set from {start_date} to {end_date} with a {interval}-day interval.")

# Load and preprocess Landsat 8 collection without cloud masking
print("Loading Landsat 8 collection without cloud masking...")
landsat_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                      .filterBounds(roi)
                      .filterDate(start_date, end_date)
                      .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']))

# Load SRTM DEM data
print("Loading SRTM DEM data...")
srtm = ee.Image('USGS/SRTMGL1_003').clip(roi).select('elevation')

# Load and preprocess ERA5 daily data
print("Loading ERA5 climate data...")
era5 = (ee.ImageCollection('ECMWF/ERA5/DAILY')
         .filterBounds(roi)
         .filterDate(start_date, end_date)
         .select(['mean_2m_air_temperature', 'total_precipitation'])
         .map(lambda image: image.reproject(crs='EPSG:4326', scale=1000)
              .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=30000)
              .reproject(crs='EPSG:4326', scale=30)
              .clip(roi)))

# Function to combine Landsat, SRTM, and ERA5 data
def combine_data(image):
    date = ee.Date(image.get('system:time_start'))
    era5_image = era5.filterDate(date, date.advance(1, 'day')).first()
    
    # Cast Landsat bands to Float32
    landsat_float = image.cast({'SR_B1': 'float', 'SR_B2': 'float', 'SR_B3': 'float', 
                                'SR_B4': 'float', 'SR_B5': 'float', 'SR_B6': 'float', 'SR_B7': 'float'})
    
    # Cast SRTM to Float32
    srtm_float = srtm.cast({'elevation': 'float'})
    
    combined = landsat_float.addBands(srtm_float)
    
    if era5_image:
        # ERA5 data is already Float32, but we'll cast it explicitly for consistency
        era5_float = era5_image.cast({'mean_2m_air_temperature': 'float', 'total_precipitation': 'float'})
        combined = combined.addBands(era5_float)
    
    return combined

# Map the combine function over the Landsat collection
print("Combining Landsat, SRTM, and ERA5 data...")
combined_collection = landsat_collection.map(combine_data)

# Function to filter by the 16-day interval and ensure all bands are present
def filter_by_interval_and_bands(collection, start_date, end_date, interval, required_bands):
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    
    def get_image_for_date(date):
        date = ee.Date(date)
        image = collection.filterDate(date, date.advance(interval, 'day')).first()
        return ee.Algorithms.If(
            ee.Algorithms.IsEqual(image, None),
            None,
            ee.Algorithms.If(
                image.bandNames().containsAll(required_bands),
                image.set('system:time_start', date.millis()),
                None
            )
        )
    
    dates = ee.List.sequence(start.millis(), end.millis(), interval * 24 * 60 * 60 * 1000)
    images = dates.map(get_image_for_date)
    return ee.ImageCollection(images.removeAll([None]))

# Define required bands
required_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'elevation', 'mean_2m_air_temperature', 'total_precipitation']

# Filter by 16-day interval and ensure all bands are present
print("Filtering by the 16-day interval and ensuring all bands are present...")
filtered_collection = filter_by_interval_and_bands(combined_collection, start_date, end_date, interval, required_bands)

# Function to export images to Google Cloud Storage (GCS)
def export_image(image, index, bucket_name):
    date = ee.Date(image.get('system:time_start'))
    date_str = date.format('YYYY-MM-dd').getInfo()
    print(f"Starting export for image {index+1} with date: {date_str}")

    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=f'Combined_Landsat_SRTM_ERA5_{date_str}',
        fileNamePrefix=f'Combined_Landsat_SRTM_ERA5_{date_str}_{index}',
        bucket=bucket_name,
        region=roi,
        scale=30,
        crs='EPSG:4326',
        maxPixels=1e13,
        formatOptions={
            'cloudOptimized': True
        }
    )
    task.start()
    return task

# GCS bucket name where the images will be saved
bucket_name = 'ce712-geo_spatail_temporal_data'  # Replace with your actual GCS bucket name

# Execute exports for each image in the filtered collection
filtered_list = filtered_collection.toList(filtered_collection.size())
total_images = filtered_collection.size().getInfo()  # Get total count
print(f"Total images to export: {total_images}")

for index in range(total_images):
    image = ee.Image(filtered_list.get(index))
    export_image(image, index, bucket_name)
    print(f'Exporting image {index + 1} of {total_images}')  # Progress indicator

print("All exports initiated.")
