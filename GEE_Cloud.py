import ee

# Initialize the Earth Engine library
print("Initializing Earth Engine...")
ee.Initialize()

# Define the region of interest (ROI)
roi = ee.Geometry.Polygon([
    [75.58679011123377, 26.725580815770467],
    [76.00152399795252, 26.725580815770467],
    [76.00152399795252, 27.08929110630265],
    [75.58679011123377, 27.08929110630265],
    [75.58679011123377, 26.725580815770467]
])

# Time range and interval
start_date = '2013-03-18'
end_date = '2020-07-09'
interval = 16  # Landsat is captured every 16 days

# Function to mask clouds in Landsat 8 images
def mask_l8_sr(image):
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(mask)

# Function to interpolate missing data for Landsat bands
def interpolate_landsat_bands(image):
    landsat_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    interpolated = image.select(landsat_bands).focal_mean(2, 'square', 'pixels', 1)
    return image.select(landsat_bands).unmask(interpolated).addBands(
        image.select(image.bandNames().removeAll(landsat_bands)))

# Load and preprocess Landsat 8 collection
print("Loading and preprocessing Landsat 8 collection...")
landsat_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                      .filterBounds(roi)
                      .filterDate(start_date, end_date)
                      .map(mask_l8_sr)
                      .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
                      .map(interpolate_landsat_bands))

# Load SRTM DEM data
print("Loading SRTM DEM data...")
srtm = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(roi)

# Load and preprocess ERA5 daily data
print("Loading ERA5 climate data...")
era5 = (ee.ImageCollection('ECMWF/ERA5/DAILY')
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .select(['mean_2m_air_temperature', 'total_precipitation'])
        .map(lambda image: image
             .reproject(crs='EPSG:4326', scale=1000)
             .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=30000)
             .reproject(crs='EPSG:4326', scale=30)
             .clip(roi)))

# Function to combine Landsat, SRTM, and ERA5 data
def combine_data(image):
    date = ee.Date(image.get('system:time_start'))
    era5_image = era5.filterDate(date, date.advance(1, 'day')).first()
    return image.addBands(srtm).addBands(era5_image).toFloat()

# Combine the data
print("Combining Landsat, SRTM, and ERA5 data...")
combined_collection = landsat_collection.map(combine_data)

# Function to filter by the 16-day interval and ensure all bands are present
def filter_collection(collection, start_date, end_date, interval, required_bands):
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

# Filter the collection
print("Filtering collection...")
filtered_collection = filter_collection(combined_collection, start_date, end_date, interval, required_bands)

# Function to export images to Google Cloud Storage (GCS)
def export_image(image, index, bucket_name):
    date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    print(f"Starting export for image {index+1} with date: {date}")

    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=f'Combined_Data_{date}',
        fileNamePrefix=f'Combined_Data_{date}_{index}',
        bucket=bucket_name,
        region=roi,
        scale=30,
        crs='EPSG:4326',
        maxPixels=1e13,
        formatOptions={'cloudOptimized': True}
    )
    task.start()
    return task

# GCS bucket name where the images will be saved
bucket_name = 'ce712-geo_spatail_temporal_data'  # Replace with your actual GCS bucket name

# Execute exports for each image in the filtered collection
filtered_list = filtered_collection.toList(filtered_collection.size())
total_images = filtered_collection.size().getInfo()
print(f"Total images to export: {total_images}")

for index in range(total_images):
    image = ee.Image(filtered_list.get(index))
    export_image(image, index, bucket_name)
    print(f'Exporting image {index + 1} of {total_images}')

print("All exports initiated.")