import rasterio
import matplotlib.pyplot as plt

# Function to display each band of the GeoTIFF in a separate window
def display_geotiff_bands(geotiff_path):
    # Open the GeoTIFF file
    with rasterio.open(geotiff_path) as src:
        # Get the number of bands in the GeoTIFF
        num_bands = src.count
        
        # Loop through all bands
        for band in range(1, num_bands + 1):
            # Read the band data
            band_data = src.read(band)
            
            # Plot each band in a separate figure
            plt.figure()
            plt.title(f'Band {band}')
            plt.imshow(band_data, cmap='gray')
            plt.colorbar()
            plt.show()

# Path to your GeoTIFF file
geotiff_path = 'Data\RAW\Jaipur\Combined_Landsat_SRTM_ERA5_2013-04-19_2.tif'

# Call the function to display the bands
display_geotiff_bands(geotiff_path)
