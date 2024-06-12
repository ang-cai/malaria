import rioxarray
import xarray
import glob
import os

if __name__ == "__main__":
    raster_pattern = 'reenvironmentalcovariatesgrid/uganda_*.tif'
    output_directory = 'StandardizedUganda/'
    
    file_paths = glob.glob(raster_pattern)
    
    for file_path in file_paths:
        xds = rioxarray.open_rasterio(file_path)
        filled = xds.fillna(xds.mean())
        standardized = (filled - filled.mean()) / filled.std()
        output_path = os.path.join(output_directory, "standard_" + os.path.basename(file_path))
        standardized.rio.to_raster(output_path)
