'''
Standardize

Fills nan values with the mean and returns a standardized raster with the 
equation xds - mean(xds) / std(xds)
'''

import rioxarray
import glob
import os

if __name__ == "__main__":
    raster_pattern = "Uganda Covariate Rasters/uganda_elevation_2km.tif"
    output_directory = 'Uganda Standardized Rasters/'
    
    file_paths = glob.glob(raster_pattern)
    
    for file_path in file_paths:
        xds = rioxarray.open_rasterio(file_path)
        filled = xds.fillna(xds.mean())
        standardized = (filled - filled.mean()) / filled.std()
        output_path = f"{output_directory}standard_{os.path.basename(file_path)}"
        standardized.rio.to_raster(output_path)
