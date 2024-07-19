'''
Standardize

Fills nan values with the mean and returns a standardized raster with the 
equation xds - mean(xds) / std(xds)
'''

import rioxarray
import pandas as pd
import glob
import os

if __name__ == "__main__":
    raster_pattern = "East Africa Covariate Data/*"
    output_directory = 'East Africa Standardized Data/'
    
    file_paths = glob.glob(raster_pattern)
    
    for file_path in file_paths:
        # for tif
        # xds = rioxarray.open_rasterio(file_path)
        # filled = xds.fillna(xds.mean())
        # standardized = (filled - filled.mean()) / filled.std()
        # output_path = f"{output_directory}standard_{os.path.basename(file_path)}"
        # standardized.rio.to_raster(output_path)

        # for csv
        data = pd.read_csv(file_path)
        means = data.iloc[:, 6:].mean()
        filled = data.iloc[:, 6:].fillna(means)
        standardized = filled.apply(lambda x: (x - x.mean()) / x.std())
        data.iloc[:, 6:] = standardized
        print(data)
        
