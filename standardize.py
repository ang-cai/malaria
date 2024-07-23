'''
Standardize

Fills nan values with the mean and returns a standardized raster with the 
equation xds - mean(xds) / std(xds)
'''

import rioxarray
import numpy as np
import geopandas as gpd
import pandas as pd
import glob
import os

if __name__ == "__main__":
    input_pattern = "East Africa Covariate Data/*.csv"
    gps_pattern = "East Africa Geodata/**/* Clusters/*.shp"
    output_directory = 'East Africa Standardized Data/'
    outdated = ["2010", "200", "19", "Malaria"]
    file_paths = sorted(glob.glob(input_pattern))
    gps_paths = sorted(glob.glob(gps_pattern))
    
    for file_path, gps_path in zip(file_paths, gps_paths):
        # for tif
        # xds = rioxarray.open_rasterio(file_path)
        # filled = xds.fillna(xds.mean())
        # standardized = (filled - filled.mean()) / filled.std()
        # output_path = f"{output_directory}standard_{os.path.basename(file_path)}"
        # standardized.rio.to_raster(output_path)

        # for csv
        data = pd.read_csv(file_path)
        coord = gpd.read_file(gps_path)
        gps_columns = 6

        # Uses -9999 and NA instead of nan
        nan = data.iloc[:, gps_columns:].replace(-9999, np.nan)
        nan = nan.iloc[:, gps_columns:].replace("NA", np.nan)
        means = nan.iloc[:, gps_columns:].mean()
        filled = nan.iloc[:, gps_columns:].fillna(means)
        standardized = filled.apply(lambda x: (x - x.mean()) / x.std())
        standardized = standardized.dropna(axis="columns")

        # Removing older than 2015 covariates
        for date in outdated:
            standardized = standardized[standardized.columns.drop(list(
                standardized.filter(regex=date)))]

        # Add GPS data except for nan (0, 0)
        standardized = gpd.GeoDataFrame(pd.concat(
            [data.iloc[:, :4], coord["LATNUM"], coord["LONGNUM"],
              coord["geometry"], standardized], axis=1))
        standardized = standardized.dropna(subset="DHSID")
        standardized = standardized[((standardized["LATNUM"] != 0) | 
                                     (standardized["LONGNUM"] != 0))]

        # Save
        basename = os.path.basename(file_path)
        components = basename.split("_")
        output_path = f"{output_directory}{components[0]}.shp"
        standardized.to_file(output_path)

