'''
Information matrix

Returns a graph of information gain for n random points of Malaria covariates 
using different lengthscales.
'''

import os
import glob
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from shapely import wkt

def _spatial_correlation(distances: list, std_err: int, spatial_std_err: int,
                         lengthscale: int):
    '''Creates a spatial correlation matrix. Formula: 
    Spatial correlation = std err^2 * I + spatial std err^2 * exp(-D^2/r^2/2)

    Returns numpy array.
    '''
    spatial_correlation = (
            np.identity(len(distances)) * std_err**2 + spatial_std_err**2 * 
            np.exp(-(np.square(distance_matrix(distances, distances))) / 
                   lengthscale**2 / 2)
    )
    return spatial_correlation

def _variance_covariance(std_err: int, spatial_std_err: int, design: np.ndarray, 
                         spatial_correlation: np.ndarray=None):
    '''Creates a variance covariance matrix. Forumla: 
    Variance-covariance = (design' * spatial correlation * design)^-1
    
    Returns numpy array.
    '''
    # Independent case
    if spatial_correlation is None or len(spatial_correlation) == 0:
        regularized = np.matmul(design.T, design)
        variance_covariance = (std_err**2 + spatial_std_err**2) * np.linalg.inv(
            regularized + np.identity(len(regularized)) * 1e-5)
    else:
        regularized = np.matmul(np.matmul(design.T, np.linalg.inv(
            spatial_correlation)), design)
        variance_covariance = np.linalg.inv(regularized + np.identity(len(regularized)) * 1e-5)

    return variance_covariance

def se_information_matrix(
        std_err: int, spatial_std_err: int, design: np.ndarray, 
        distances: np.ndarray, lengthscale: int):
    '''Creates an information gain matrix scaled to show variance. 

    Returns squared exponential information matrix square root scaled.
    '''
    # Independent case
    if lengthscale == 0:
        variance_covar = _variance_covariance(std_err, spatial_std_err, design)
    else:
        spatial_corr = _spatial_correlation(
            distances, std_err, spatial_std_err, lengthscale)
        variance_covar = _variance_covariance(
            std_err, spatial_std_err, design, spatial_corr)

    # Information gain at point i = -(log det variance covariance - log det 
    # variance covariance without point i)
    sign, with_info = np.linalg.slogdet(variance_covar)
    information_gain = []
    for i in range(len(distances)):
        design_minus = np.delete(design, i, 0)
        if lengthscale == 0:
            variance_covar_minus = _variance_covariance(
                std_err, spatial_std_err, design_minus)
        else:
            spatial_corr_minus = np.delete(np.delete(spatial_corr, i, 0), i, 1)
            variance_covar_minus = _variance_covariance(
                std_err, spatial_std_err, design_minus, spatial_corr_minus)
        sign, without_info = np.linalg.slogdet(variance_covar_minus)
        information_gain.append(-(with_info - without_info))
    
    information_gain_scaled = np.array(information_gain)
    information_gain_scaled = np.divide(information_gain_scaled, .125)
    information_gain_scaled = np.float_power(information_gain_scaled, .25)

    return information_gain_scaled

if __name__ == "__main__":
    gps_pattern = "East Africa Standardized Data/*.shp"
    boundary_pattern = "East Africa Geodata/**/* Bounds/sdr_subnational_boundaries.shp"
    output_directory = "East Africa Information Gain/"
    gps_paths = sorted(glob.glob(gps_pattern))
    boundary_paths = sorted(glob.glob(boundary_pattern))
    
    for gps_path, boundary_path in zip(gps_paths, boundary_paths):
        data = gpd.read_file(gps_path)
        bounds = gpd.read_file(boundary_path)
        year = int(data.at[0, "DHSYEAR"])
        n_spots = data.sample(100)
        
        # Variables
        STD_ERR = .05
        SPATIAL_STD_ERR = .5
        lenscales = [5, 1, .5, .1, 0]
        gps_columns = 6

        # Create design matrix where factors are column vectors
        design = n_spots.iloc[:, gps_columns:-1].to_numpy()

        # Distance values
        x = n_spots["LATNUM"].to_numpy()
        y = n_spots["LONGNUM"].to_numpy()
        distances = np.stack((x, y), axis = -1)

        # Initialize graphic
        figure, axis = plt.subplots(1, len(lenscales), figsize=(30, 6), sharey=True)

        # Create spatial correlation matrix
        for i in range(len(lenscales)):
            information_gain_scaled = se_information_matrix(
                STD_ERR, SPATIAL_STD_ERR, design, distances, lenscales[i])

            # Plot information gain with lengthscale
            # TIF
            # raster = "Uganda Covariate Rasters/uganda_Rainfall_CHIRPS_2km_2018.tif"
            # with rasterio.open(raster) as src:
            #     image = src.read(1)
            #     extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, 
            #             src.bounds.top)
            # axis[i].imshow(image, cmap='gray', extent=extent)

            # CSV
            # axis[i].scatter(x, y, c=information_gain_scaled, cmap="viridis", 
            #                 marker="$O$")
            # axis[i].get_xaxis().set_visible(False)
            # axis[i].get_yaxis().set_visible(False)
            # figure.subplots_adjust(wspace=0)
            # if lenscales[i] == 0:
            #     axis[i].set_title("Independent")
            # else:
            #     axis[i].set_title(f"GLS: length scale={lenscales[i]}") 

            # SHP
            bounds.plot(ax=axis[i], color="gray")
            n_spots["info_gain"] = information_gain_scaled.tolist()
            n_spots.plot(column="info_gain", ax=axis[i], markersize=15)
            axis[i].get_xaxis().set_visible(False)
            axis[i].get_yaxis().set_visible(False)
            figure.subplots_adjust(wspace=0)
            if lenscales[i] == 0:
                axis[i].set_title("Independent")
            else:
                axis[i].set_title(f"GLS: length scale={lenscales[i]}") 
        
        basename = os.path.basename(gps_path)
        components = basename.split(".")
        title = f"{components[0].title()} Information Gain of All Covariates"
        figure.suptitle(title, fontsize="xx-large")

        file = f"East Africa Information Gain/information_matrix_{components[0]}_{year}.png"
        plt.savefig(file, bbox_inches="tight", pad_inches=0) 
        plt.show()