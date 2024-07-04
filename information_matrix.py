'''
Information matrix

Returns a graph of information gain for n random points of Malaria covariates 
using different lengthscales
'''

import pandas as pd
import rasterio
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

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

def _variance_covariance(std_err: int, spatial_std_err: int, design: list, 
                         spatial_correlation: list=None):
    '''Creates a variance covariance matrix. Forumla: 
    Variance-covariance = (design' * spatial correlation * design)^-1
    
    Returns numpy array.
    '''
    # Independent case
    if spatial_correlation is None:
        variance_covariance = (std_err**2 + spatial_std_err**2) * np.linalg.inv(
            np.matmul(design.T, design))
    else:
        variance_covariance = np.linalg.inv(np.matmul(
            np.matmul(design.T, np.linalg.inv(spatial_correlation)), design))

    return variance_covariance

def information_matrix(
        std_err: int, spatial_std_err: int, design: list, distances: list,
        lengthscale: int):
    '''Creates an information gain matrix scaled to show variance. 

    Returns information matrix square root scaled.
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
    data_path = "Uganda Malaria Data/uganda_mock_malaria_cases_2km_2018.csv"
    data = pd.read_csv(data_path)
    
    # Variables
    STD_ERR = .05
    SPATIAL_STD_ERR = .5
    lenscales = [5, 1, .5, .1, 0]

    # Create design matrix where factors are column vectors
    # lst = data["lst"].to_numpy()
    rain = data["rainfall"].to_numpy()
    # elevation = data["elevation"].to_numpy()
    # design = np.stack((lst, rain, elevation), axis = -1)
    design = rain[:, np.newaxis]

    # Distance values
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    distances = np.stack((x, y), axis = -1)

    # Initialize graphic
    figure, axis = plt.subplots(1, len(lenscales), figsize=(30, 6), sharey=True)

    # Create spatial correlation matrix
    for i in range(len(lenscales)):
        information_gain_scaled = information_matrix(
            STD_ERR, SPATIAL_STD_ERR, design, distances, lenscales[i])

        # Plot information gain with lengthscale
        # Raster underneath
        raster = "Uganda Covariate Rasters/uganda_Rainfall_CHIRPS_2km_2018.tif"
        with rasterio.open(raster) as src:
            image = src.read(1)
            extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, 
                      src.bounds.top)
        axis[i].imshow(image, cmap='gray', extent=extent)
        # Scatter
        axis[i].scatter(x, y, c=information_gain_scaled, cmap="viridis", 
                        marker="$O$")
        axis[i].get_xaxis().set_visible(False)
        axis[i].get_yaxis().set_visible(False)
        figure.subplots_adjust(wspace=0)
        if lenscales[i] == 0:
            axis[i].set_title("Independent")
        else:
            axis[i].set_title(f"GLS: length scale={lenscales[i]}") 
    
    title = "Uganda Information Gain of All Covariates"
    figure.suptitle(title, fontsize="xx-large")

    file = "Uganda Malaria Data/information_matrix_uganda_all_2km_2018.png"
    plt.savefig(file, bbox_inches="tight", pad_inches=0) 
    plt.show()


