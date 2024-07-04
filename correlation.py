'''
Creates correlation matrix 

Returns a graphic of a correlation matrix given covariates
'''

import numpy as np
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray.raster_dataset

def subtract_smooth(dataset: xr.Dataset, covariate: str, distance: int, 
                    year: int) -> list:
    '''Subtracts a smoothed by km distance raster from covariate in given year
    and flattens the values into a numpy array.

    Returns 1D list of values.
    '''
    path = f"Uganda Smoothed Rasters/smooth_uganda_{covariate}_{distance}km_{year}.tif"
    smooth = rioxarray.open_rasterio(path)
    smoothed = (dataset - smooth).to_numpy().flatten()
    return smoothed

if __name__ == "__main__":
    # Covariates
    lst_path = "Uganda Standardized Rasters/standard_uganda_LSTday_2km_2018.tif"
    rain_path = "Uganda Standardized Rasters/standard_uganda_Rainfall_CHIRPS_2km_2018.tif"
    ele_path = "Uganda Standardized Rasters/standard_uganda_elevation_2km.tif"

    lst = rioxarray.open_rasterio(lst_path)
    rainfall = rioxarray.open_rasterio(rain_path)
    elevation = rioxarray.open_rasterio(ele_path)
    
    # Different smoothing graphics
    distances = [0, 102, 22, 10] # in km
    year = 2018

    # Initialize graphic
    figure, axis = plt.subplots(1, len(distances), figsize=(15, 6))

    # Subtract smooth and then create a correlation graphic between covariates
    for i in range(len(distances)):
        if distances[i] > 0:
            lst_values = subtract_smooth(lst, "lst", distances[i], year)
            rainfall_values = subtract_smooth(rainfall, "rain", distances[i], 
                                              year)
            elevation_values = subtract_smooth(elevation, "elevation", 
                                               distances[i], year)

        # Prepare matrix in style of slideshow example
        covariates = np.vstack((lst_values, rainfall_values, elevation_values))
        correlation_matrix = np.corrcoef(covariates)
        pretty_correlation_matrix = np.absolute(np.rot90(correlation_matrix))

        # Plot the correlation matrix
        axis[i].matshow(pretty_correlation_matrix, cmap="viridis")
        axis[i].set_xticks([0, 1, 2])
        axis[i].set_yticks([0, 1, 2])
        axis[i].set_xticklabels(["LST", "Rainfall", "Elevation"], rotation=90, 
                                fontsize=16)
        axis[i].set_yticklabels(["Elevation", "Rainfall", "LST"], fontsize=16)
        axis[i].xaxis.set_ticks_position('bottom')
        axis[i].xaxis.set_label_position('bottom')
        if distances[i] > 0:
            axis[i].set_title(str(-distances[i]) + "km smooth", fontsize=20)
        else:
            axis[i].set_title("Raw Data", fontsize=20)
    
    # Save graphic
    plt.tight_layout()

    title = "Uganda Malaria Data/correlation_uganda_lst_rain_elevation_2km_2018.png"
    plt.savefig(title, bbox_inches="tight") 