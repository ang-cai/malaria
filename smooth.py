'''
Smooth

Returns a smoothed version of a raster given different distance by distance 
windows. 
'''
import rioxarray 
import xarray as xr
from scipy.ndimage import generic_filter

def mean_filter(window):
    '''Returns the mean of a given window area, which is used for generic_filter 
    to take the mean of values with a given pixel size window.
    '''
    return window.mean()

if __name__ == "__main__":
    cov_path = "Uganda Standardized Rasters/standard_uganda_Rainfall_CHIRPS_2km_2018.tif"
    covariate = rioxarray.open_rasterio(cov_path)
    covariate_values = covariate.to_numpy()

    # Convert raster distance (km) to pixel size for generic_filter
    KM_PER_PIXEL = 2
    smooth_distances = [5, 11, 51] # in pixels

    # Apply smoothing
    for distance in smooth_distances:
        smoothed = generic_filter(covariate_values, mean_filter, size=distance)
        smoothed_data_xr = xr.DataArray(
            smoothed, dims=covariate.dims, coords=covariate.coords, 
            attrs=covariate.attrs
        )
        path = f"Uganda Smoothed Rasters/smooth_uganda_rain_{distance*KM_PER_PIXEL}km_2018.tif"
        smoothed_data_xr.rio.to_raster(path)