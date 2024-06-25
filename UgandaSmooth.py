import rioxarray
import xarray
from scipy.ndimage import generic_filter

# Function for generic_filter
def mean_filter(window):
    return window.mean()

# Array for generic_filter
covariate = rioxarray.open_rasterio("Uganda Standardized Rastors/standard_uganda_Rainfall_CHIRPS_2km_2018.tif")
covariate_values = covariate.to_numpy()

# Convert raster distance (degrees) to pixel size for generic_filter
smooth_distances = [10, 22, 51] # in km

# Apply smoothing
for distance in smooth_distances:
    smoothed = generic_filter(covariate_values, mean_filter, size=distance)
    smoothed_data_xr = xarray.DataArray(smoothed, dims=covariate.dims, coords=covariate.coords, attrs=covariate.attrs)
    smoothed_data_xr.rio.to_raster("Uganda Smoothed Rasters/smooth_uganda_rain_" + str(distance) + "km_2018.tif")