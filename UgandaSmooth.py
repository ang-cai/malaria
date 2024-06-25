import rioxarray
import xarray
from scipy.ndimage import generic_filter

# Function for generic_filter
def mean_filter(window):
    return window.mean()

# Array for generic_filter
covariate = rioxarray.open_rasterio("/Users/angelacai/TKI/Standardized Uganda/standard_uganda_elevation.tif")
covariate_values = covariate.to_numpy()

# Convert raster distance (degrees) to pixel size for generic_filter
DEGREES_PER_KM = 111
resolution = covariate.rio.resolution()[0]
smooth_distances = [.001, .0022, .0102] # in km

# Apply smoothing
for distance in smooth_distances:
    smooth_window = int(distance * DEGREES_PER_KM / resolution)
    smoothed = generic_filter(covariate_values, mean_filter, size=smooth_window)
    smoothed_data_xr = xarray.DataArray(smoothed, dims=covariate.dims, coords=covariate.coords, attrs=covariate.attrs)
    smoothed_data_xr.rio.to_raster("Smoothed Uganda/smooth_uganda_elevation_" + str(distance * 1000) + "m_2018.tif")