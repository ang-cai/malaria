import rioxarray
import xarray
from scipy.ndimage import generic_filter

def mean_filter(window):
    return window.mean()

covariate = rioxarray.open_rasterio("/Users/angelacai/TKI/Standardized Uganda/standard_uganda_Rainfall_CHIRPS_2km_2018.tif")
covariate_values = covariate.to_numpy()

resolution = covariate.rio.resolution()[0]
smooth_distances = [.1, .22, 1.02] # in meters

for distance in smooth_distances:
    smooth_window = int(distance / resolution)
    smoothed = generic_filter(covariate_values, mean_filter, size=smooth_window)
    smoothed_data_xr = xarray.DataArray(smoothed, dims=covariate.dims, coords=covariate.coords, attrs=covariate.attrs)
    smoothed_data_xr.rio.to_raster("Smoothed Uganda/smooth_uganda_rain_" + str(distance * 100) + "cm_2018.tif")