import numpy
import rioxarray
import matplotlib.pyplot as plt

# Returns a graphic of a correlation matrix given covariates
if __name__ == "__main__":
    # Covariates
    lst = rioxarray.open_rasterio("Uganda Standardized Rasters/standard_uganda_LSTday_2km_2018.tif")
    rainfall = rioxarray.open_rasterio("Uganda Standardized Rasters/standard_uganda_Rainfall_CHIRPS_2km_2018.tif")
    elevation= rioxarray.open_rasterio("Uganda Standardized Rasters/standard_uganda_elevation_2km.tif")

    # Different smoothing graphics
    distances = [0, 102, 22, 10] # in km

    # Initialize graphic
    figure, axis = plt.subplots(1, len(distances), figsize=(15, 6))

    # Subtract smooth from raw and then create a correlation graphic between covariates
    for i in range(len(distances)):
        if i > 0:
            # Subtract smooth
            lst_smooth = rioxarray.open_rasterio("Uganda Smoothed Rasters/smooth_uganda_lst_" + str(distances[i]) + "km_2018.tif")
            rainfall_smooth = rioxarray.open_rasterio("Uganda Smoothed Rasters/smooth_uganda_rain_" + str(distances[i]) + "km_2018.tif")
            elevation_smooth = rioxarray.open_rasterio("Uganda Smoothed Rasters/smooth_uganda_elevation_" + str(distances[i]) + "km_2018.tif")
            
            lst = lst - lst_smooth
            rainfall = rainfall - rainfall_smooth
            elevation = elevation - elevation_smooth

        lst_values = lst.to_numpy().flatten()
        rainfall_values = rainfall.to_numpy().flatten()
        elevation_values = elevation.to_numpy().flatten()

        # Prepare matrix in style of slideshow example
        covariates = numpy.vstack((lst_values, rainfall_values, elevation_values))
        correlation_matrix = numpy.corrcoef(covariates)
        pretty_correlation_matrix = numpy.absolute(numpy.rot90(correlation_matrix))

        # Plot the correlation matrix
        axis[i].matshow(pretty_correlation_matrix, cmap="viridis")
        axis[i].set_xticks([0, 1, 2])
        axis[i].set_yticks([0, 1, 2])
        axis[i].set_xticklabels(["LST", "Rainfall", "Elevation"], rotation=90, fontsize=16)
        axis[i].set_yticklabels(["Elevation", "Rainfall", "LST"], fontsize=16)
        axis[i].xaxis.set_ticks_position('bottom')
        axis[i].xaxis.set_label_position('bottom')
        if i > 0:
            axis[i].set_title(str(-distances[i]) + "m smooth", fontsize=20)
        else:
            axis[i].set_title("Raw Data", fontsize=20)
    
    # Save graphic
    plt.tight_layout()
    plt.savefig("Uganda Malaria Data/covariates_correlation_uganda_2km_2018.png", bbox_inches="tight") 