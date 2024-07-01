import pandas as pd
import rasterio
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
# Returns a raster of Malaria prevalence using a linear function on the covariates plus a random field function
if __name__ == "__main__":
    # Load in Malaria data
    data = pd.read_csv("Uganda Malaria Data/mock_malaria_cases_uganda_2km_2018.csv")
    
    # Variables
    std_err = .05
    spatial_std_err = .5
    lengthscales = [5, 1, .5, .1, 0]

    # Create design matrix where factors are column vectors
    lst = data["lst"].to_numpy()
    rain = data["rainfall"].to_numpy()
    elevation = data["elevation"].to_numpy()
    design = np.stack((lst, rain, elevation), axis = -1)
    # design = rain[:, np.newaxis]

    # Distance values
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    distances = np.stack((x, y), axis = -1)

    # Initialize graphic
    figure, axis = plt.subplots(1, len(lengthscales), figsize=(30, 6), sharey=True)

    # Create spatial correlation matrix
    for i in range(len(lengthscales)):
        # Independent case
        if lengthscales[i] == 0:
            variance_covar = (std_err**2 + spatial_std_err**2) * np.linalg.inv(np.matmul(design.T, design))
        else:
            # Spatial correlation = std err^2 * I + spatial std err^2 * exp(-D^2/r^2/2)
            spatial_corr = (
                np.identity(len(distances)) * std_err**2 + spatial_std_err**2 * 
                np.exp(-(np.square(distance_matrix(distances, distances))) / lengthscales[i]**2 / 2)
            )
            # Variance-covariance = (X' * spatial correlation * X)^-1
            variance_covar = np.linalg.inv(np.matmul(np.matmul(design.T, np.linalg.inv(spatial_corr)), design))

        # Information gain at point i = -(log det variance covariance - log det variance covariance without point i)
        sign, with_info = np.linalg.slogdet(variance_covar)
        information_gain = []
        for j in range(len(distances)):
            design_minus = np.delete(design, j, 0)
            if lengthscales[i] == 0:
                variance_covar_minus = (std_err**2 + spatial_std_err**2) * np.linalg.inv(np.matmul(design_minus.T, design_minus))
            else:
                spatial_corr_minus = np.delete(np.delete(spatial_corr, j, 0), j, 1)
                variance_covar_minus = np.linalg.inv(np.matmul(np.matmul(design_minus.T, np.linalg.inv(spatial_corr_minus)), design_minus))
            sign, without_info = np.linalg.slogdet(variance_covar_minus)
            information_gain.append(-(with_info - without_info))
        
        information_gain_scaled = np.array(information_gain)
        information_gain_scaled = np.divide(information_gain_scaled, .125)
        information_gain_scaled = np.float_power(information_gain_scaled, .25)

        # Plot information gain with lengthscale
        # Raster
        # with rasterio.open("Uganda Covariate Rasters/uganda_Rainfall_CHIRPS_2km_2018.tif") as src:
        #     image = src.read(1)  # Read the first band
        #     extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        # axis[i].imshow(image, cmap='gray', extent=extent)
        # Scatter
        # axis[i].scatter(x, y, c=information_gain_scaled, cmap="viridis", marker="$O$")
        axis[i].scatter(x, y, c=information_gain_scaled, cmap="viridis")
        axis[i].get_xaxis().set_visible(False)
        axis[i].get_yaxis().set_visible(False)
        figure.subplots_adjust(wspace=0)
        if lengthscales[i] == 0:
            axis[i].set_title("Independent")
        else:
            axis[i].set_title("GLS: length scale=" + str(lengthscales[i]))  
    figure.suptitle("Uganda Information Gain of All Covariates", fontsize="xx-large")

    plt.savefig("Uganda Malaria Data/mock_malaria_cases_uganda_all_information_matrix_2km_2018.png", bbox_inches="tight", pad_inches=0) 
    plt.show()


