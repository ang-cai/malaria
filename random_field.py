import pandas as pd
import numpy as np
import rioxarray
import xarray
import gpflow
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Returns a raster of Malaria prevalence using a linear function on the covariates plus a random field function
if __name__ == "__main__":
    # Load in Malaria data
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    uganda_data = pd.read_csv("Uganda Malaria Data/mock_malaria_cases_uganda_2km_2018.csv")
    
    # numpy requires column vectors
    Z = uganda_data["malaria"].to_numpy().reshape(-1, 1)
    x = uganda_data["x"].to_numpy()
    y = uganda_data["y"].to_numpy()
    lst = uganda_data["lst"].to_numpy()
    rain = uganda_data["rainfall"].to_numpy()
    elevation = uganda_data["elevation"].to_numpy()
    X = np.stack((x, y, lst, rain, elevation), axis = -1)

    # Create train points and test grid
    X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.9)
    malaria_prevalence = rioxarray.open_rasterio("Uganda Malaria Data/mock_malaria_prevelance_uganda_2km_2018.tif").squeeze()
    left, bottom, right, top = malaria_prevalence.rio.bounds()
    x = np.linspace(left, right, len(X_test))
    y = np.linspace(bottom, top, len(X_test))
    x1, x2 = np.meshgrid(x, y)
    X_grid = np.stack((x1.ravel(), x2.ravel()), axis = -1)
    covariates = ["LSTday_2km_2018", "Rainfall_CHIRPS_2km_2018", "elevation_2km"]
    x_points = xarray.DataArray(X_grid[:, 0])
    y_points = xarray.DataArray(X_grid[:, 1])
    for covariate in covariates:
        src = rioxarray.open_rasterio("Uganda Standardized Rasters/standard_uganda_" + covariate + ".tif")
        data = src.sel(x=x_points, y=y_points, method="nearest")
        X_grid = np.concatenate((X_grid, data.to_numpy().reshape(-1, 1)), axis = -1)

    # Different lengthscales in graphic
    lengthscales = [5, 1, .5, .1, .05]
    
    # Initialize graphic
    figure, axis = plt.subplots(1, len(lengthscales), figsize=(30, 6), sharey=True)

    for i in range(len(lengthscales)):
        # Kernal is made of RBF acting on the coordinates and linear acting on the covariates
        rbf = gpflow.kernels.RBF(active_dims = [0, 1], lengthscales=lengthscales[i])
        linear = gpflow.kernels.Linear(active_dims = [2, 3, 4])
        kernel = rbf + linear

        # Train model
        model = gpflow.models.GPR(data = (X_train, Z_train), kernel = kernel)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)

        # Fit model onto coordinate area
        Z_fmean_grid, Z_fvar_grid = model.predict_f(X_grid)

        # Reshape the grid predictions to match the grid shape
        Z_fmean_grid = Z_fmean_grid.reshape(x1.shape)
        Z_fvar_grid = Z_fvar_grid.reshape(x1.shape)

        # Add figure to graphic
        axis[i].contourf(x, y, Z_fmean_grid, levels=100, cmap='viridis')
        axis[i].get_xaxis().set_visible(False)
        axis[i].get_yaxis().set_visible(False)
        figure.subplots_adjust(wspace=0)
        axis[i].set_title("GLS: length scale=" + str(lengthscales[i]))

    plt.savefig("Uganda Malaria Data/mock_malaria_cases_uganda_randomfield_2km_2018.png", bbox_inches="tight", pad_inches=0) 
    plt.show()