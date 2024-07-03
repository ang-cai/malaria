'''
Random Field

Returns a raster of Malaria prevalence using a linear function on the covariates 
plus a random field function
'''
import pandas as pd
import numpy as np
import rioxarray
import xarray as xr
import gpflow
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == "__main__":
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    # Load in Malaria data
    uganda_path = "Uganda Malaria Data/uganda_mock_malaria_cases_2km_2018.csv"
    uganda_data = pd.read_csv(uganda_path)
    
    # numpy requires column vectors
    Z = uganda_data["malaria"].to_numpy().reshape(-1, 1)
    x = uganda_data["x"].to_numpy()
    y = uganda_data["y"].to_numpy()
    lst = uganda_data["lst"].to_numpy()
    rain = uganda_data["rainfall"].to_numpy()
    elevation = uganda_data["elevation"].to_numpy()
    X = np.stack((x, y, lst, rain, elevation), axis = -1)

    # Create train points and test grid
    X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.5)

    mal_prev_path = "Uganda Malaria Datauganda_mock_malaria_prevelance_2km_2018.tif"
    malaria_prevalence = rioxarray.open_rasterio(mal_prev_path).squeeze()
    left, bottom, right, top = malaria_prevalence.rio.bounds()
    x = np.linspace(left, right, len(X_test))
    y = np.linspace(bottom, top, len(X_test))
    x1, x2 = np.meshgrid(x, y)
    X_grid = np.stack((x1.ravel(), x2.ravel()), axis = -1)
    x_points = xr.DataArray(X_grid[:, 0])
    y_points = xr.DataArray(X_grid[:, 1])

    covariates = ["LSTday_2km_2018", "Rainfall_CHIRPS_2km_2018", "elevation_2km"]
    for covariate in covariates:
        src_path = f"Uganda Standardized Rasters/standard_uganda_{covariate}.tif"
        src = rioxarray.open_rasterio(src_path)
        data = src.sel(x=x_points, y=y_points, method="nearest").to_numpy()
        X_grid = np.concatenate((X_grid, data.reshape(-1, 1)), axis = -1)

    # Different lengthscales in graphic
    length = [5, 1, .5, .1, .05]
    
    # Initialize graphic
    figure, axis = plt.subplots(1, len(length), figsize=(30, 6), sharey=True)

    for i in range(len(length)):
        # Kernal is made of RBF acting on the coordinates and linear acting on the covariates
        rbf = gpflow.kernels.RBF(active_dims = [0, 1], lengthscales=length[i])
        linear = gpflow.kernels.Linear(active_dims = [2, 3, 4])
        kernel = rbf + linear

        # Train model
        model = gpflow.models.GPR(data = (X_train, Z_train), kernel = kernel)

        # Fix lengthscale?
        gpflow.set_trainable(model.kernel.kernels[0].lengthscales, False) 

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


    plt.savefig("Uganda Malaria Data/randomfield_uganda_mock_malaria_cases_2km_2018.png", bbox_inches="tight", pad_inches=0) 
    plt.show()