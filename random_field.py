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

def random_field_train(
        x_train: list, z_train: list, lengthscale: int, dims: int, 
        covariates: int):
    '''Given list x_train of coordinates in the first dims columns and 
    covariates in the next columns, trains dims columns with the RBF kernel and 
    covariates columns with the linear model.

    Returns model from gpflow.
    '''
    # Create kernel
    rbf = gpflow.kernels.RBF(active_dims = range(dims + 1), 
                             lengthscales=lengthscale)
    linear = gpflow.kernels.Linear(active_dims = range(dims, dims + covariates))
    kernel = rbf + linear

    # Train model
    model = gpflow.models.GPR(data = (x_train, z_train), kernel = kernel)

    # Fix lengthscale
    # gpflow.set_trainable(model.kernel.kernels[0].lengthscales, False) 

    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)

    return model

def random_field_test(model: gpflow.models.GPR, test_data: list, shape: list):
    '''Fits a random field using the Gaussian Process given a model and 
    test_data. Reshapees the results into shape.

    Returns tuples Z_fmean_grid and Z_fvar_grid from gpflow predict_f.
    '''
    Z_fmean_grid, Z_fvar_grid = model.predict_f(test_data)

    # Reshape the grid predictions to match the grid shape
    Z_fmean_grid = Z_fmean_grid.reshape(shape.shape)
    Z_fvar_grid = Z_fvar_grid.reshape(shape.shape)

    return (Z_fmean_grid, Z_fvar_grid)

if __name__ == "__main__":
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    year = "2000"

    # Load in Malaria data
    uganda_path = f"Uganda Malaria Data/uganda_mock_malaria_cases_2km_{year}.csv"
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

    mal_prev_path = f"Uganda Malaria Data/uganda_mock_malaria_prevalence_2km_{year}.tif"
    malaria_prevalence = rioxarray.open_rasterio(mal_prev_path).squeeze()
    left, bottom, right, top = malaria_prevalence.rio.bounds()
    x = np.linspace(left, right, len(X_test))
    y = np.linspace(bottom, top, len(X_test))
    x1, x2 = np.meshgrid(x, y)
    X_grid = np.stack((x1.ravel(), x2.ravel()), axis = -1)
    x_points = xr.DataArray(X_grid[:, 0])
    y_points = xr.DataArray(X_grid[:, 1])

    covariates = [f"LSTday_2km_{year}", f"Rainfall_CHIRPS_2km_{year}", "elevation_2km"]
    for covariate in covariates:
        src_path = f"Uganda Standardized Rasters/standard_uganda_{covariate}.tif"
        src = rioxarray.open_rasterio(src_path)
        data = src.sel(x=x_points, y=y_points, method="nearest").to_numpy()
        X_grid = np.concatenate((X_grid, data.reshape(-1, 1)), axis = -1)

    # Different lengthscales in graphic
    lenscale = [1]
    
    # Initialize graphic
    figure, axis = plt.subplots(1, len(lenscale), figsize=(7, 6), sharey=True)

    for i in range(len(lenscale)):
        model = random_field_train(X_train, Z_train, lenscale[i], 2, 3)
        Z_fmean_grid, Z_fvar_grid = random_field_test(model, X_grid, x1)

        # Add figure to graphic
        image = axis.contourf(x, y, Z_fmean_grid, levels=100, cmap='viridis', 
                              vmax=.8)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image)
        # axis[i].get_xaxis().set_visible(False)
        # axis[i].get_yaxis().set_visible(False)
        # figure.subplots_adjust(wspace=0)
        # axis[i].set_title(f"GLS: length scale={lenscale[i]}")
        axis.set_title(f"GLMM Predicted Malaria Prevalence Uganda {year}")

    title = f"Uganda Malaria Data/randomfield_uganda_mock_malaria_prevalene_2km_{year}.png"
    plt.savefig(title, bbox_inches="tight", pad_inches=0) 
    plt.show()