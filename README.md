# Neural Network/Linear Regression TOA Kernel Reconstruction

This repository contains the code used to reconstruct the top-of-atmosphere (TOA) clear-sky surface temperature kernel from ERA5 atmospheric predictors using both 
linear regression and neural networks (NNs).

## Overview
The project compares linear regression and NN approaches to determine how well atmospheric variables from ERA5 can reproduce the spatial structure of the 
TOA clear-sky temperature kernel. 

### 1. linear_regression.py 
* Computes the multivariable linear regressions with statsmodels.api
* Prints the regression summary
* Option of plotting single variable linear regressions

### 2. linear_kernel_estimates_TOA.py
* Plots the linear regression predictions as a longitude by latitude map

### 3. neural_networks.py
* Runs and trains the neural network using PyTorch
* Splits the data (into train and test) and scales it
* Plots the NN prediction as a longitude by latitude map
* Saves the trained model weights and scaler

### 4. nn_tests.py
* Runs the trained NN model
* Plots error statistics
* Other various tests

### 5. nn_tests_grad.py
* Computes the gradients of each input variable
* Plots the gradients as a bar graph

## Usage
1. Download ERA5 data from the Copernicus Climate Data Store (https://cds.climate.copernicus.eu/). 
2. Download TOA clear-sky kernels (https://data.mendeley.com/datasets/vmg3s67568/4).
3. Modify the file names in the scripts to match the downloaded data.
4. Enter the desired variables and pressures in the predictor_vars, t_pres_to_use, and q_pres_to_use arrays.
