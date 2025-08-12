import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

"""
IMPORT DATA
"""
# Load kernel data: SPECIFY FILE NAMES
kernel_ds = xr.open_dataset("ERA5_kernel_ts_TOA.nc")
kernel = kernel_ds["TOA_clr"][0] # shape: (lat, lon) - Month of February # <----------------------------------- Modify month
kernel_flat = kernel.values.flatten()  # shape: (n_points,)

# Load predictors
ERA5_single_layer = xr.open_dataset("2013_jan_single_layer.nc") # <----------------------------------- 
# ERA5_pressures = xr.open_dataset("2013_feb_Ta_q_pres.nc") # TAKE OUT OR ADD IN IF DOING FEBRUARY **************************
# ERA5_ta_pressures = ERA5_pressures["t"] # TAKE OUT OR ADD IN IF DOING FEBRUARY **************************
# ERA5_q_pressures = ERA5_pressures["q"]  # TAKE OUT OR ADD IN IF DOING FEBRUARY **************************

ERA5_ta_pressures = xr.open_dataset("2013_jan_Ta_pressures.nc") # <----------------------------------- 
ERA5_q_pressures = xr.open_dataset("2013_jan_wv_pres.nc") # <----------------------------------- 

predictor_vars = ["q", "t", "skt", "sp", "tcwv"] # <----------------------------------- Arrows indicate where to modify
t_pres_to_use = ["100"] # Atmosphere temp pressures" <----------------------------------- 
q_pres_to_use = ["750", "1000"] # Water Vapor pressures <----------------------------------- 

predictors = []

for var in predictor_vars:
    if (var != "t") and (var != "q"): # Not t or q
        field = ERA5_single_layer[var].mean(dim='valid_time')  # shape: (lat, lon)
        field_regridded = field.interp_like(kernel)  # regrid to kernel grid
        predictors.append(field_regridded.values.flatten())
    elif (var == "t"): # t
        for pres in t_pres_to_use:
            field = ERA5_ta_pressures["t"].sel(pressure_level=pres).mean(dim='valid_time') # Take out ["t"] for FEB
            field_regridded = field.interp_like(kernel)  # regrid to kernel grid
            predictors.append(field_regridded.values.flatten())
    else:
        for pres in q_pres_to_use: # wv/q
            field = ERA5_q_pressures["q"].sel(pressure_level=pres).mean(dim='valid_time') # Take out ["q"] for FEB
            field_regridded = field.interp_like(kernel)  # regrid to kernel grid
            predictors.append(field_regridded.values.flatten())

X = np.stack(predictors, axis=1) # shape: (n_points, n_features)

# Mask NaN points
valid_mask = np.isfinite(kernel_flat) & np.all(np.isfinite(X), axis=1)

X_valid = X[valid_mask]
y_valid = kernel_flat[valid_mask]

"""
DEFINE MODEL
"""
class KernelRegressor(nn.Module):
    def __init__(self, input_dim):
        super(KernelRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), # Input layer has n nodes, where n = number of different variables
            nn.ReLU(), # applies the rectified linear activation function
            nn.Linear(128, 64), # First hidden layer: 128 nodes, second hidden layer: 64 nodes
            nn.ReLU(),
            nn.Linear(64, 1) # Output layer has one node: the predicted kernel value at that point
        )
    def forward(self, x):
        return self.net(x)

"""
RUN TRAINED MODEL
"""
# Set the same number of inputs used during training
input_dim = 6  # adjust if needed <----------------------------------- 
model = KernelRegressor(input_dim)

# Load the saved weights
model.load_state_dict(torch.load("q750_q1000_t100_skt_sp_tcwv_weights.pth")) # <----------------------------------- 
model.eval()

# Load the saved scaler
scaler = joblib.load("q750_q1000_t100_skt_sp_tcwv_scaler.pkl") # <----------------------------------- 

# Apply to new input data (February predictors)
X_valid_scaled = scaler.transform(X_valid)

# Convert data to tensor and make predictions
X_tensor = torch.tensor(X_valid_scaled, dtype=torch.float32)

with torch.no_grad():
    y_pred = model(X_tensor).numpy().flatten()

r2 = r2_score(y_valid, y_pred)
print(f"R-squared: {r2:.4f}")

"""
PLOTTING
"""
full_pred_grid = np.full_like(kernel_flat, np.nan)
full_pred_grid[valid_mask] = y_pred

# Reshape
full_pred_2D = full_pred_grid.reshape(kernel.shape)
true_kernel_2D = kernel.values

# Flip to be the correct orientation
true_kernel_2D_plot = np.flipud(true_kernel_2D)
full_pred_2D_plot = np.flipud(full_pred_2D)

"""
Errors
"""
errors = y_pred - y_valid # Vector of residuals
mean_error = np.mean(errors) # Mean error
std_error = np.std(errors) # Standard deviation of errors
rmse = np.sqrt(mean_squared_error(y_valid, y_pred)) # Root mean square error

# Scatter plot: Predicted vs True values
plt.figure(figsize=(8,5))
plt.scatter(y_valid, y_pred, alpha=0.3, s=10)
plt.plot([min(y_valid), max(y_valid)], [min(y_valid), max(y_valid)], 'r--', label='y=x line') # y=x line
plt.xlabel("True Kernel Values")
plt.ylabel("Predicted Kernel Values")
plt.legend()
plt.title("Predicted vs True Kernel Values")
plt.grid(True)
plt.tight_layout()
plt.savefig("q750_q1000_t100_skt_sp_tcwv_NN_error_scatter.png")  # <----------------------------------- 

# Histogram of Errors
plt.figure(figsize=(8,5))
plt.hist(errors, bins=50, alpha=0.8, edgecolor='grey')
plt.axvline(mean_error, color='red', linestyle='--', label=f"Mean: {mean_error:.3f}")
plt.axvline(mean_error + std_error, color='green', linestyle='--', label=f"+1 std (+{std_error:.3f})")
plt.axvline(mean_error - std_error, color='green', linestyle='--', label=f"-1 std (-{std_error:.3f})")
plt.xlabel("Prediction Error (Predicted Kernel - True Kernel)")
plt.ylabel("Count")
plt.title("Distribution of Prediction Errors")
plt.legend()
plt.tight_layout()
plt.savefig("q750_q1000_t100_skt_sp_tcwv_NN_error_hist.png")  # <----------------------------------- 

# Scatter Plot: Error vs. True Value
plt.figure(figsize=(8,5))
plt.scatter(y_valid, errors, alpha=0.3, s=10)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("True Kernel Value")
plt.ylabel("Prediction Error")
plt.title("Error vs True Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("q750_q1000_t100_skt_sp_tcwv_NN_error_scatter2.png")  # <----------------------------------- 

"""
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("True Kernel (Feb)")
plt.imshow(true_kernel_2D_plot, cmap="viridis", origin="lower")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Predicted Kernel (Feb) - " f"PyTorch MLP R-squared: {r2_feb:.3f}")
plt.imshow(full_pred_2D_plot, cmap="viridis", origin="lower")
plt.colorbar()

plt.tight_layout()
plt.savefig("q750_q1000_t100_skt_sp_t2m_tcwv_NN_feb_estimate.png")
"""