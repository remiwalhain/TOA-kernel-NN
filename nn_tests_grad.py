import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score

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

predictor_vars = ["q", "t", "skt", "sp", "t2m", "tcwv"] # <----------------------------------- Arrows indicate where to modify
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
input_dim = 7  # adjust if needed
model = KernelRegressor(input_dim)

# Load the saved weights
model.load_state_dict(torch.load("q750_q1000_t100_skt_sp_t2m_tcwv_weights.pth")) # <----------------------------------- 
model.eval()

# Load the saved scaler
scaler = joblib.load("q750_q1000_t100_skt_sp_t2m_tcwv_scaler.pkl") # <----------------------------------- 

# Apply to new input data (February predictors)
X_valid_scaled = scaler.transform(X_valid)

# Convert data to tensor and make predictions
X_tensor = torch.tensor(X_valid_scaled, dtype=torch.float32)

with torch.no_grad():
    y_pred = model(X_tensor).numpy().flatten()

r2 = r2_score(y_valid, y_pred)
print(f"R-squared: {r2:.4f}")

x_sample = torch.tensor(X_valid_scaled.mean(axis=0), dtype=torch.float32, requires_grad=True)

# Forward pass
y_pred = model(x_sample.unsqueeze(0))  # shape: (1, 1)

# Backward pass to compute gradients
y_pred.backward()

# Get gradients: dR/dX
grads = x_sample.grad.detach().numpy()

feature_names = ["q750", "q1000", "t100", "skt", "sp", "t2m", "tcwv"]

plt.figure(figsize=(8, 5))
bars = plt.bar(feature_names, grads, alpha=0.7)
plt.ylabel("dR/dX")
plt.xlabel("Variables")
plt.title("Sensitivity of NN Output to Input Variables (January 2013 model,\n variables: q750, q1000, t100, skt, sp, t2m, tcwv)")
plt.axhline(0, color='black', linewidth=0.5)

for bar, val in zip(bars, grads):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.005 if height >= 0 else height - 0.007, # Offset up or down
        f"{val:.3f}", # Format to 3 decimals
        ha='center',
        va='bottom' if height >= 0 else 'top',
        fontsize=9
    )

plt.tight_layout()
plt.savefig("Sensitivity_q750_q1000_t100_skt_sp_t2m_tcwv")