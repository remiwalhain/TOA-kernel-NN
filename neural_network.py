import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

"""
IMPORT DATA
"""
# Load kernel data: SPECIFY FILE NAMES
kernel_ds = xr.open_dataset("ERA5_kernel_ts_TOA.nc")
kernel = kernel_ds["TOA_clr"][0] # shape: (lat, lon) - Month of January # <----------------------------------- Modify month
kernel_flat = kernel.values.flatten()  # shape: (n_points,)

# Load predictors
ERA5_single_layer = xr.open_dataset("2013_jan_single_layer.nc")
#ERA5_pressures = xr.open_dataset("2013_jul_Ta_q_pres.nc")
#ERA5_ta_pressures = ERA5_pressures["t"]
#ERA5_q_pressures = ERA5_pressures["q"]
ERA5_ta_pressures = xr.open_dataset("2013_jan_Ta_pressures.nc")
ERA5_q_pressures = xr.open_dataset("2013_jan_wv_pres.nc")

predictor_vars = ["q", "skt", "sp", "tcwv"] # <----------------------------------- Arrows indicate where to modify
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
            field = ERA5_ta_pressures["t"].sel(pressure_level=pres).mean(dim='valid_time') # Change pressures
            field_regridded = field.interp_like(kernel)  # regrid to kernel grid
            predictors.append(field_regridded.values.flatten())
    else:
        for pres in q_pres_to_use: # wv/q
            field = ERA5_q_pressures["q"].sel(pressure_level=pres).mean(dim='valid_time') # Change pressures
            field_regridded = field.interp_like(kernel)  # regrid to kernel grid
            predictors.append(field_regridded.values.flatten())

X = np.stack(predictors, axis=1) # shape: (n_points, n_features)

# Mask NaN points
valid_mask = np.isfinite(kernel_flat) & np.all(np.isfinite(X), axis=1)

X_valid = X[valid_mask]
y_valid = kernel_flat[valid_mask]

print(f"Valid points: {X_valid.shape[0]}")

"""
SPLIT + SCALE
"""
# Split data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(
    X_valid, y_valid, test_size=0.2, random_state=42 # sets the random seed so the split is reproducible
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

"""
BUILD NEURAL NETWORK
"""
# Input layer -> hidden layers with activations -> output node
# For each spatial point, the predictors are a 'input_dim'-element vector
# Each input vector (10512, input_dim) predicts corresponding kernel value
# For each point, the target is the corresponding kernel value at that location
# The NN takes each row of X_train (predictors at a grid point) and tries to predict the corresponding value in y_train (kernel value at that point)
# It adjusts its weights across the 5 inputs, hidden layers, and single output to minimize error between predicted and actual kernel values

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

model = KernelRegressor(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100 # Can be modified
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy() # Run the model on the test data

r2 = r2_score(y_test, y_pred) # Compare the true values to the predicted values
print(f"PyTorch MLP R-squared: {r2:.3f}") # MLP: multilayer perceptron (aka Neural Network)

"""
PLOTTING
"""
# To visualize, need to predict on all valid points, then scatter those predictions back into a (lat, lon) array
# Predict on all valid points (both train and test combined) for spatial map
X_valid_scaled = scaler.transform(X_valid)
X_valid_tensor = torch.tensor(X_valid_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    y_pred_all = model(X_valid_tensor).numpy().flatten()

# Reconstruct the full lat-lon grid
full_pred_grid = np.full_like(kernel_flat, np.nan)  # start with nan array
full_pred_grid[valid_mask] = y_pred_all

# Reshape to (lat, lon)
full_pred_2D = full_pred_grid.reshape(kernel.shape)
true_kernel_2D = kernel.values  # original kernel

# Flip to be the correct orientation
true_kernel_2D_plot = np.flipud(true_kernel_2D)
full_pred_2D_plot = np.flipud(full_pred_2D)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("True Kernel")
plt.imshow(true_kernel_2D_plot, cmap="viridis", origin="lower")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Predicted Kernel - " f"PyTorch MLP R-squared: {r2:.3f}")
plt.imshow(full_pred_2D_plot, cmap="viridis", origin="lower")
plt.colorbar()

plt.tight_layout()
#plt.show()
plt.savefig("q750_q1000_skt_sp_tcwv_NN_estimate.png") # <-----------------------------------

"""
SAVE TRAINED MODEL
"""
# Save trained model weights
torch.save(model.state_dict(), "q750_q1000_skt_sp_tcwv_weights.pth") # <-----------------------------------

# Save the scaler
joblib.dump(scaler, "q750_q1000_skt_sp_tcwv_scaler.pkl") # <-----------------------------------

# Save predictions
np.save("q750_q1000_skt_sp_tcwv_predicted_kernel_2D.npy", full_pred_2D_plot) # <-----------------------------------

# Save metrics (R-squared, losses)
results = {
    "final_loss": float(loss.item()),
    "R_squared": r2
}
with open("q750_q1000_skt_sp_tcwv_metrics.json", "w") as f: # <-----------------------------------
    json.dump(results, f, indent=4)


# To save straight to NetCDF
#pred_ds = xr.DataArray(
#    full_pred_2D,
#    coords={"latitude": kernel["latitude"], "longitude": kernel["longitude"]},
#    dims=["latitude", "longitude"],
#    name="predicted_kernel"
#)
#pred_ds.to_netcdf("predicted_kernel.nc")