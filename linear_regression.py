import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the kernels
kernel_ds = xr.open_dataset("ERA5_kernel_ts_TOA.nc")
kernel = kernel_ds["TOA_clr"][0]  # Replace with actual name, kernel for January ONLY, 2.5 deg resolution

era5_files = ["q", "t", "skt", "sp", "tcwv"] # All variables to use
t_pres_to_use = ["100"] # Atmosphere temp pressures"
q_pres_to_use = ["750", "1000"] # Water Vapor pressures
all_variables = []
ERA5_single_layer = xr.open_dataset("2013_jan_single_layer.nc")
ERA5_ta_pressures = xr.open_dataset("2013_jan_Ta_pressures.nc")
ERA5_q_pressures = xr.open_dataset("2013_jan_wv_pres.nc")


kernel_lat = np.linspace(-90, 90, 73)
kernel_lon = np.linspace(0, 357.5, 144)

def regrid(field, kernel_lat, kernel_lon):
    # Interpolate onto kernel grid
    return field.interp(latitude=kernel_lat, longitude=kernel_lon, method='linear')

# Each regrid has 124 sets of nested lists, each one with 80 arrays of 144 points (for long and lat)
#print(regrid(ERA5_ds["tcwv"])[0][0].size)

regridded_fields = {}
for var in era5_files:
    if (var != "t") and (var != "q"): # Not t or q
        field = ERA5_single_layer[var].mean(dim='valid_time') # .sel(valid_time='2013-01-01T00:00:00.000000000')
        regridded_fields[var] = regrid(field, kernel_lat, kernel_lon)
        all_variables.append(var)
    elif (var == "t"): # t
        for pres in t_pres_to_use:
            field = ERA5_ta_pressures["t"].sel(pressure_level=pres).mean(dim='valid_time') # Change pressures
            regridded_fields["t"+pres] = regrid(field, kernel_lat, kernel_lon)
            all_variables.append("t"+pres)
    else:
        for pres in q_pres_to_use: # wv/q
            field = ERA5_q_pressures["q"].sel(pressure_level=pres).mean(dim='valid_time') # Change pressures
            regridded_fields["q"+pres] = regrid(field, kernel_lat, kernel_lon)
            all_variables.append("q"+pres)


# Flatten everything and build regression matrix
kernel_flat = kernel.values.flatten() # Flatten kernel into a 1D array
X_list = []
for var in all_variables:
    X_list.append(regridded_fields[var].values.flatten()) # Flatten ERA5 data arrays into X_list 1D array

X = np.stack(X_list, axis=1)  # shape (n_points, n_vars), [[var1 var2 ...], [], [], ... last point[]]

# Find valid (non-NaN) indices across all variables
mask = np.isfinite(kernel_flat)
for x in X_list:
    mask &= np.isfinite(x)

# Apply mask to both X and y
X_clean = X[mask, :]
y_clean = kernel_flat[mask]

# Run multiple linear regression
model = sm.OLS(y_clean, sm.add_constant(X_clean))
results = model.fit()
print("R-squared:", results.rsquared)
print(results.params) 
print(results.summary())


plt.figure(figsize=(7,5))
plt.scatter(X_list, kernel_flat, alpha=0.3, label='Data', s=10)
x_vals = np.linspace(X.min(), X.max(), 100)
y_vals = results.params[0] + results.params[1] * x_vals

plt.plot(x_vals, y_vals, color='red', label='OLS fit')
plt.xlabel('Skin temperature')
plt.ylabel('Surface Temperature Kernel (TOA)')
plt.title(f'Regression: Kernel vs. skt, R-squared: {results.rsquared:.4}')
plt.legend()
plt.grid(True)
# plt.tight_layout()
plt.savefig("SFC_temp_kernel_TOA_vs_ta_100hPa.png")
plt.show()


"""
# Load the kernels
kernel_ds = xr.open_dataset("ERA5_kernel_ts_TOA.nc")
kernel = kernel_ds["TOA_clr"][0]  # Replace with actual name, kernel for January ONLY, 2.5 deg resolution
kernel_flat = kernel.values.flatten() # Flatten kernel into a 1D array

ERA5_Ta = xr.open_dataset("2013_jan_Ta_pressures.nc")
pressure_arr = ["1000.0", "975.0", "950.0", "925.0", "900.0", "875.0", "850.0", "825.0", 
                "800.0", "775.0", "750.0", "700.0", "650.0", "600.0", "550.0", "500.0", "450.0", 
                "400.0", "350.0", "300.0", "250.0", "225.0", "200.0", "175.0", "150.0", "125.0", "100.0",
                  "70.0", "50.0", "30.0", "20.0", "10.0", "7.0", "5.0", "3.0", "2.0", "1.0"
]
era5_files = ["t"]


kernel_lat = np.linspace(-90, 90, 73)
kernel_lon = np.linspace(0, 357.5, 144)

def regrid(field, kernel_lat, kernel_lon):
    # Interpolate onto kernel grid
    return field.interp(latitude=kernel_lat, longitude=kernel_lon, method='linear')

# Each regrid has 124 sets of nested lists, each one with 80 arrays of 144 points (for long and lat)
#print(regrid(ERA5_ds["tcwv"])[0][0].size)

fig, axes = plt.subplots(len(pressure_arr), 1, figsize=(8, 3 * len(pressure_arr)), sharex=True)
counter = 0
for pressure in pressure_arr:  # Cleaner loop with enumerate
    regridded_fields = {}
    for var, path in ERA5_Ta.items():
        field = ERA5_Ta[var].sel(pressure_level= pressure).mean(dim='valid_time')
        regridded_fields[var] = regrid(field, kernel_lat, kernel_lon)


    X_list = []
    for var in era5_files:
        X_list.append(regridded_fields[var].values.flatten()) # Flatten ERA5 data arrays into X_list 1D array

    X = np.stack(X_list, axis=1)  # shape (n_points, n_vars), [[var1 var2 ...], [], [], ... last point[]]

    x_vals_flat = X[:, 0]  # 1D array of temps

    model = sm.OLS(kernel_flat, sm.add_constant(X))
    results = model.fit()

    with open("ols_results_TOA_ta.txt", "a") as f:
        # Write R-squared values
        f.write(f"Fit: Ta {pressure}hPa\n")
        f.write(f"R-squared: {results.rsquared:.4f}\n")

        # Write coefficients and p-values
        f.write("Coefficients with format: const, x1, x2, ...:\n")
        formatted = ", ".join([f"{coef:.4f}" for coef in results.params])
        f.write(formatted)
        f.write("\n")
        f.write("-------------------------------\n")

    ax = axes[counter]
    ax.scatter(x_vals_flat, kernel_flat, alpha=0.3, label='Data', s=10)
    
    x_range = np.linspace(x_vals_flat.min(), x_vals_flat.max(), 100)
    y_range = results.params[0] + results.params[1] * x_range
    ax.plot(x_range, y_range, color='red', label='OLS fit')

    ax.set_xlim(x_vals_flat.min(), x_vals_flat.max())
    ax.set_xlabel(f'Atmosphere Temperature at {pressure} hPa')
    ax.set_ylabel('Surface Temperature Kernel')
    ax.set_title(f'Pressure {pressure} hPa')
    counter+=1
plt.tight_layout()
plt.savefig("stacked_plots.png")
plt.close()
"""
