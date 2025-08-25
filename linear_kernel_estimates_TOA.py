import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Each linear regression was entered manually
def q1000(x1):
    return 33.6557 * x1 -1.5194 

def t900(x1):
    return 0.0128 * x1 - 4.8273

def q1000_skt(x1, x2):
    return 34.761191 * x1 -0.0003505381 * x2 -1.4294306

def q50_q70(x1,x2):
    return 1.419*10**5 * x1 -7.085*10**5 * x2 + 0.1144

def q750_q1000(x1,x2):
    return 12.2707 * x1 + 28.5448 * x2 -1.5207

def t100_t300(x1,x2):
    return -0.0070 * x1 + 0.0117 * x2 -2.5077

def t100_t300_q950(x1,x2,x3):
    return -0.0053*x1+0.0070*x2+13.1778*x3-1.8790

def t100_t300_q70_q950(x1,x2,x3,x4):
    return -0.0011 * x1 -0.0054 * x2 -7.584*10**5 * x3 -1.3825 * x4 + 2.1117

def t100_t300_q70_q950_skt(x1,x2,x3,x4,x5):
    return -0.0009 * x1 -0.0071 * x2 -7.739*10**5 * x3 -2.7803 * x4 + 0.0011 * x5 + 2.2215

def q750_q850_q950_q1000_skt(x1,x2,x3,x4,x5):
    return -1.6299 * x1 + 19.6877 * x2 + 8.8110*x3+13.5930*x4-2.435*10**-5 * x5-1.5149

def q750_q1000_skt_t2m(x1,x2,x3,x4):
    return 12.5982*x1+26.5006*x2-0.0105*x3+0.0116*x4-1.7983

def q750_q850_q950_q1000_skt_t2m(x1,x2,x3,x4,x5,x6):
    return -1.2875*x1+19.6617*x2+11.0142*x3+9.1468*x4-0.0108*x5+0.0120*x6-1.8262

def t1000_q750_q1000_skt_t2m(x1,x2,x3,x4,x5):
    return 0.0085*x1+6.0771*x2+28.0058*x3-0.0004*x4-0.0056*x5-2.2230

def q750_q1000_skt_t2m_tcwv(x1,x2,x3,x4,x5):
    return 15.9415*x1+29.5983*x2-0.0103*x3+0.0113*x4-0.0016*x5-1.7778

def t1000_q750_q1000_skt_t2m_tcwv(x1,x2,x3,x4,x5,x6):
    return 0.0092*x1-3.8719*x2+19.3965*x3-0.0002*x4-0.0061*x5+0.0045*x6-2.3151

def q750_q1000_skt_t2m_tcwv_sp_tco3(x1,x2,x3,x4,x5,x6,x7):
    return 7.9629*x1+21.1342*x2+0.0024*x3-0.0055*x4+0.0021*x5+7.897*10**-7*x6-82.3537*x7-0.1883

def t300_t900_q70_q750_q1000_sp(x1,x2,x3,x4,x5,x6):
    return -0.0106 * x1 + 0.0031 * x2 -7.767*10**5 * x3 -10.9876 * x4 + 6.4420 * x5 -1.543*10**-6 * x6 + 2.3825

def t70_t100_t125_t300_t350_t400_t450_t900_q850_q950_q1000_skt(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12):
    return -0.0194*x1-0.0382*x2+0.0637*x3+0.0241*x4-0.0667*x5+0.0703*x6-0.0256*x7+0.0058*x8+7.2998*x9-5.0880*x10+19.7677*x11-0.0044*x12-3.7497

def t70_t100_t125_t300_t350_t400_t450_t900_q50_q70_q850_q950_q1000_skt(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12, x13, x14):
    return -0.0143*x1-0.0032*x2+0.0196*x3+0.0094*x4-0.0248*x5-0.0170*x6+0.0249*x7+0.0026*x8-1.099*10**5*x9-6.689*10**5*x10+2.2318*x11-9.8851*x12+10.1632*x13-0.0006*x14+1.3153

def t70_t100_t125_t300_t350_t400_t450_t900_q850_q950_q1000_skt_t2m_tcwv_tco3(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12, x13, x14, x15):
    return -0.0250*x1-0.0308*x2+0.0542*x3+0.0013*x4-0.0093*x5+0.0135*x6-0.0267*x7-0.0002*x8-2.8604*x9+3.9178*x10+0.4036*x11-0.0100*x12+0.0122*x13+0.0073*x14-157.3508*x15+4.4601

def t70_t100_t125_t300_t350_t400_t450_t900_q50_q70_q850_q950_q1000_skt_t2m_tcwv_tco3(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12, x13, x14, x15, x16, x17):
    return -0.0195*x1-0.0038*x2+0.0181*x3-0.0075*x4+0.0162*x5-0.0559*x6+0.0205*x7-0.0036*x8-5.469*10**5*x9-4.381*10**5*x10-0.0830*x11-0.7036*x12-2.4013*x13-0.0042*x14+0.0087*x15+0.0029*x16-132.0176*x17+9.3365

def q750_q1000_t100_skt_sp_tcwv(x1,x2,x3,x4,x5,x6):
    return -16.0478*x1+0.7576*x2-0.0077*x3+0.0017*x4-6.887*10**-6*x5+0.0086*x6+0.4043

era5_files = ["q", "t", "skt", "sp", "tcwv"] # <--------------------------------------- 
t_pres_to_use = ["100"] # <--------------------------------------- 
q_pres_to_use = ["750", "1000"]
all_variables = []
ERA5_single_layer = xr.open_dataset("2013_jan_single_layer.nc")
ERA5_ta_pressures = xr.open_dataset("2013_jan_Ta_pressures.nc")
ERA5_q_pressures = xr.open_dataset("2013_jan_wv_pres.nc")

kernel_lat = np.linspace(-90, 90, 73)
kernel_lon = np.linspace(0, 357.5, 144)

def regrid(field, kernel_lat, kernel_lon):
    # Interpolate onto kernel grid
    return field.interp(latitude=kernel_lat, longitude=kernel_lon, method='linear')

regridded_fields = {}
for var in era5_files:
    if (var != "t") and (var != "q"): # Not t or q
        field = ERA5_single_layer[var].mean(dim='valid_time') # Maybe take off .mean()
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

X_list = []
for var in all_variables:
    X_list.append(regridded_fields[var].values.flatten()) # Flatten ERA5 data arrays into X_list 1D array

X = np.stack(X_list, axis=1)  # shape (n_points, n_vars), [[var1 var2 ...], [], [], ... last point[]]


Y_est = q750_q1000_t100_skt_sp_tcwv(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]) # <------------------------------ 
grid = Y_est.reshape((73, 144))

plt.figure(figsize=(10, 5))
plt.imshow(grid, origin='lower', cmap='viridis', extent=[-180, 180, -90, 90], aspect='auto', vmin=-2.2, vmax=-0.6)
plt.colorbar(label='Clear-sky SFC temperature kernel Estimate [W/m²/K]')
plt.title('Surface Temperature Kernel Estimate (Clear-sky, TOA). R-squared: 0.691') # <-------------------------------------------------------
plt.grid(False)
plt.savefig("t70_t100_t125_t300_t350_t400_t450_t900_q50_q70_q850_q950_q1000_skt_t2m_tcwv_tco3_TOA_est.png") # <-------------------------------------------------------------------
