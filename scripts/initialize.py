from interpolate import interp_tools 
from conversions import convert_tools
from grid import grid_tools
import xarray as xr
import numpy as np
import cmocean
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import hvplot.xarray
import geoviews as gv
import os 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Set paths to data files and grid file 
base_path = '/global/cfs/cdirs/m4304/enuss/model-tools/'
grid_nc = 'roms_grid_1km_smoothed.nc' 
forcing_datapath = '/global/cfs/cdirs/m4304/enuss/US_East_Coast/Forcing_Data'

# Load the grid 
grid = xr.open_dataset(os.path.join(base_path, 'output', grid_nc))

# Load initialization data 
init_data = xr.open_dataset(os.path.join(forcing_datapath, 'GLORYS_data', 'GLORYS_January.nc'))

# Define the new depth value and fill values for each variable
new_depth = 10000.0
fill_values = {
    'zos': np.nan,      # sea surface height
    'uo': 0.0,       # eastward velocity  
    'vo': 0.0,       # northward velocity
    'so': 35.0,      # salinity (typical deep ocean value)
    'thetao': 1.0    # potential temperature (typical deep ocean value)
}

# Create new dataset with the additional depth layer
new_data_vars = {}
for var_name, fill_val in fill_values.items():
    # Create array with same shape as original but with depth dimension = 1
    shape = (len(init_data.time), 1, len(init_data.latitude), len(init_data.longitude))
    data = np.full(shape, fill_val, dtype=np.float32)
    
    new_data_vars[var_name] = xr.DataArray(
        data,
        coords={
            'time': init_data.time,
            'depth': [new_depth],
            'latitude': init_data.latitude,
            'longitude': init_data.longitude
        },
        dims=['time', 'depth', 'latitude', 'longitude']
    )

# Create new dataset and concatenate with original
new_layer = xr.Dataset(new_data_vars, attrs=init_data.attrs)
init_data = xr.concat([init_data, new_layer], dim='depth')
 

# Set variables names used in initialization data 
salt_var = 'so'
temp_var = 'thetao' 
u_var = 'uo'
v_var = 'vo'
zeta_var = 'zos'
depth = 'depth'
lat_var = 'latitude'
lon_var = 'longitude'


# === CREATE INITIAL CONDITIONS FILE === 
# Set the initialization time
init_time = np.datetime64('2024-01-01T00:00:00')
time_idx = np.argmin(np.abs(init_data.time.values - init_time))

# Compute time in seconds since 2000-01-01 00:00:00
ref_time = np.datetime64('2000-01-01T00:00:00')
seconds_since_2000 = (init_time - ref_time) / np.timedelta64(1, 's')
print(f"Seconds since 2000-01-01 00:00:00: {seconds_since_2000:.6e}")

# Select first time step from GLORYS data
glorys_time_step = init_data.isel(time=time_idx)

# Get grid dimensions 
eta_rho, xi_rho, eta_v, xi_u, s_rho, s_w = grid_tools.get_grid_dims(grid)

# Get initialization data depth
n_depth = len(init_data[depth])

# Prepare GLORYS source coordinates
# Note: GLORYS depths are typically positive downward, ROMS depths are negative upward
glorys_depth = init_data[depth].values  # Assuming positive depths
glorys_lat = init_data[lat_var].values
glorys_lon = init_data[lon_var].values

# Create 2D lat/lon grids for GLORYS data
glorys_lon_2d, glorys_lat_2d = np.meshgrid(glorys_lon, glorys_lat, indexing='xy')

# Create 3D depth array for GLORYS (depth varies with depth dimension, constant across lat/lon)
# Shape: (n_lat, n_lon, n_depth)
glorys_depth_3d = np.broadcast_to(
    glorys_depth[np.newaxis, np.newaxis, :], 
    (len(glorys_lat), len(glorys_lon), len(glorys_depth))
)

# Compute ROMS vertical coordinates
z_rho = grid_tools.compute_z(
    grid.sigma_r.values, 
    grid.hc, 
    grid.Cs_r.values, 
    grid.h.values, 
    np.zeros((eta_rho, xi_rho, 1))
)
z_rho = np.squeeze(z_rho)  # Remove singleton dimension
z_rho = np.transpose(z_rho, (1, 2, 0))  # Shape: (eta_rho, xi_rho, s_rho)

# Convert ROMS depths to positive values if they're negative (for consistent interpolation)
roms_depth_3d = np.abs(z_rho)  # Shape: (eta_rho, xi_rho, s_rho)

# Get ROMS target coordinates
roms_lat_2d = grid.lat_rho.values  # Shape: (eta_rho, xi_rho)
roms_lon_2d = grid.lon_rho.values  # Shape: (eta_rho, xi_rho)

# === TEMPERATURE INTERPOLATION ===
# Prepare GLORYS temperature data
# Original shape is typically (depth, lat, lon), need to transpose to (nz_src, ny_src, nx_src)
temp_data = glorys_time_step.thetao.values  # Shape: (n_depth, n_lat, n_lon)
if temp_data.shape[0] != len(glorys_depth):
    temp_data = np.transpose(temp_data, (2, 0, 1))

print("Interpolating temperature with interp3d...")
print(f"Source data shape: {temp_data.shape}")
print(f"Source lat shape: {glorys_lat_2d.shape}")
print(f"Source lon shape: {glorys_lon_2d.shape}")
print(f"Source depth shape: {glorys_depth.shape}")
print(f"Target lat shape: {roms_lat_2d.shape}")
print(f"Target lon shape: {roms_lon_2d.shape}")
print(f"Target depth shape: {roms_depth_3d.shape}")

# Output: (nz_tgt, eta_rho, xi_rho)
temp_interp = interp_tools.interp3d(
    temp_data, glorys_lon_2d, glorys_lat_2d, glorys_depth, roms_lon_2d, roms_lat_2d, -z_rho, method='linear'
)

# Find negative temps and set to minimum temperature 
zind, yind, xind = np.where(temp_interp<0)
tmin = 0.1
temp_interp[zind, yind, xind] = tmin  # Set negative values to minimum temperature

# Mask temp_interp with grid.mask_rho (1=water, 0=land)
mask = grid.mask_rho.values.astype(bool)
temp_interp = np.where(mask, temp_interp, np.nan)
fill_value = 5
temp_interp = np.where(np.isnan(temp_interp), fill_value, temp_interp)

print(f"Interpolated temperature shape: {temp_interp.shape}")


# === SALINITY INTERPOLATION ===
sal_data = glorys_time_step.so.values  # Shape: (n_depth, n_lat, n_lon)
if sal_data.shape[0] != len(glorys_depth):
    sal_data = np.transpose(sal_data, (2, 0, 1))

print("Interpolating salinity with interp3d...")
sal_interp = interp_tools.interp3d(
    sal_data, glorys_lon_2d, glorys_lat_2d, glorys_depth, roms_lon_2d, roms_lat_2d, -z_rho, method='linear'
)

# Mask sal_interp with grid.mask_rho (1=water, 0=land)
#sal_interp = np.where(mask, sal_interp, np.nan)
fill_value = 32.0  # Typical ocean salinity
sal_interp = np.where(np.isnan(sal_interp), fill_value, sal_interp)

print(f"Interpolated salinity shape: {sal_interp.shape}")


# === U AND V VELOCITY INTERPOLATION === 
u_data = glorys_time_step.uo.values
v_data = glorys_time_step.vo.values
if u_data.shape[0] != len(glorys_depth):
    u_data = np.transpose(u_data, (2, 0, 1))
if v_data.shape[0] != len(glorys_depth):
    v_data = np.transpose(v_data, (2, 0, 1))

print("Interpolating u and v velocities with interp3d...")

# Get ROMS target coordinates for u and v grids
roms_latu_2d = grid.lat_u.values  
roms_lonu_2d = grid.lon_u.values  

roms_latv_2d = grid.lat_v.values  
roms_lonv_2d = grid.lon_v.values  

u_interp = interp_tools.interp3d(
    u_data, glorys_lon_2d, glorys_lat_2d, glorys_depth, roms_lonu_2d, roms_latu_2d, -z_rho, method='linear'
)
v_interp = interp_tools.interp3d(
    v_data, glorys_lon_2d, glorys_lat_2d, glorys_depth, roms_lonv_2d, roms_latv_2d, -z_rho, method='linear'
)

# Mask with grid
mask_u = grid.mask_u.values.astype(bool)
mask_v = grid.mask_v.values.astype(bool)
#u_interp = np.where(mask_u, u_interp, np.nan)
#v_interp = np.where(mask_v, v_interp, np.nan)
fill_value = 0
u_interp = np.where(np.isnan(u_interp), fill_value, u_interp)
v_interp = np.where(np.isnan(v_interp), fill_value, v_interp)

print(f"Interpolated u velocity shape: {u_interp.shape}")
print(f"Interpolated v velocity shape: {v_interp.shape}")

# Reshape z_rho from (ny, nx, nt) to (nt, ny, nx)
z_rho = np.transpose(z_rho, (2, 0, 1))
ubar, vbar = convert_tools.compute_uvbar(u_interp, v_interp, z_rho)

w = convert_tools.compute_w(u_interp, v_interp, grid.pm.values, grid.pn.values, z_rho)

# === ZETA INTERPOLATION ===
zeta_data = glorys_time_step[zeta_var].values 
# select surface level  
zeta_data = zeta_data[0, :, :] 

zeta_interp = interp_tools.interp2d(
    zeta_data, glorys_lon_2d, glorys_lat_2d, roms_lon_2d, roms_lat_2d, method='linear'
)
fill_value = 0 
zeta_interp = np.where(np.isnan(zeta_interp), fill_value, zeta_interp)

output_nc = os.path.join(base_path, 'output', 'initial_conditions_1km.nc')
ds = xr.Dataset(
    {
        'temp': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), temp_interp[np.newaxis, :, :, :]),
        'salt': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), sal_interp[np.newaxis, :, :, :]),
        'u': (('ocean_time', 's_rho', 'eta_rho', 'xi_u'), u_interp[np.newaxis, :, :, :]),
        'v': (('ocean_time', 's_rho', 'eta_v', 'xi_rho'), v_interp[np.newaxis, :, :, :]),
        'w': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), w[np.newaxis, :, :, :]),
        'Cs_r': (('ocean_time', 's_rho'), grid.Cs_r.values[np.newaxis, :]),
        'Cs_w': (('ocean_time', 's_w'), grid.Cs_w.values[np.newaxis, :]),
        'zeta': (('ocean_time', 'eta_rho', 'xi_rho'), zeta_interp[np.newaxis, :, :]),
        'ubar': (('ocean_time', 'eta_rho', 'xi_u'), ubar[np.newaxis, :, :]),
        'vbar': (('ocean_time', 'eta_v', 'xi_rho'), vbar[np.newaxis, :, :]),
    },
    coords={
        'ocean_time': [seconds_since_2000 / 86400.0],
        's_rho': np.arange(s_rho),
        'eta_rho': np.arange(eta_rho),
        'xi_rho': np.arange(xi_rho),
        'xi_u': np.arange(xi_u),
        'eta_v': np.arange(eta_v),
        's_w': np.arange(s_w),
    },
    attrs={
        'title': "ROMS initial conditions file created by model-tools",
        'roms_tools_version': "2.4.0",
        'ini_time': str(init_time),
        'model_reference_date': "2000-01-01 00:00:00",
        'source': "GLORYS",
        'theta_s': grid.theta_s,
        'theta_b': grid.theta_b,
        'hc': grid.hc,
    }
)
# Variable attributes
ds['temp'].attrs = dict(long_name="potential temperature", units="degrees Celsius", coordinates="ocean_time")
ds['salt'].attrs = dict(long_name="salinity", units="PSU", coordinates="ocean_time")
ds['u'].attrs = dict(long_name="u-flux component", units="m/s", coordinates="ocean_time")
ds['v'].attrs = dict(long_name="v-flux component", units="m/s", coordinates="ocean_time")
ds['w'].attrs = dict(long_name="w-flux component", units="m/s", coordinates="ocean_time")
ds['Cs_r'].attrs = dict(long_name="Vertical stretching function at rho-points", units="nondimensional", coordinates="ocean_time")
ds['Cs_w'].attrs = dict(long_name="Vertical stretching function at w-points", units="nondimensional", coordinates="ocean_time")
ds['zeta'].attrs = dict(long_name="sea surface height", units="m", coordinates="ocean_time")
ds['ubar'].attrs = dict(long_name="vertically integrated u-flux component", units="m/s", coordinates="ocean_time")
ds['vbar'].attrs = dict(long_name="vertically integrated v-flux component", units="m/s", coordinates="ocean_time")
ds['ocean_time'].attrs = dict(long_name='relative time: days since 2000-01-01 00:00:00', units='days')
ds.to_netcdf(output_nc, format='NETCDF4', engine='netcdf4')