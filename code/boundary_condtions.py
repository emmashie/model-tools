from interpolate import interp_tools
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
import pandas as pd

base_path = '/global/cfs/cdirs/m4304/enuss/model-tools'
grid_nc = 'roms_grid.nc' 
forcing_datapath = '/global/cfs/cdirs/m4304/enuss/US_East_Coast/Forcing_Data'

grid = xr.open_dataset(os.path.join(base_path, 'output', grid_nc))

# grid range 
lat_min = grid.lat_rho.min().values
lat_max = grid.lat_rho.max().values
lon_min = grid.lon_rho.min().values
lon_max = grid.lon_rho.max().values

# set which boundaries need data 
boundaries = ['west', 'east', 'north', 'south'] 
flag = {'west': False, 'east': True, 'north': True, 'south': True}

# set time range 
start_time = pd.Timestamp("2024-01-01 00:00:00")
end_time = pd.Timestamp("2024-01-02 00:00:00")

# load boundary condition data
bc_data = xr.open_mfdataset(os.path.join(forcing_datapath, 'GLORYS_data', 'GLORYS_*.nc'))
# Subset bc_data to the specified time range
bc_data = bc_data.sel(time=slice(start_time, end_time))

# Get grid dimensions
eta_rho, xi_rho = grid.lat_rho.shape
eta_v, _ = grid.lat_v.shape
_, xi_u = grid.lat_u.shape
s_rho = len(grid.s_rho)
s_w = len(grid.s_w)
n_depth = len(bc_data.depth)

# Prepare GLORYS source coordinates
# Note: GLORYS depths are typically positive downward, ROMS depths are negative upward
glorys_depth = bc_data.depth.values  # Assuming positive depths
glorys_lat = bc_data.latitude.values
glorys_lon = bc_data.longitude.values

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

roms_latu_2d = grid.lat_u.values  
roms_lonu_2d = grid.lon_u.values  

roms_latv_2d = grid.lat_v.values  
roms_lonv_2d = grid.lon_v.values  

nt = len(bc_data['time'])
temp_interp = np.empty((nt, s_rho, eta_rho, xi_rho))
salt_interp = np.empty((nt, s_rho, eta_rho, xi_rho))
u_interp = np.empty((nt, s_rho, eta_rho, xi_u))
v_interp = np.empty((nt, s_rho, eta_v, xi_rho))
zeta_interp = np.empty((nt, eta_rho, xi_rho))
seconds_since_2000 = np.empty(nt)

ref_time = pd.Timestamp("2000-01-01 00:00:00")

for t in range(nt):
    temp_data = bc_data['thetao'][t,:,:,:].values 
    salt_data = bc_data['so'][t,:,:,:].values
    u_data = bc_data['uo'][t,:,:,:].values
    v_data = bc_data['vo'][t,:,:,:].values
    zeta_data = bc_data['zos'][t,0,:,:].values
    time = bc_data['time'][t].values
    current_time = pd.to_datetime(str(time))
    seconds_since_2000[t] = (current_time - ref_time).total_seconds()

    temp_interp[t] = interp_tools.interp3d(
        temp_data, glorys_lon_2d, glorys_lat_2d, glorys_depth, roms_lon_2d, roms_lat_2d, -z_rho, method='linear'
    )

    salt_interp[t] = interp_tools.interp3d(
        salt_data, glorys_lon_2d, glorys_lat_2d, glorys_depth, roms_lon_2d, roms_lat_2d, -z_rho, method='linear'
    )

    u_interp[t] = interp_tools.interp3d(
        u_data, glorys_lon_2d, glorys_lat_2d, glorys_depth, roms_lonu_2d, roms_latu_2d, -z_rho, method='linear'
    )

    v_interp[t] = interp_tools.interp3d(
        v_data, glorys_lon_2d, glorys_lat_2d, glorys_depth, roms_lonv_2d, roms_latv_2d, -z_rho, method='linear'
    )   

    zeta_interp[t] = interp_tools.interp2d(
        zeta_data, glorys_lon_2d, glorys_lat_2d, roms_lon_2d, roms_lat_2d, method='linear'
    )
    print("Interpolated time step %d/%d" % (t + 1, nt))


output_nc = os.path.join(base_path, 'output', 'clim_%s.nc' % str(bc_data.time[0].values)[:10])
ds = xr.Dataset(
    {
        'temp': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), temp_interp[np.newaxis, :, :, :]),
        'salt': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), salt_interp[np.newaxis, :, :, :]),
        'u': (('ocean_time', 's_rho', 'eta_rho', 'xi_u'), u_interp[np.newaxis, :, :, :] if 'u_interp' in locals() else np.full((1, s_rho, eta_rho, xi_u), np.nan)),
        'v': (('ocean_time', 's_rho', 'eta_v', 'xi_rho'), v_interp[np.newaxis, :, :, :] if 'v_interp' in locals() else np.full((1, s_rho, eta_v, xi_rho), np.nan)),
        'w': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), np.full((1, s_rho, eta_rho, xi_rho), np.nan)),
        'Cs_r': (('ocean_time', 's_rho'), grid.Cs_r.values[np.newaxis, :]),
        'Cs_w': (('ocean_time', 's_w'), grid.Cs_w.values[np.newaxis, :]),
        'zeta': (('ocean_time', 'eta_rho', 'xi_rho'), np.full((1, eta_rho, xi_rho), np.nan)),
        'ubar': (('ocean_time', 'eta_rho', 'xi_u'), np.full((1, eta_rho, xi_u), np.nan)),
        'vbar': (('ocean_time', 'eta_v', 'xi_rho'), np.full((1, eta_v, xi_rho), np.nan)),
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
        'ini_time': "2024-01-01 00:00:00",
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
ds.to_netcdf(output_nc, format='NETCDF4', engine='netcdf4')

# Extract boundary transects from interpolated arrays based on flags and grid extents

boundary_transects = {}

if flag['west']:
    # Western boundary: first xi index (xi=0), all eta
    boundary_transects['west'] = {
        'temp': temp_interp[:, :, :, 0],
        'salt': salt_interp[:, :, :, 0],
        'u': u_interp[:, :, :, 0],
        'v': v_interp[:, :, :, 0],
        'zeta': zeta_interp[:, :, 0],
        'lon': roms_lon_2d[:, 0],
        'lat': roms_lat_2d[:, 0]
    }

if flag['east']:
    # Eastern boundary: last xi index (xi=-1), all eta
    boundary_transects['east'] = {
        'temp': temp_interp[:, :, :, -1],
        'salt': salt_interp[:, :, :, -1],
        'u': u_interp[:, :, :, -1],
        'v': v_interp[:, :, :, -1],
        'zeta': zeta_interp[:, :, -1],
        'lon': roms_lon_2d[:, -1],
        'lat': roms_lat_2d[:, -1]
    }

if flag['south']:
    # Southern boundary: first eta index (eta=0), all xi
    boundary_transects['south'] = {
        'temp': temp_interp[:, :, 0, :],
        'salt': salt_interp[:, :, 0, :],
        'u': u_interp[:, :, 0, :],
        'v': v_interp[:, :, 0, :],
        'zeta': zeta_interp[:, 0, :],
        'lon': roms_lon_2d[0, :],
        'lat': roms_lat_2d[0, :]
    }

if flag['north']:
    # Northern boundary: last eta index (eta=-1), all xi
    boundary_transects['north'] = {
        'temp': temp_interp[:, :, -1, :],
        'salt': salt_interp[:, :, -1, :],
        'u': u_interp[:, :, -1, :],
        'v': v_interp[:, :, -1, :],
        'zeta': zeta_interp[:, -1, :],
        'lon': roms_lon_2d[-1, :],
        'lat': roms_lat_2d[-1, :]
    }

print("Extracted boundary transects for:", list(boundary_transects.keys()))
# === Write boundary condition file using xarray ===

# Prepare time and output path
bry_time = seconds_since_2000 / 86400.0  # days since 2000-01-01
output_bry = os.path.join(base_path, 'output', f'boundary_forcing_{str(start_time)[:10]}.nc')

# Get dimensions
nt_bry = len(bry_time)
s_rho_dim = s_rho
xi_u_dim = xi_u
xi_rho_dim = xi_rho
eta_rho_dim = eta_rho
eta_v_dim = eta_v

# Helper to get boundary or fill with NaN if not present
def get_bry(var, b, shape):
    if b in boundary_transects:
        return boundary_transects[b][var]
    else:
        return np.full(shape, np.nan)

ds_bry = xr.Dataset(
    {
        # South boundary
        'u_south': (['bry_time', 's_rho', 'xi_u'], get_bry('u', 'south', (nt_bry, s_rho_dim, xi_u_dim))),
        'v_south': (['bry_time', 's_rho', 'xi_rho'], get_bry('v', 'south', (nt_bry, s_rho_dim, xi_rho_dim))),
        'temp_south': (['bry_time', 's_rho', 'xi_rho'], get_bry('temp', 'south', (nt_bry, s_rho_dim, xi_rho_dim))),
        'salt_south': (['bry_time', 's_rho', 'xi_rho'], get_bry('salt', 'south', (nt_bry, s_rho_dim, xi_rho_dim))),
        'zeta_south': (['bry_time', 'xi_rho'], get_bry('zeta', 'south', (nt_bry, xi_rho_dim))),
        'ubar_south': (['bry_time', 'xi_u'], np.full((nt_bry, xi_u_dim), np.nan)),
        'vbar_south': (['bry_time', 'xi_rho'], np.full((nt_bry, xi_rho_dim), np.nan)),

        # East boundary
        'u_east': (['bry_time', 's_rho', 'eta_rho'], get_bry('u', 'east', (nt_bry, s_rho_dim, eta_rho_dim))),
        'v_east': (['bry_time', 's_rho', 'eta_v'], get_bry('v', 'east', (nt_bry, s_rho_dim, eta_v_dim))),
        'temp_east': (['bry_time', 's_rho', 'eta_rho'], get_bry('temp', 'east', (nt_bry, s_rho_dim, eta_rho_dim))),
        'salt_east': (['bry_time', 's_rho', 'eta_rho'], get_bry('salt', 'east', (nt_bry, s_rho_dim, eta_rho_dim))),
        'zeta_east': (['bry_time', 'eta_rho'], get_bry('zeta', 'east', (nt_bry, eta_rho_dim))),
        'ubar_east': (['bry_time', 'eta_rho'], np.full((nt_bry, eta_rho_dim), np.nan)),
        'vbar_east': (['bry_time', 'eta_v'], np.full((nt_bry, eta_v_dim), np.nan)),

        # North boundary
        'u_north': (['bry_time', 's_rho', 'xi_u'], get_bry('u', 'north', (nt_bry, s_rho_dim, xi_u_dim))),
        'v_north': (['bry_time', 's_rho', 'xi_rho'], get_bry('v', 'north', (nt_bry, s_rho_dim, xi_rho_dim))),
        'temp_north': (['bry_time', 's_rho', 'xi_rho'], get_bry('temp', 'north', (nt_bry, s_rho_dim, xi_rho_dim))),
        'salt_north': (['bry_time', 's_rho', 'xi_rho'], get_bry('salt', 'north', (nt_bry, s_rho_dim, xi_rho_dim))),
        'zeta_north': (['bry_time', 'xi_rho'], get_bry('zeta', 'north', (nt_bry, xi_rho_dim))),
        'ubar_north': (['bry_time', 'xi_u'], np.full((nt_bry, xi_u_dim), np.nan)),
        'vbar_north': (['bry_time', 'xi_rho'], np.full((nt_bry, xi_rho_dim), np.nan)),

        # West boundary (not used in your example, but can be added similarly)
        # 'u_west': ...
        # 'v_west': ...
        # 'temp_west': ...
        # 'salt_west': ...
        # 'zeta_west': ...
        # 'ubar_west': ...
        # 'vbar_west': ...

    },
    coords={
        'bry_time': ('bry_time', bry_time),
        's_rho': ('s_rho', np.arange(s_rho_dim)),
        'xi_u': ('xi_u', np.arange(xi_u_dim)),
        'xi_rho': ('xi_rho', np.arange(xi_rho_dim)),
        'eta_rho': ('eta_rho', np.arange(eta_rho_dim)),
        'eta_v': ('eta_v', np.arange(eta_v_dim)),
    },
    attrs={
        'title': 'ROMS boundary forcing file created by model-tools',
        'start_time': str(start_time),
        'end_time': str(end_time),
        'source': 'GLORYS',
        'model_reference_date': '2000-01-01 00:00:00',
        'theta_s': float(grid.theta_s),
        'theta_b': float(grid.theta_b),
        'hc': float(grid.hc),
    }
)


# Set variable attributes (without _FillValue)
ds_bry['u_south'].attrs = dict(long_name="southern boundary u-flux component", units="m/s", coordinates="abs_time")
ds_bry['v_south'].attrs = dict(long_name="southern boundary v-flux component", units="m/s", coordinates="abs_time")
ds_bry['temp_south'].attrs = dict(long_name="southern boundary potential temperature", units="degrees Celsius", coordinates="abs_time")
ds_bry['salt_south'].attrs = dict(long_name="southern boundary salinity", units="PSU", coordinates="abs_time")
ds_bry['u_east'].attrs = dict(long_name="eastern boundary u-flux component", units="m/s", coordinates="abs_time")
ds_bry['v_east'].attrs = dict(long_name="eastern boundary v-flux component", units="m/s", coordinates="abs_time")
ds_bry['temp_east'].attrs = dict(long_name="eastern boundary potential temperature", units="degrees Celsius", coordinates="abs_time")
ds_bry['salt_east'].attrs = dict(long_name="eastern boundary salinity", units="PSU", coordinates="abs_time")
ds_bry['u_north'].attrs = dict(long_name="northern boundary u-flux component", units="m/s", coordinates="abs_time")
ds_bry['v_north'].attrs = dict(long_name="northern boundary v-flux component", units="m/s", coordinates="abs_time")
ds_bry['temp_north'].attrs = dict(long_name="northern boundary potential temperature", units="degrees Celsius", coordinates="abs_time")
ds_bry['salt_north'].attrs = dict(long_name="northern boundary salinity", units="PSU", coordinates="abs_time")
ds_bry['zeta_north'].attrs = dict(long_name="northern boundary sea surface height", units="m", coordinates="abs_time")
ds_bry['zeta_south'].attrs = dict(long_name="southern boundary sea surface height", units="m", coordinates="abs_time")
ds_bry['zeta_east'].attrs = dict(long_name="eastern boundary sea surface height", units="m", coordinates="abs_time")
# ...add more variable attributes as needed...

# Write to NetCDF
ds_bry.to_netcdf(output_bry, format='NETCDF4', encoding={var: {'_FillValue': np.nan} for var in ds_bry.data_vars})
print(f"Boundary condition NetCDF file written: {output_bry}")

