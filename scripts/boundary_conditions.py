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

# load boundary condition data
bc_data = xr.open_mfdataset(os.path.join(forcing_datapath, 'GLORYS_data', 'GLORYS_*.nc'))

# set time range 
start_time = bc_data['time'][24].values
end_time = pd.Timestamp("2024-02-01 00:00:00")
ref_time = pd.Timestamp("2000-01-01 00:00:00")

# Subset bc_data to the specified time range
bc_data = bc_data.sel(time=slice(start_time, end_time))

# Set variable names
salt_var = 'so'
temp_var = 'thetao' 
u_var = 'uo'
v_var = 'vo'
zeta_var = 'zos'
depth = 'depth'
lat_var = 'latitude'
lon_var = 'longitude'

# Get valid time indices 
valid_time_indices = np.where(np.isfinite(bc_data['uo'][:,0,-1,-1].values))[0]
#valid_time_indices = [t for t in range(len(bc_data['time'])) if np.isfinite(bc_data['uo'][t, :, :, :].values).any()]
print(f"Indices of time steps with any finite 'uo' values: {valid_time_indices}")

# Get grid dimensions 
eta_rho, xi_rho, eta_v, xi_u, s_rho, s_w = grid_tools.get_grid_dims(grid)

# Get data depth
n_depth = len(bc_data[depth])

# Prepare GLORYS source coordinates
# Note: GLORYS depths are typically positive downward, ROMS depths are negative upward
glorys_depth = bc_data[depth].values  # Assuming positive depths
glorys_lat = bc_data[lat_var].values
glorys_lon = bc_data[lon_var].values

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

nt = len(valid_time_indices)
temp_interp = np.empty((nt, s_rho, eta_rho, xi_rho))
salt_interp = np.empty((nt, s_rho, eta_rho, xi_rho))
u_interp = np.empty((nt, s_rho, eta_rho, xi_u))
v_interp = np.empty((nt, s_rho, eta_v, xi_rho))
zeta_interp = np.empty((nt, eta_rho, xi_rho))
seconds_since_2000 = np.empty(nt)

for t in range(nt):
    temp_data = bc_data['thetao'][valid_time_indices[t],:,:].values
    salt_data = bc_data['so'][valid_time_indices[t],:,:].values
    u_data = bc_data['uo'][valid_time_indices[t],:,:].values
    v_data = bc_data['vo'][valid_time_indices[t],:,:].values
    zeta_data = bc_data['zos'][valid_time_indices[t],0,:,:].values
    time = bc_data['time'][valid_time_indices[t]].values
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

# fill nans with values 
fill_value = 0
zeta_interp = np.where(np.isnan(zeta_interp), fill_value, zeta_interp)
fill_value = 5
temp_interp = np.where(np.isnan(temp_interp), fill_value, temp_interp)
fill_value = 32.0   
sal_interp = np.where(np.isnan(salt_interp), fill_value, salt_interp)
fill_value = 0
u_interp = np.where(np.isnan(u_interp), fill_value, u_interp)
v_interp = np.where(np.isnan(v_interp), fill_value, v_interp)

## calculate ubar, vbar, and w  
z_rho = np.transpose(z_rho, (2, 0, 1))
ubar = np.empty((nt, eta_rho, xi_u))
vbar = np.empty((nt, eta_v, xi_rho))
for t in range(nt):
    ubar[t], vbar[t] = convert_tools.compute_uvbar(u_interp[t], v_interp[t], z_rho)

w = np.empty((nt, s_rho, eta_rho, xi_rho))
for t in range(nt):
    w[t] = convert_tools.compute_w(u_interp[t], v_interp[t], grid.pm.values, grid.pn.values, z_rho)

output_nc = os.path.join(base_path, 'output', 'clim_%s.nc' % str(bc_data.time[0].values)[:10])
ds = xr.Dataset(
    {
        'temp': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), temp_interp),
        'salt': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), salt_interp),
        'u': (('ocean_time', 's_rho', 'eta_rho', 'xi_u'), u_interp),
        'v': (('ocean_time', 's_rho', 'eta_v', 'xi_rho'), v_interp),
        'w': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), w),
        'Cs_r': (('s_rho'), grid.Cs_r.values),
        'Cs_w': (('s_w'), grid.Cs_w.values),
        'zeta': (('ocean_time', 'eta_rho', 'xi_rho'), zeta_interp),
        'ubar': (('ocean_time', 'eta_rho', 'xi_u'), ubar),
        'vbar': (('ocean_time', 'eta_v', 'xi_rho'), vbar),
    },
    coords={
        'ocean_time': seconds_since_2000 / 86400.0,
        's_rho': np.arange(s_rho),
        'eta_rho': np.arange(eta_rho),
        'xi_rho': np.arange(xi_rho),
        'xi_u': np.arange(xi_u),
        'eta_v': np.arange(eta_v),
        's_w': np.arange(s_w),
    },
    attrs={
        'title': "ROMS climatology file created by model-tools",
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
        'ubar': ubar[:, :, 0],
        'vbar': vbar[:, :, 0],
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
        'ubar': ubar[:, :, -1],
        'vbar': vbar[:, :, -1],
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
        'ubar': ubar[:, 0, :],
        'vbar': vbar[:, 0, :],
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
        'ubar': ubar[:, -1, :],
        'vbar': vbar[:, -1, :],
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
        'ubar_south': (['bry_time', 'xi_u'], get_bry('ubar', 'south', (nt_bry, xi_u_dim))),
        'vbar_south': (['bry_time', 'xi_rho'], get_bry('vbar', 'south', (nt_bry, xi_rho_dim))),

        # East boundary
        'u_east': (['bry_time', 's_rho', 'eta_rho'], get_bry('u', 'east', (nt_bry, s_rho_dim, eta_rho_dim))),
        'v_east': (['bry_time', 's_rho', 'eta_v'], get_bry('v', 'east', (nt_bry, s_rho_dim, eta_v_dim))),
        'temp_east': (['bry_time', 's_rho', 'eta_rho'], get_bry('temp', 'east', (nt_bry, s_rho_dim, eta_rho_dim))),
        'salt_east': (['bry_time', 's_rho', 'eta_rho'], get_bry('salt', 'east', (nt_bry, s_rho_dim, eta_rho_dim))),
        'zeta_east': (['bry_time', 'eta_rho'], get_bry('zeta', 'east', (nt_bry, eta_rho_dim))),
        'ubar_east': (['bry_time', 'eta_rho'], get_bry('ubar', 'east', (nt_bry, eta_rho_dim))),
        'vbar_east': (['bry_time', 'eta_v'], get_bry('vbar', 'east', (nt_bry, eta_v_dim))),

        # North boundary
        'u_north': (['bry_time', 's_rho', 'xi_u'], get_bry('u', 'north', (nt_bry, s_rho_dim, xi_u_dim))),
        'v_north': (['bry_time', 's_rho', 'xi_rho'], get_bry('v', 'north', (nt_bry, s_rho_dim, xi_rho_dim))),
        'temp_north': (['bry_time', 's_rho', 'xi_rho'], get_bry('temp', 'north', (nt_bry, s_rho_dim, xi_rho_dim))),
        'salt_north': (['bry_time', 's_rho', 'xi_rho'], get_bry('salt', 'north', (nt_bry, s_rho_dim, xi_rho_dim))),
        'zeta_north': (['bry_time', 'xi_rho'], get_bry('zeta', 'north', (nt_bry, xi_rho_dim))),
        'ubar_north': (['bry_time', 'xi_u'], get_bry('ubar', 'north', (nt_bry, xi_u_dim))),
        'vbar_north': (['bry_time', 'xi_rho'], get_bry('vbar', 'north', (nt_bry, xi_rho_dim))),

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
ds_bry['u_south'].attrs = dict(long_name="southern boundary u-flux component", units="m/s", coordinates="bry_time")
ds_bry['v_south'].attrs = dict(long_name="southern boundary v-flux component", units="m/s", coordinates="bry_time")
ds_bry['temp_south'].attrs = dict(long_name="southern boundary potential temperature", units="degrees Celsius", coordinates="bry_time")
ds_bry['salt_south'].attrs = dict(long_name="southern boundary salinity", units="PSU", coordinates="bry_time")
ds_bry['u_east'].attrs = dict(long_name="eastern boundary u-flux component", units="m/s", coordinates="bry_time")
ds_bry['v_east'].attrs = dict(long_name="eastern boundary v-flux component", units="m/s", coordinates="bry_time")
ds_bry['temp_east'].attrs = dict(long_name="eastern boundary potential temperature", units="degrees Celsius", coordinates="bry_time")
ds_bry['salt_east'].attrs = dict(long_name="eastern boundary salinity", units="PSU", coordinates="bry_time")
ds_bry['u_north'].attrs = dict(long_name="northern boundary u-flux component", units="m/s", coordinates="bry_time")
ds_bry['v_north'].attrs = dict(long_name="northern boundary v-flux component", units="m/s", coordinates="bry_time")
ds_bry['temp_north'].attrs = dict(long_name="northern boundary potential temperature", units="degrees Celsius", coordinates="bry_time")
ds_bry['salt_north'].attrs = dict(long_name="northern boundary salinity", units="PSU", coordinates="bry_time")
ds_bry['zeta_north'].attrs = dict(long_name="northern boundary sea surface height", units="m", coordinates="bry_time")
ds_bry['zeta_south'].attrs = dict(long_name="southern boundary sea surface height", units="m", coordinates="bry_time")
ds_bry['zeta_east'].attrs = dict(long_name="eastern boundary sea surface height", units="m", coordinates="bry_time")
ds_bry['bry_time'].attrs = dict(long_name='relative time: days since 2000-01-01 00:00:00', units='days')
# ...add more variable attributes as needed...

# Write to NetCDF
ds_bry.to_netcdf(output_bry, format='NETCDF4', encoding={var: {'_FillValue': np.nan} for var in ds_bry.data_vars})
print(f"Boundary condition NetCDF file written: {output_bry}")

