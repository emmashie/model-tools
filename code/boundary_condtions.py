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

t = 0 
temp_data = bc_data['thetao'][t,:,:,:].values 
salt_data = bc_data['so'][t,:,:,:].values
u_data = bc_data['uo'][t,:,:,:].values
v_data = bc_data['vo'][t,:,:,:].values
zeta_data = bc_data['zos'][t,:,:,:].values

# interpolate boundary condition data to ROMS grid
temp_interp = interp_tools.interp3d(
    temp_data, glorys_lon_2d, glorys_lat_2d, glorys_depth, roms_lon_2d, roms_lat_2d, -z_rho, method='linear'
)

