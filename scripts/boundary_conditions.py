import xarray as xr
import numpy as np
import sys
import os

# Add parent directory to path to import from code/
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(os.path.dirname(script_dir), 'code')
except NameError:
    code_dir = '/global/cfs/cdirs/m4304/enuss/model-tools/code'

if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from conversions import convert_tools
from initialization import init_tools
from boundary import boundary_tools
from grid import grid_tools
import pandas as pd

# ========== Location-Specific Parameters ==========
# Specify which boundaries to include
boundaries = {
    'west': False,
    'east': True,
    'north': True,
    'south': True
}

# Time range for boundary conditions
start_time = '2024-1-01'  # bc_data['time'][24].values
end_time = '2024-1-02'

# Output options
save_climatology = True  # Set to False to skip climatology file creation
save_boundary = True     # Set to False to skip boundary forcing file creation

# File paths
base_path = '/global/cfs/cdirs/m4304/enuss/model-tools'
forcing_datapath = '/global/cfs/cdirs/m4304/enuss/US_East_Coast/Forcing_Data'
grid_file = os.path.join(base_path, 'output', 'roms_grid_1km_smoothed.nc')
glorys_data_pattern = os.path.join(forcing_datapath, 'GLORYS_data', 'GLORYS_*.nc')
output_climatology_file = os.path.join(base_path, 'output', 'east-coast-climatology_2023-2024.nc')
output_boundary_file = os.path.join(base_path, 'output', 'east-coast-boundary_2023-2024.nc')

# Source data name
source_name = 'GLORYS12v1'

# Model reference time
ref_time = '2000-01-01 00:00:00'


print("Loading data...")
# Load grid and boundary condition data
grid = xr.open_dataset(grid_file)
bc_data = xr.open_mfdataset(glorys_data_pattern)

# Subset bc_data to the specified time range
bc_data = bc_data.sel(time=slice(start_time, end_time))

# Set variable names for GLORYS data
salt_var = 'so'
temp_var = 'thetao' 
u_var = 'uo'
v_var = 'vo'
zeta_var = 'zos'
depth_var = 'depth'
lat_var = 'latitude'
lon_var = 'longitude'

# Get valid time indices (where uo data is not all NaN)
valid_time_indices = np.where(np.isfinite(bc_data['uo'][:,0,-1,-1].values))[0]
print(f"Found {len(valid_time_indices)} valid time steps")

# Prepare source coordinates using init_tools
print("Preparing source coordinates...")
source_coords = init_tools.prepare_source_coords(
    bc_data, depth_var, lat_var, lon_var
)

# Compute ROMS vertical coordinates
print("Computing ROMS vertical coordinates...")

eta_rho, xi_rho, eta_v, xi_u, s_rho, s_w = grid_tools.get_grid_dims(grid)

z_rho = grid_tools.compute_z(
    grid.sigma_r.values, 
    grid.hc, 
    grid.Cs_r.values, 
    grid.h.values, 
    np.zeros((eta_rho, xi_rho, 1))
)
z_rho = np.squeeze(z_rho)
z_rho = np.transpose(z_rho, (1, 2, 0))  # Shape: (eta_rho, xi_rho, s_rho)

# Convert to positive depths for interpolation
roms_depth_3d = np.abs(z_rho)
roms_lat_2d = grid.lat_rho.values
roms_lon_2d = grid.lon_rho.values
roms_latu_2d = grid.lat_u.values
roms_lonu_2d = grid.lon_u.values
roms_latv_2d = grid.lat_v.values
roms_lonv_2d = grid.lon_v.values

nt = len(valid_time_indices)

# Initialize arrays for interpolated variables
print("Interpolating variables to ROMS grid...")

temp_interp = np.empty((nt, s_rho, eta_rho, xi_rho))
salt_interp = np.empty((nt, s_rho, eta_rho, xi_rho))
u_interp = np.empty((nt, s_rho, eta_rho, xi_u))
v_interp = np.empty((nt, s_rho, eta_v, xi_rho))
zeta_interp = np.empty((nt, eta_rho, xi_rho))

# Time in days since reference
ref_time_pd = pd.Timestamp(ref_time)
time_days = np.empty(nt)

# Interpolate each time step
for t in range(nt):
    idx = valid_time_indices[t]
    
    # Calculate time in days since reference
    time = bc_data['time'][idx].values
    current_time = pd.to_datetime(str(time))
    time_days[t] = (current_time - ref_time_pd).total_seconds() / 86400.0
    
    # Interpolate 3D variables using init_tools methods
    temp_data = bc_data[temp_var][idx, :, :, :].values
    temp_interp[t] = init_tools.interpolate_and_mask_3d(
        temp_data, source_coords,
        roms_lon_2d, roms_lat_2d, roms_depth_3d, 
        grid.mask_rho.values, fill_value=5.0
    )
    
    salt_data = bc_data[salt_var][idx, :, :, :].values
    salt_interp[t] = init_tools.interpolate_and_mask_3d(
        salt_data, source_coords,
        roms_lon_2d, roms_lat_2d, roms_depth_3d,
        grid.mask_rho.values, fill_value=32.0
    )
    
    u_data = bc_data[u_var][idx, :, :, :].values
    u_interp[t] = init_tools.interpolate_and_mask_3d(
        u_data, source_coords,
        roms_lonu_2d, roms_latu_2d, roms_depth_3d,
        grid.mask_u.values, fill_value=0.0
    )
    
    v_data = bc_data[v_var][idx, :, :, :].values
    v_interp[t] = init_tools.interpolate_and_mask_3d(
        v_data, source_coords,
        roms_lonv_2d, roms_latv_2d, roms_depth_3d,
        grid.mask_v.values, fill_value=0.0
    )
    
    # Interpolate 2D zeta
    zeta_data = bc_data[zeta_var][idx, :, :, :].values
    surf_idx = np.where(~np.isnan(zeta_data))[0][0] if zeta_data.ndim > 2 else 0
    zeta_data = zeta_data[surf_idx, :, :] if zeta_data.ndim > 2 else zeta_data
    zeta_interp[t] = init_tools.interpolate_and_mask_2d(
        zeta_data, source_coords['lon_2d'], source_coords['lat_2d'],
        roms_lon_2d, roms_lat_2d, fill_value=0.0
    )
    
    print(f"Interpolated time step {t + 1}/{nt}")

# Compute derived variables (ubar, vbar, w)
print("Computing derived variables (ubar, vbar, w)...")
z_rho_transposed = np.transpose(z_rho, (2, 0, 1))  # Shape: (s_rho, eta_rho, xi_rho)
ubar = np.empty((nt, eta_rho, xi_u))
vbar = np.empty((nt, eta_v, xi_rho))
w = np.empty((nt, s_rho, eta_rho, xi_rho))

for t in range(nt):
    ubar[t], vbar[t] = convert_tools.compute_uvbar(u_interp[t], v_interp[t], z_rho_transposed)
    w[t] = convert_tools.compute_w(u_interp[t], v_interp[t], grid.pm.values, grid.pn.values, z_rho_transposed)

# Save climatology file if requested
if save_climatology:
    print("Creating climatology dataset...")
    ds_clim = boundary_tools.create_climatology_dataset(
        temp_interp, salt_interp, u_interp, v_interp, w, ubar, vbar, zeta_interp,
        time_days, grid, source_name=source_name
    )
    
    print(f"Saving climatology to {output_climatology_file}...")
    ds_clim.to_netcdf(output_climatology_file, mode='w')
    print("Climatology file saved successfully!")

# Save boundary forcing file if requested
if save_boundary:
    print("Extracting boundary transects...")
    boundary_transects = boundary_tools.extract_boundary_transects(
        temp_interp, salt_interp, u_interp, v_interp, ubar, vbar, zeta_interp,
        grid, boundaries
    )
    
    print("Creating boundary forcing dataset...")
    ds_bry = boundary_tools.create_boundary_dataset(
        boundary_transects, time_days, grid, 
        start_time, end_time, source_name=source_name
    )
    
    print(f"Saving boundary forcing to {output_boundary_file}...")
    ds_bry.to_netcdf(output_boundary_file, mode='w')
    print("Boundary forcing file saved successfully!")

print("\nBoundary conditions script completed successfully!")


