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

from interpolate import interp_tools 
from conversions import convert_tools
from grid import grid_tools
from initialization import init_tools
import xarray as xr
import numpy as np
import os 

# ============================================================================
# LOCATION-SPECIFIC PARAMETERS - Customize these for your domain
# ============================================================================

# Set paths to data files and grid file 
base_path = '/global/cfs/cdirs/m4304/enuss/model-tools/'
grid_nc = 'roms_grid_1km_smoothed.nc' 
forcing_datapath = '/global/cfs/cdirs/m4304/enuss/US_East_Coast/Forcing_Data'
output_nc = 'initial_conditions_api.nc'

# ============================================================================
# DATA SOURCE CONFIGURATION
# Choose between NetCDF file or Copernicus Marine API
# ============================================================================

# Set to True to use Copernicus Marine API, False to use local NetCDF file
USE_API = True

# If USE_API is False, specify path to existing NetCDF file
NETCDF_PATH = '/global/cfs/cdirs/m4304/enuss/US_East_Coast/Forcing_Data/GLORYS_data/GLORYS_January.nc'

# If USE_API is True, these parameters define the spatial/temporal extent
# (The lon/lat ranges can be auto-detected from the grid if set to None)
API_LON_RANGE = None  # e.g., (-80.0, -60.0) or None to auto-detect
API_LAT_RANGE = None  # e.g., (30.0, 50.0) or None to auto-detect
API_TIME_BUFFER_DAYS = 1  # Days before/after init_time to download

# Initialization time
init_time = np.datetime64('2024-01-01T00:00:00')

# Source data variable names (map to your data source)
source_vars = {
    'salt': 'so',
    'temp': 'thetao',
    'u': 'uo',
    'v': 'vo',
    'zeta': 'zos',
    'depth': 'depth',
    'lat': 'latitude',
    'lon': 'longitude'
}

# Deep ocean layer parameters
new_depth = 10000.0
deep_ocean_fill_values = {
    'zos': np.nan,      # sea surface height
    'uo': 0.0,          # eastward velocity  
    'vo': 0.0,          # northward velocity
    'so': 35.0,         # salinity (typical deep ocean value)
    'thetao': 1.0       # potential temperature (typical deep ocean value)
}

# Fill values for interpolated fields
fill_values = {
    'temp': 5.0,        # degrees Celsius
    'salt': 32.0,       # PSU
    'u': 0.0,           # m/s
    'v': 0.0,           # m/s
    'zeta': 0.0         # m
}

# Minimum temperature constraint
min_temp = 0.1  # degrees Celsius

# ============================================================================
# INITIALIZATION - Uses reusable functions from code classes
# ============================================================================

# Load the grid 
grid = xr.open_dataset(os.path.join(base_path, 'output', grid_nc))

# Auto-detect lon/lat ranges from grid if not specified
if USE_API and (API_LON_RANGE is None or API_LAT_RANGE is None):
    lon_min, lon_max = float(grid.lon_rho.min()), float(grid.lon_rho.max())
    lat_min, lat_max = float(grid.lat_rho.min()), float(grid.lat_rho.max())
    
    # Add buffer for interpolation
    lon_buffer = (lon_max - lon_min) * 0.1
    lat_buffer = (lat_max - lat_min) * 0.1
    
    if API_LON_RANGE is None:
        API_LON_RANGE = (lon_min - lon_buffer, lon_max + lon_buffer)
    if API_LAT_RANGE is None:
        API_LAT_RANGE = (lat_min - lat_buffer, lat_max + lat_buffer)
    
    print(f"Auto-detected spatial extent from grid:")
    print(f"  Longitude: {API_LON_RANGE}")
    print(f"  Latitude: {API_LAT_RANGE}")

# Load initialization data
if USE_API:
    print("\n=== Loading data via Copernicus Marine API ===")
    init_data = init_tools.load_glorys_data(
        init_time=init_time,
        lon_range=API_LON_RANGE,
        lat_range=API_LAT_RANGE,
        use_api=True,
        time_buffer_days=API_TIME_BUFFER_DAYS
    )
else:
    print("\n=== Loading data from NetCDF file ===")
    init_data = init_tools.load_glorys_data(
        init_time=init_time,
        lon_range=None,  # Not used for file loading
        lat_range=None,  # Not used for file loading
        use_api=False,
        netcdf_path=NETCDF_PATH
    )

# Add deep ocean layer to prevent extrapolation issues
init_data = init_tools.add_deep_ocean_layer(
    init_data, new_depth, deep_ocean_fill_values,
    time_var='time', depth_var=source_vars['depth'],
    lat_var=source_vars['lat'], lon_var=source_vars['lon']
)

# Compute time since reference
seconds_since_2000, days_since_2000 = init_tools.compute_time_since_reference(init_time)
print(f"Initialization time: {init_time}")
print(f"Seconds since 2000-01-01: {seconds_since_2000:.6e}")
print(f"Days since 2000-01-01: {days_since_2000:.6f}")

# Select time step from source data
time_idx = np.argmin(np.abs(init_data.time.values - init_time))
source_time_step = init_data.isel(time=time_idx)

# Get grid dimensions 
eta_rho, xi_rho, eta_v, xi_u, s_rho, s_w = grid_tools.get_grid_dims(grid)

# Prepare source coordinates
source_coords = init_tools.prepare_source_coords(
    init_data, source_vars['depth'], source_vars['lat'], source_vars['lon']
)

# Compute ROMS vertical coordinates
# Convert hc to float if it's a DataArray to avoid broadcasting issues
hc = float(grid.hc.values) if hasattr(grid.hc, 'values') else float(grid.hc)
z_rho = grid_tools.compute_z(
    grid.sigma_r.values, hc, grid.Cs_r.values, 
    grid.h.values, np.zeros((eta_rho, xi_rho, 1))
)
z_rho = np.squeeze(z_rho)  # Remove singleton dimension
z_rho = np.transpose(z_rho, (1, 2, 0))  # Shape: (eta_rho, xi_rho, s_rho)

# Get ROMS target coordinates
roms_coords = {
    'lon_rho': grid.lon_rho.values,
    'lat_rho': grid.lat_rho.values,
    'lon_u': grid.lon_u.values,
    'lat_u': grid.lat_u.values,
    'lon_v': grid.lon_v.values,
    'lat_v': grid.lat_v.values,
    'z_rho': -z_rho  # Negative for interpolation
}

# ============================================================================
# INTERPOLATE VARIABLES TO ROMS GRID
# ============================================================================

print("\n=== Interpolating Temperature ===")
temp_interp = init_tools.interpolate_and_mask_3d(
    source_time_step[source_vars['temp']].values,
    source_coords,
    roms_coords['lon_rho'], roms_coords['lat_rho'], roms_coords['z_rho'],
    grid.mask_rho.values.astype(bool),
    fill_values['temp'],
    interp_method='linear',
    min_value=min_temp
)
print(f"Temperature shape: {temp_interp.shape}")

print("\n=== Interpolating Salinity ===")
sal_interp = init_tools.interpolate_and_mask_3d(
    source_time_step[source_vars['salt']].values,
    source_coords,
    roms_coords['lon_rho'], roms_coords['lat_rho'], roms_coords['z_rho'],
    grid.mask_rho.values.astype(bool),
    fill_values['salt'],
    interp_method='linear'
)
print(f"Salinity shape: {sal_interp.shape}")

print("\n=== Interpolating U Velocity ===")
u_interp = init_tools.interpolate_and_mask_3d(
    source_time_step[source_vars['u']].values,
    source_coords,
    roms_coords['lon_u'], roms_coords['lat_u'], roms_coords['z_rho'],
    grid.mask_u.values.astype(bool),
    fill_values['u'],
    interp_method='linear'
)
print(f"U velocity shape: {u_interp.shape}")

print("\n=== Interpolating V Velocity ===")
v_interp = init_tools.interpolate_and_mask_3d(
    source_time_step[source_vars['v']].values,
    source_coords,
    roms_coords['lon_v'], roms_coords['lat_v'], roms_coords['z_rho'],
    grid.mask_v.values.astype(bool),
    fill_values['v'],
    interp_method='linear'
)
print(f"V velocity shape: {v_interp.shape}")

print("\n=== Interpolating Sea Surface Height ===")
zeta_data = source_time_step[source_vars['zeta']].values
# Find the surface by identifying which dimension doesn't match x/y
zeta_shape = zeta_data.shape
expected_y = source_coords['lat_2d'].shape[0]
expected_x = source_coords['lon_2d'].shape[1]

# Determine which axis is the depth/vertical dimension
if len(zeta_shape) == 3:
    # Find which dimension doesn't match expected x or y
    for i, dim_size in enumerate(zeta_shape):
        if dim_size != expected_y and dim_size != expected_x:
            # This is likely the depth dimension, select surface (index 0)
            zeta_data = np.take(zeta_data, 0, axis=i)
            print(f"Selected surface level from axis {i}")
            break
elif len(zeta_shape) == 2:
    # Already 2D, no selection needed
    print("Zeta data is already 2D")
else:
    raise ValueError(f"Unexpected zeta_data shape: {zeta_shape}")

zeta_interp = init_tools.interpolate_and_mask_2d(
    zeta_data,
    source_coords['lon_2d'], source_coords['lat_2d'],
    roms_coords['lon_rho'], roms_coords['lat_rho'],
    fill_values['zeta'],
    interp_method='linear'
)
print(f"Zeta shape: {zeta_interp.shape}")

# ============================================================================
# COMPUTE DERIVED VARIABLES
# ============================================================================

print("\n=== Computing Barotropic Velocities ===")
# Reshape z_rho from (ny, nx, nz) to (nz, ny, nx) for compute_uvbar
z_rho_transposed = np.transpose(z_rho, (2, 0, 1))
ubar, vbar = convert_tools.compute_uvbar(u_interp, v_interp, z_rho_transposed)
print(f"Ubar shape: {ubar.shape}")
print(f"Vbar shape: {vbar.shape}")

print("\n=== Computing Vertical Velocity ===")
w = convert_tools.compute_w(u_interp, v_interp, grid.pm.values, grid.pn.values, z_rho_transposed)
print(f"W velocity shape: {w.shape}")

# ============================================================================
# CREATE AND SAVE INITIAL CONDITIONS DATASET
# ============================================================================

print("\n=== Creating Initial Conditions Dataset ===")
ds = init_tools.create_initial_conditions_dataset(
    grid, temp_interp, sal_interp, u_interp, v_interp, w, 
    ubar, vbar, zeta_interp, days_since_2000, init_time, 
    source_name='GLORYS'
)

output_path = os.path.join(base_path, 'output', output_nc)
ds.to_netcdf(output_path, format='NETCDF4', engine='netcdf4')
print(f"\nInitial conditions saved to: {output_path}")