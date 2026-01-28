import xarray as xr
import numpy as np
import cmocean
import matplotlib.pyplot as plt
import os 
import sys

# Add parent directory to path to import from code/
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # When running as a script, go to parent directory then to code
    code_dir = os.path.join(os.path.dirname(script_dir), 'code')
except NameError:
    # When running interactively, use base_path if available or manually specify
    # This will be defined later in the script, so we'll add it after
    code_dir = '/global/cfs/cdirs/m4304/enuss/model-tools/code'

if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from grid import grid_tools

import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.ion()

# ============================================================================
# LOCATION-SPECIFIC PARAMETERS - Customize these for your domain
# ============================================================================

base_path = '/global/cfs/cdirs/m4304/enuss/model-tools'
bathy_nc = 'east_coast_bathy_final.nc' 
output_dir = "/global/cfs/cdirs/m4304/enuss/model-tools/output"
output_nc = os.path.join(output_dir, 'roms_grid_1km_smoothed.nc')  # Output ROMS grid file

# Grid resolution
dx = 0.01  # grid resolution in degrees longitude
dy = 0.01  # grid resolution in degrees latitude

# Domain extent
lon_range = (-80, -60)  # longitude range for the grid
lat_range = (33, 46)    # latitude range for the grid

# Vertical parameters
N = 50         # number of vertical levels
theta_s = 5    # surface stretching parameter
theta_b = 0.5  # bottom stretching parameter
hc = -500      # critical depth (negative)

# Bathymetry smoothing parameters
initial_smooth_sigma = 10  # Initial Gaussian smoothing strength
hmin = -5                  # Minimum depth threshold (negative)

# Iterative smoothing parameters
rx0_thresh = 0.2     # rx0 threshold for iterative smoothing
max_iter = 10        # Maximum iterations
smooth_sigma = 6     # Smoothing strength for iterative method
buffer_size = 5      # Buffer around steep regions


# ============================================================================
# GRID GENERATION - Uses grid_tools class for reusable operations
# ============================================================================

# Load bathymetry data
ds = xr.open_dataset(bathy_nc)
bathy = ds.z.values
lon = ds.lon.values
lat = ds.lat.values

# Define new grid
lon_min, lon_max = lon_range
lat_min, lat_max = lat_range
nx = int((lon_max - lon_min) / dx) + 1
ny = int((lat_max - lat_min) / dy) + 1

lon_rho = np.linspace(lon_min, lon_max, nx)
lat_rho = np.linspace(lat_min, lat_max, ny)
lon_rho_grid, lat_rho_grid = np.meshgrid(lon_rho, lat_rho)

# Create staggered grids (u, v, psi)
staggered_grids = grid_tools.create_staggered_grids(lon_rho_grid, lat_rho_grid)

# Compute grid metrics (pm, pn, f, xl, el)
metrics = grid_tools.compute_grid_metrics(lon_rho_grid, lat_rho_grid)

# Interpolate and smooth bathymetry
h = grid_tools.interpolate_bathymetry(
    bathy, lon, lat, lon_rho_grid, lat_rho_grid, 
    smooth_sigma=initial_smooth_sigma, use_log_smoothing=True
)

# Create initial masks
masks = grid_tools.create_masks(h, hmin)
mask_rho = masks['mask_rho']

# ============================================================================
# LOCATION-SPECIFIC MASK MODIFICATIONS
# ============================================================================

# Add masking of lake in the grid 
mask_rho[-60:,:150] = 0 

# Mask out Northumberland strait 
lat_ind = np.where(lat_rho>45.6)[0]
lon_ind = np.where((lon_rho>-64.4) & (lon_rho<-61.2))[0]
mask_rho[lat_ind[0]:,lon_ind[0]:lon_ind[-1]] = 0

# Recreate all masks with modified mask_rho
masks = grid_tools.create_masks(h, hmin, mask_rho=mask_rho)

# ============================================================================
# BATHYMETRY SMOOTHING
# ============================================================================

# Fill h values where h is shallower than hmin with hmin 
h[h > hmin] = hmin
h = np.nan_to_num(h, nan=hmin)

# Check initial rx0
rx0_x, rx0_y = grid_tools.compute_rx0(np.abs(h)) 
rx0_x = np.pad(rx0_x, ((0, 0), (0, 1)), mode='edge')
rx0_y = np.pad(rx0_y, ((0, 1), (0, 0)), mode='edge')
print(f"Initial max(rx0) = {max(np.max(rx0_x), np.max(rx0_y)):.4f}")

# Iterative localized smoothing for steep regions
h = grid_tools.iterative_smoothing(h, rx0_thresh, max_iter, smooth_sigma, buffer_size)

# ============================================================================
# VERTICAL COORDINATES
# ============================================================================

# Vertical levels (sigma coordinates) 
sigma_w = grid_tools.compute_sigma(N, type='w')
sigma_r = grid_tools.compute_sigma(N, type='r')

Cs_w = grid_tools.compute_cs(sigma_w, theta_s, theta_b)
Cs_r = grid_tools.compute_cs(sigma_r, theta_s, theta_b)

# Compute vertical levels for visualization
h_mask = np.where(masks['mask_rho'] == 1, h, np.nan)
z_rho = grid_tools.compute_z(sigma_r, hc, Cs_r, h_mask, np.zeros((ny, nx, 1)))  
z_rho = np.squeeze(z_rho)  # shape: (nz, ny, nx)

# ============================================================================
# VISUALIZATION
# ============================================================================


plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=(lon_min + lon_max) / 2,
                                               central_latitude=(lat_min + lat_max) / 2))
pc = ax.pcolormesh(lon_rho_grid, lat_rho_grid, -h, cmap=cmocean.cm.deep, shading='auto', transform=ccrs.PlateCarree())
ax.coastlines(resolution='10m', color='k')
ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k', facecolor='0.8')
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
plt.colorbar(pc, ax=ax, orientation='vertical', label='Depth (m)')
plt.title('Bathymetry (h) on ROMS Grid')
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'plots', 'roms_grid_bathymetry.png'), dpi=300)

# ============================================================================
# CREATE AND SAVE ROMS GRID DATASET
# ============================================================================

# Prepare sigma parameters dictionary
sigma_params = {
    'sigma_r': sigma_r,
    'sigma_w': sigma_w,
    'Cs_r': Cs_r,
    'Cs_w': Cs_w,
    'N': N
}

# Prepare global attributes
global_attrs = {
    "title": "ROMS grid created by model-tools",
    "size_x": nx,
    "size_y": ny,
    "center_lon": (lon_min + lon_max) / 2,
    "center_lat": (lat_min + lat_max) / 2,
    "rot": 0,
    "topography_source": "SRTM15+",
    "hmin": -hmin,
    "theta_s": theta_s,
    "theta_b": theta_b,
    "hc": -hc  
}

# Create ROMS grid dataset
ds_out = grid_tools.create_roms_grid_dataset(
    lon_rho_grid, lat_rho_grid, -h,  # Note: h is converted to positive for ROMS
    masks, staggered_grids, metrics, sigma_params, global_attrs
)

# Save to NetCDF file
output_path = os.path.join(base_path, 'output', output_nc)
ds_out.to_netcdf(output_path)
print(f"Grid saved to {output_path}")

