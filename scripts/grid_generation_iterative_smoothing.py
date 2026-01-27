import xarray as xr
import numpy as np
from netCDF4 import Dataset
import cmocean
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import hvplot.xarray
import geoviews as gv
import os 
from grid import grid_tools

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation

import funpy.model_utils as mod_utils 
plt.ion()

base_path = '/global/cfs/cdirs/m4304/enuss/model-tools'
bathy_nc = 'east_coast_bathy_final.nc' 
output_nc = 'roms_grid_1km_smoothed.nc'  # Output ROMS grid file
dx = 0.01  # grid resolution in degrees longitude
dy = 0.01  # grid resolution in degrees latitude
N = 50     # number of vertical levels
lon_range = (-80, -60)  # longitude range for the grid
lat_range = (33, 46)    # latitude range for the grid

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

## create u and v grids 
# u-grid: points are halfway between rho points in the x-direction (longitude)
lon_u = 0.5 * (lon_rho_grid[:, :-1] + lon_rho_grid[:, 1:])
lat_u = 0.5 * (lat_rho_grid[:, :-1] + lat_rho_grid[:, 1:])

# v-grid: points are halfway between rho points in the y-direction (latitude)
lon_v = 0.5 * (lon_rho_grid[:-1, :] + lon_rho_grid[1:, :])
lat_v = 0.5 * (lat_rho_grid[:-1, :] + lat_rho_grid[1:, :])

# psi grid 
lon_psi = 0.25 * (lon_rho_grid[:-1, :-1] + lon_rho_grid[1:, :-1] + 
                  lon_rho_grid[:-1, 1:] + lon_rho_grid[1:, 1:])
lat_psi = 0.25 * (lat_rho_grid[:-1, :-1] + lat_rho_grid[1:, :-1] + 
                  lat_rho_grid[:-1, 1:] + lat_rho_grid[1:, 1:])

# compute grid metrics 
# Earth's radius in meters
R = 6371000

# Convert degrees to radians for calculations
lat_rho_rad = np.deg2rad(lat_rho_grid)
lon_rho_rad = np.deg2rad(lon_rho_grid)

# Compute dx (meters) - distance between adjacent points in x (longitude) direction
dx_m = np.zeros_like(lon_rho_grid)
dx_m[:, :-1] = R * np.cos(lat_rho_rad[:, :-1]) * (lon_rho_rad[:, 1:] - lon_rho_rad[:, :-1])
dx_m[:, -1] = dx_m[:, -2]  # pad last column
pm = 1/dx_m

# Compute dy (meters) - distance between adjacent points in y (latitude) direction
dy_m = np.zeros_like(lat_rho_grid)
dy_m[:-1, :] = R * (lat_rho_rad[1:, :] - lat_rho_rad[:-1, :])
dy_m[-1, :] = dy_m[-2, :]  # pad last row
pn = 1/dy_m

# compute xl and el 
xl = np.sum(dx_m[0, :])
el = np.sum(dy_m[:, 0])

# compute f 
f = 2 * 7.2921159e-5 * np.sin(lat_rho_rad)  

# Smooth the log of the bathymetry
bathy_masked = np.ma.masked_where(bathy > 0, bathy)
sigma = 10  # smoothing strength, adjust as needed

# Smooth the entire grid, then blend only steep regions
bathy_smooth = gaussian_filter(np.log(np.abs(bathy_masked)), sigma=sigma)
bathy_smooth = -np.exp(bathy_smooth)  # Convert back to depth
bathy_smooth = np.where(bathy_masked.mask, bathy, bathy_smooth)
#bathy_smooth = mod_utils.spatially_avg(np.log(np.abs(bathy_masked)), lon, lat, order=1, filtx=dx*10, filty=dy*10)
#bathy_smooth = -np.exp(bathy_smooth)  # Convert back to depth
#bathy_smooth = np.where(bathy_masked.mask, bathy, bathy_smooth)

# Interpolate bathymetry to new grid 
lon_grid, lat_grid = np.meshgrid(lon, lat)
values = bathy_smooth.flatten()
h = griddata(
    (lon_grid.flatten(), lat_grid.flatten()), values,
    (lon_rho_grid.flatten(), lat_rho_grid.flatten()), method='linear'
)
h = h.reshape(ny, nx)

# Create mask (1 for ocean, 0 for land)
hmin = -5
mask_rho = np.where(h < -hmin, 1, 0)

# Add masking of lake in the grid 
mask_rho[-60:,:150] = 0 

# Mask out Northumberland strait 
lat_ind = np.where(lat_rho>45.6)[0]
lon_ind = np.where((lon_rho>-64.4) & (lon_rho<-61.2))[0]
mask_rho[lat_ind[0]:,lon_ind[0]:lon_ind[-1]] = 0

# Create mask for u and v grids
mask_u = np.minimum(mask_rho[:, :-1], mask_rho[:, 1:])
mask_v = np.minimum(mask_rho[:-1, :], mask_rho[1:, :])

# Psi points are at cell corners - a psi point is wet only if 
# all 4 surrounding rho points are wet
mask_psi = (mask_rho[:-1, :-1] *    # SW rho point
            mask_rho[1:, :-1] *     # NW rho point  
            mask_rho[:-1, 1:] *     # SE rho point
            mask_rho[1:, 1:])       # NE rho point

# fill h values where h is shallower than hmin (aka mask is 0) with 0 m depth 
h[h > hmin] = hmin
# check that all nans are replaced with 0
h = np.nan_to_num(h, nan=hmin)

rx0_x, rx0_y = grid_tools.compute_rx0(np.abs(h)) 
# Pad rx0_x with an extra row in the y-direction
rx0_x = np.pad(rx0_x, ((0, 0), (0, 1)), mode='edge')
# Pad rx0_y with an extra column in the x-direction
rx0_y = np.pad(rx0_y, ((0, 1), (0, 0)), mode='edge')

print(max(np.max(rx0_x), np.max(rx0_y)))

# Iterative localized smoothing for steep regions
buffer_size = 5  # number of grid cells for buffer (adjust as needed)
sigma = 6  # smoothing strength, adjust as needed
max_iter = 10
rx0_thresh = 0.2

h_iter = h.copy()
for i in range(max_iter):
    rx0_x, rx0_y = grid_tools.compute_rx0(np.abs(h_iter))
    # Pad rx0_x with an extra row in the y-direction
    rx0_x = np.pad(rx0_x, ((0, 0), (0, 1)), mode='edge')
    # Pad rx0_y with an extra column in the x-direction
    rx0_y = np.pad(rx0_y, ((0, 1), (0, 0)), mode='edge')
    max_rx0 = max(np.max(rx0_x), np.max(rx0_y))
    print(f"Iteration {i+1}: max(rx0) = {max_rx0:.4f}")
    if max_rx0 < rx0_thresh:
        print(f"Smoothing converged: max(rx0) < {rx0_thresh} after {i+1} iterations.")
        break
    steep_mask = (rx0_x > rx0_thresh) | (rx0_y > rx0_thresh)
    steep_mask_buffered = binary_dilation(steep_mask, iterations=buffer_size)
    h_gauss = gaussian_filter(h_iter, sigma=sigma)
    # Create a smooth blending mask using a Gaussian filter on the binary mask
    blend_mask = gaussian_filter(steep_mask_buffered.astype(float), sigma=2)
    blend_mask = np.clip(blend_mask, 0, 1)  # Ensure values are in [0, 1]
    # Blend the original and smoothed bathymetry
    h_iter = h_iter * (1 - blend_mask) + h_gauss * blend_mask
    print(f"Iteration {i+1}: {np.sum(steep_mask_buffered)} grid points blended.")
h = h_iter 


# Vertical levels (sigma coordinates) 
sigma_w = grid_tools.compute_sigma(N, type='w')
sigma_r = grid_tools.compute_sigma(N, type='r')

theta_s = 5 
theta_b = 0.5 

Cs_w = grid_tools.compute_cs(sigma_w, theta_s, theta_b)
Cs_r = grid_tools.compute_cs(sigma_r, theta_s, theta_b)

hc = -500 

h_mask = np.where(mask_rho == 1, h, np.nan)
z_rho = grid_tools.compute_z(sigma_r, hc, Cs_r, h_mask, np.zeros((ny, nx, 1)))  
z_rho = np.squeeze(z_rho)  # shape: (nz, ny, nx)

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
plt.show()

# Create ROMS grid xarray Dataset and save to NetCDF
# Create the dataset
ds_out = xr.Dataset(
    {
        "lat_rho": (("eta_rho", "xi_rho"), lat_rho_grid, {
            "_FillValue": np.nan,
            "long_name": "latitude of rho-points",
            "units": "degrees North"
        }),
        "lon_rho": (("eta_rho", "xi_rho"), lon_rho_grid, {
            "_FillValue": np.nan,
            "long_name": "longitude of rho-points",
            "units": "degrees East"
        }),
        "lat_u": (("eta_rho", "xi_u"), lat_u, {
            "_FillValue": np.nan,
            "long_name": "latitude of u-points",
            "units": "degrees North"
        }),
        "lon_u": (("eta_rho", "xi_u"), lon_u, {
            "_FillValue": np.nan,
            "long_name": "longitude of u-points",
            "units": "degrees East"
        }),
        "lat_v": (("eta_v", "xi_rho"), lat_v, {
            "_FillValue": np.nan,
            "long_name": "latitude of v-points",
            "units": "degrees North"
        }),
        "lon_v": (("eta_v", "xi_rho"), lon_v, {
            "_FillValue": np.nan,
            "long_name": "longitude of v-points",
            "units": "degrees East"
        }),
        "y_rho": (("eta_rho", "xi_rho"), lat_rho_grid, {
            "_FillValue": np.nan,
            "long_name": "latitude of rho-points",
            "units": "degrees North"
        }),
        "lat_psi": (("eta_psi", "xi_psi"), lat_psi, {
            "_FillValue": np.nan,
            "long_name": "latitude of psi-points",
            "units": "degrees North"
        }),
        "lon_psi": (("eta_psi", "xi_psi"), lon_psi, {
            "_FillValue": np.nan,
            "long_name": "longitude of psi-points",
            "units": "degrees East"
        }),        
        "x_rho": (("eta_rho", "xi_rho"), lon_rho_grid, {
            "_FillValue": np.nan,
            "long_name": "longitude of rho-points",
            "units": "degrees East"
        }),        
        "y_u": (("eta_rho", "xi_u"), lat_u, {
            "_FillValue": np.nan,
            "long_name": "latitude of u-points",
            "units": "degrees North"
        }),
        "x_u": (("eta_rho", "xi_u"), lon_u, {
            "_FillValue": np.nan,
            "long_name": "longitude of u-points",
            "units": "degrees East"
        }),
        "y_v": (("eta_v", "xi_rho"), lat_v, {
            "_FillValue": np.nan,
            "long_name": "latitude of v-points",
            "units": "degrees North"
        }),
        "x_v": (("eta_v", "xi_rho"), lon_v, {
            "_FillValue": np.nan,
            "long_name": "longitude of v-points",
            "units": "degrees East"
        }),  
        "y_psi": (("eta_psi", "xi_psi"), lat_psi, {
            "_FillValue": np.nan,
            "long_name": "latitude of psi-points",
            "units": "degrees North"
        }),
        "x_psi": (("eta_psi", "xi_psi"), lon_psi, {
            "_FillValue": np.nan,
            "long_name": "longitude of psi-points",
            "units": "degrees East"
        }),                      
        "angle": (("eta_rho", "xi_rho"), np.zeros_like(lat_rho_grid), {
            "_FillValue": np.nan,
            "long_name": "Angle between xi axis and east",
            "units": "radians",
            "coordinates": "lat_rho lon_rho"
        }),
        "f": (("eta_rho", "xi_rho"), f, {
            "_FillValue": np.nan,
            "long_name": "Coriolis parameter at rho-points",
            "units": "second-1",
            "coordinates": "lat_rho lon_rho"
        }),
        "pm": (("eta_rho", "xi_rho"), pm, {
            "_FillValue": np.nan,
            "long_name": "Curvilinear coordinate metric in xi-direction",
            "units": "meter-1",
            "coordinates": "lat_rho lon_rho"
        }),
        "pn": (("eta_rho", "xi_rho"), pn, {
            "_FillValue": np.nan,
            "long_name": "Curvilinear coordinate metric in eta-direction",
            "units": "meter-1",
            "coordinates": "lat_rho lon_rho"
        }),
        "spherical": ((), "T", {
            "Long_name": "Grid type logical switch",
            "option_T": "spherical"
        }),
        "mask_rho": (("eta_rho", "xi_rho"), mask_rho, {
            "long_name": "Mask at rho-points",
            "units": "land/water (0/1)",
            "coordinates": "lat_rho lon_rho"
        }),
        "mask_u": (("eta_rho", "xi_u"), mask_u, {
            "long_name": "Mask at u-points",
            "units": "land/water (0/1)",
            "coordinates": "lat_u lon_u"
        }),
        "mask_v": (("eta_v", "xi_rho"), mask_v, {
            "long_name": "Mask at v-points",
            "units": "land/water (0/1)",
            "coordinates": "lat_v lon_v"
        }),
        "mask_psi": (("eta_psi", "xi_psi"), mask_psi, {
            "long_name": "Mask at psi-points",
            "units": "land/water (0/1)",
            "coordinates": "lat_psi lon_psi"
        }),        
        "h": (("eta_rho", "xi_rho"), -h, {
            "_FillValue": np.nan,
            "long_name": "Bathymetry at rho-points",
            "units": "meter",
            "coordinates": "lat_rho lon_rho"
        }),
        "sigma_r": (("s_rho",), sigma_r, {
            "_FillValue": np.nan,
            "long_name": "Fractional vertical stretching coordinate at rho-points",
            "units": "nondimensional"
        }),
        "Cs_r": (("s_rho",), Cs_r, {
            "_FillValue": np.nan,
            "long_name": "Vertical stretching function at rho-points",
            "units": "nondimensional"
        }),
        "sigma_w": (("s_w",), sigma_w, {
            "_FillValue": np.nan,
            "long_name": "Fractional vertical stretching coordinate at w-points",
            "units": "nondimensional"
        }),
        "Cs_w": (("s_w",), Cs_w, {
            "_FillValue": np.nan,
            "long_name": "Vertical stretching function at w-points",
            "units": "nondimensional"
        }),
        "xl": ((), xl, {
            "long_name": "domain length in the XI-direction",
            "units": "meter"
        }),
        "el": ((), el, {
            "long_name": "domain length in the ETA-direction",
            "units": "meter"
        }),
    },
    coords={
        "eta_rho": np.arange(ny),
        "xi_rho": np.arange(nx),
        "xi_u": np.arange(nx - 1),
        "eta_v": np.arange(ny - 1),
        "eta_psi": np.arange(ny-1),
        "xi_psi": np.arange(nx -1),
        "s_rho": np.arange(N),
        "s_w": np.arange(N + 1)
    }
)

# Assign global attributes
ds_out.attrs = {
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

# Save to NetCDF file
ds_out.to_netcdf(os.path.join(base_path, 'output', output_nc))

