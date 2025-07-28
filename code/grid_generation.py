import xarray as xr
import numpy as np
from netCDF4 import Dataset
import cmocean
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import hvplot.xarray
import geoviews as gv
import os 

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

import funpy.model_utils as mod_utils 
plt.ion()

plot = 1

base_path = '/global/cfs/cdirs/m4304/enuss/model-tools'
bathy_nc = 'east_coast_bathy_final.nc' 
output_nc = 'roms_grid.nc'  # Output ROMS grid file
dx = 0.05  # grid resolution in degrees longitude
dy = 0.05  # grid resolution in degrees latitude
N = 100     # number of vertical levels
lon_range = (-80, -45)  # longitude range for the grid
lat_range = (35, 45)    # latitude range for the grid

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

if plot==1:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    p0 = axs[0].pcolormesh(lon_rho_grid, lat_rho_grid, dx_m, cmap='viridis', shading='auto')
    axs[0].set_title('dx (meters)')
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    fig.colorbar(p0, ax=axs[0], label='Distance (m)')
    p0.set_clim(dx_m.min(), dx_m.max())

    p1 = axs[1].pcolormesh(lon_rho_grid, lat_rho_grid, dy_m, cmap='viridis', shading='auto')
    axs[1].set_title('dy (meters)')
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Latitude')
    fig.colorbar(p1, ax=axs[1], label='Distance (m)')
    p1.set_clim(dy_m.min()-10, dy_m.max()+10)
    fig.savefig('../plots/grid_metrics.png', dpi=300)

# compute xl and el 
xl = np.sum(dx_m[0, :])
el = np.sum(dy_m[:, 0])

# compute f 
f = 2 * 7.2921159e-5 * np.sin(lat_rho_rad)  

if plot==1:
    fig, ax = plt.subplots(figsize=(8, 6))
    p = ax.pcolormesh(lon_rho_grid, lat_rho_grid, f, cmap=cmocean.cm.phase, shading='auto')
    fig.colorbar(p, ax=ax)
    ax.set_title('Coriolis Parameter (f)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.savefig('../plots/coriolis_parameter.png', dpi=300)


# Smooth the log of the bathymetry
bathy_masked = np.ma.masked_where(bathy > 0, bathy)
bathy_smooth = mod_utils.spatially_avg(np.log(np.abs(bathy_masked)), lon, lat, order=1, filtx=dx, filty=dy)
bathy_smooth = -np.exp(bathy_smooth)  # Convert back to depth
bathy_smooth = np.where(bathy_masked.mask, bathy, bathy_smooth)

if plot==1:
    # Plot original and smoothed bathymetry side by side
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    pcm0 = axs[0].pcolormesh(lon, lat, bathy, cmap=cmocean.cm.topo, shading='auto')
    axs[0].set_title('Original Bathymetry')
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    fig.colorbar(pcm0, ax=axs[0], label='Depth (m)')
    pcm0.set_clim(-6000, 6000)

    pcm1 = axs[1].pcolormesh(lon, lat, bathy_smooth, cmap=cmocean.cm.topo, shading='auto')
    axs[1].set_title('Smoothed Bathymetry')
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Latitude')
    fig.colorbar(pcm1, ax=axs[1], label='Depth (m)')
    pcm1.set_clim(-6000, 6000)

    plt.tight_layout()
    fig.savefig('../plots/bathy_smoothing_comparison.png', dpi=300)


if plot==1:
    fig, ax = plt.subplots(figsize=(8, 6))
    # Define contour bins (e.g., every 500 meters from -6000 to 0)
    contour_bins = np.arange(-6000, 0, 500)
    p = ax.contourf(lon, lat, bathy_smooth, levels=contour_bins, cmap=cmocean.cm.deep_r)
    ax.set_title('Smoothed Bathymetry Contours')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.colorbar(p, ax=ax, label='Depth (m)')
    plt.savefig('../plots/bathy_smoothing_contours.png', dpi=300)
    plt.show()

# Interpolate bathymetry to new grid 
lon_grid, lat_grid = np.meshgrid(lon, lat)
values = bathy_smooth.flatten()
h = griddata(
    (lon_grid.flatten(), lat_grid.flatten()), values,
    (lon_rho_grid.flatten(), lat_rho_grid.flatten()), method='linear'
)
h = h.reshape(ny, nx)

if plot==1:
    # Plot contour plots of bathy_smooth and interpolated h side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

    contour_bins = np.arange(-6000, 0, 500)

    # Left: bathy_smooth (original grid)
    p0 = axs[0].contourf(lon, lat, bathy_smooth, levels=contour_bins, cmap=cmocean.cm.deep_r)
    fig.colorbar(p0, ax=axs[0], label='Depth (m)')
    axs[0].set_title('Smoothed Bathymetry (Original Grid)')
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    p0.set_clim(-6000, 0)
    # Add bounding box around lat/lon range of lat_rho and lon_rho
    bbox_lon = [lon_rho.min(), lon_rho.max(), lon_rho.max(), lon_rho.min(), lon_rho.min()]
    bbox_lat = [lat_rho.min(), lat_rho.min(), lat_rho.max(), lat_rho.max(), lat_rho.min()]
    axs[0].plot(bbox_lon, bbox_lat, color='black', linewidth=2)

    # Right: h (interpolated grid)
    p1 = axs[1].contourf(lon_rho, lat_rho, h, levels=contour_bins, cmap=cmocean.cm.deep_r)
    fig.colorbar(p1, ax=axs[1], label='Depth (m)')
    axs[1].set_title('Interpolated Bathymetry (ROMS Grid)')
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Latitude')
    p1.set_clim(-6000, 0)
    axs[1].plot(bbox_lon, bbox_lat, color='black', linewidth=2)


    plt.tight_layout()
    plt.show()
    fig.savefig('../plots/bathy_smooth_vs_interpolated.png', dpi=300)

# Create mask (1 for ocean, 0 for land)
hmin = -5
mask_rho = np.where(h < -hmin, 1, 0)
# Create mask for u and v grids
mask_u = np.minimum(mask_rho[:, :-1], mask_rho[:, 1:])
mask_v = np.minimum(mask_rho[:-1, :], mask_rho[1:, :])

# fill h values where h is shallower than hmin (aka mask is 0) with 0 m depth 
h[h > hmin] = hmin
# check that all nans are replaced with 0
h = np.nan_to_num(h, nan=hmin)

if plot==1:
    # Plot the mask
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(lon_rho_grid, lat_rho_grid, mask_rho, cmap='gray', shading='auto')
    plt.colorbar(label='Mask (1=Ocean, 0=Land)')
    plt.title('Bathymetry Mask')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('../plots/bathymetry_mask.png', dpi=300)
    plt.show() 

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(lon_rho_grid, lat_rho_grid, np.where(mask_rho == 1, h, np.nan), cmap=cmocean.cm.deep_r, shading='auto')
    plt.colorbar(label='Depth (m)')
    plt.title('Interpolated Bathymetry')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.clim(-6000,0)
    plt.savefig('../plots/bathymetry_wmask.png', dpi=300)
    plt.show()

def compute_slope_factor(h, dx=1.0, dy=1.0, epsilon=1e-10):
    """
    Compute the slope factor r = |grad(h)|/h for a 2D array h.
    
    Parameters:
    -----------
    h : numpy.ndarray
        2D array representing the field (e.g., height, depth, etc.)
    dx : float, optional
        Grid spacing in x-direction (default: 1.0)
    dy : float, optional
        Grid spacing in y-direction (default: 1.0)
    epsilon : float, optional
        Small value to avoid division by zero (default: 1e-10)
    
    Returns:
    --------
    r : numpy.ndarray
        2D array of slope factors with same shape as input h
    """
    
    # Compute gradients using central differences
    grad_y, grad_x = np.gradient(h)
    grad_y /= dy  # Scale by grid spacing in y-direction
    grad_x /= dx  # Scale by grid spacing in x-direction
    
    # Compute magnitude of gradient
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Compute slope factor r = |grad(h)|/h
    r = grad_magnitude / (np.abs(h) + epsilon)

    return grad_y, grad_x, grad_magnitude, r


grad_y, grad_x, grad, r = compute_slope_factor(h, dx=dx_m, dy=dy_m)

if plot==1:
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(lon_rho_grid, lat_rho_grid, np.log10(r), cmap=cmocean.cm.amp, shading='auto')
    plt.colorbar(label='Log(r)')
    plt.title('Log of Slope Factor of Bathymetry')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.clim(-2, np.log10(0.2))
    plt.savefig('../plots/slope_factor_bathymetry.png', dpi=300)
    plt.show() 

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    pcm0 = axs[0].pcolormesh(lon_rho_grid, lat_rho_grid, grad_y / np.abs(h), cmap=cmocean.cm.amp, shading='auto')
    axs[0].set_title('grad_y / |h|')
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    fig.colorbar(pcm0, ax=axs[0], label='grad_y / |h|')
    pcm0.set_clim(0, 0.2)

    pcm1 = axs[1].pcolormesh(lon_rho_grid, lat_rho_grid, grad_x / np.abs(h), cmap=cmocean.cm.amp, shading='auto')
    axs[1].set_title('grad_x / |h|')
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Latitude')
    fig.colorbar(pcm1, ax=axs[1], label='grad_x / |h|')
    pcm1.set_clim(0, 0.2)

    plt.tight_layout()
    fig.savefig('../plots/slope_factor_components.png', dpi=300)
    plt.show()

r = np.nan_to_num(r, nan=0.0)
r_big_indy, r_big_indx = np.where(r>0.2)

if plot==1:
    fig, ax = plt.subplots(figsize=(8, 6))
    # Define contour bins (e.g., every 500 meters from -6000 to 0)
    contour_bins = np.arange(-6000, 500, 500)
    p = ax.contourf(lon_rho, lat_rho, h, levels=contour_bins, cmap=cmocean.cm.deep_r)
    ax.plot(lon_rho[r_big_indx], lat_rho[r_big_indy], 'ro', markersize=3, label='Slope > 0.2')
    ax.set_title('Smoothed Bathymetry Contours')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.colorbar(p, ax=ax, label='Depth (m)')
    plt.savefig('../plots/bathy_smoothing_contours_wrfactor.png', dpi=300)
    plt.show()

def compute_sigma(N, type='r'):
    if type=='r':
        k = np.arange(1, N + 1)
        sigma = (k - N - 0.5) / N
    elif type=='w':
        k = np.arange(N + 1)
        sigma = (k - N) / N
    else:
        raise ValueError("type must be 'r' for rho or 'w' for w coordinates")
    return sigma 

def compute_cs(sigma, theta_s, theta_b):
    C = (1 - np.cosh(theta_s * sigma)) / (np.cosh(theta_s) - 1)
    C = (np.exp(theta_b * C) - 1) / (1 - np.exp(-theta_b))
    return C

def compute_z(sigma, hc, C, h, zeta):
    # sigma: (nz,), C: (nz,), h: (ny, nx), zeta: (ny, nx, nt)
    nz = sigma.shape[0]
    ny, nx = h.shape
    nt = zeta.shape[2]

    # Expand sigma and C to (nz, 1, 1, 1)
    sigma = sigma[:, None, None, None]  # (nz, 1, 1, 1)
    C = C[:, None, None, None]          # (nz, 1, 1, 1)

    # Expand h to (1, ny, nx, 1)
    h = h[None, :, :, None]             # (1, ny, nx, 1)

    # Expand zeta to (1, ny, nx, nt)
    zeta = zeta[None, :, :, :]          # (1, ny, nx, nt)

    # Compute z0: (nz, ny, nx, nt)
    z0 = (hc * sigma + C * h) / (hc + h)
    z = zeta + (zeta + h) * z0
    return z  # shape: (nz, ny, nx, nt)


# Vertical levels (sigma coordinates) 
sigma_w = compute_sigma(N, type='w')
sigma_r = compute_sigma(N, type='r')

theta_s = 5 
theta_b = 0.5 

Cs_w = compute_cs(sigma_w, theta_s, theta_b)
Cs_r = compute_cs(sigma_r, theta_s, theta_b)

hc = -500 

h_mask = np.where(mask_rho == 1, h, np.nan)
z_rho = compute_z(sigma_r, hc, Cs_r, h_mask, np.zeros((ny, nx, 1)))  
z_rho = np.squeeze(z_rho)  # shape: (nz, ny, nx)

# Plot vertical coordinate system at several longitude cross sections
cross_lons = np.linspace(-75, lon_max, 5)  # Choose 5 cross-section longitudes
fig, axs = plt.subplots(1, len(cross_lons), figsize=(18, 5), sharey=True)

for i, lon_val in enumerate(cross_lons):
    # Find nearest longitude index
    idx = np.abs(lon_rho - lon_val).argmin()
    # Extract vertical coordinate (z_rho) and bathymetry (h) at this longitude
    z_section = z_rho[:, :, idx]  # shape: (nz, ny)
    # Plot bathymetry along the cross section
    h_section = h[:, idx]         # shape: (ny,)
    print(idx, np.min(z_section), np.max(z_section) )
    lat_section = lat_rho_grid[:, idx]
    axs[i].plot(lat_section, -h_section, 'k--', label='Bathymetry')
    # Plot vertical layers as curves
    for k in range(len(sigma_r)):
        axs[i].plot(lat_section, z_section[k, :], color='blue', alpha=0.5, linewidth=0.8)
    # Plot
    axs[i].set_title(f'Lon={lon_rho[idx]:.2f}')
    axs[i].set_xlabel('Latitude')
    if i == 0:
        axs[i].set_ylabel('Sigma Level')
    axs[i].invert_yaxis()
    axs[i].legend()

fig.suptitle('Vertical Coordinate System at Longitude Cross Sections')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('../plots/vertical_coords_cross_sections.png', dpi=300)
plt.show()

if plot==1:
    fig, ax = plt.subplots(figsize=(8, 6))
    cross_lons = np.linspace(-75, lon_max, 5)  # Choose 5 cross-section longitudes
    k = 2 
    idx = np.abs(lon_rho - cross_lons[k]).argmin()
    # Extract vertical coordinate (z_rho) and bathymetry (h) at this longitude
    z_section = z_rho[:, :, idx]  # shape: (nz, ny)
    # Plot bathymetry along the cross section
    h_section = h[:, idx]         # shape: (ny,)
    print(idx, np.min(z_section), np.max(z_section) )
    lat_section = lat_rho_grid[:, idx]
    ax.plot(lat_section, -h_section, 'k--', label='Bathymetry')  
    for k in range(len(sigma_r)):
        ax.plot(lat_section, z_section[k, :], color='blue', alpha=0.5, linewidth=0.8)
    ax.invert_yaxis()
    ax.legend()    
    fig.savefig('../plots/vertical_coords_cross_section.png', dpi=300)


if plot==1:
    plt.figure()
    plt.plot(Cs_r, 'o-')
    plt.plot(Cs_w, 'o-')
    plt.legend(['Cs_r', 'Cs_w'])
    plt.title('Vertical Levels')
    plt.savefig('../plots/s_rho_levels.png', dpi=300)

if plot==1:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
    ax.set_position([0.08, 0.02, 0.8, 0.9])
    p = ax.pcolormesh(
        lon_rho_grid, lat_rho_grid, h,
        transform=ccrs.PlateCarree(),
        cmap=cmocean.cm.deep_r, zorder=0
    )
    p.set_clim(-6000, 0)
    cbar = fig.colorbar(p, ax=ax, label='Bathymetry (m)', orientation='vertical')
    cbar.ax.set_position([0.82, 0.15, 0.02, 0.7])
    ax.add_feature(cfeature.LAND, zorder=1, facecolor='grey')
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    plt.title('ROMS Grid Bathymetry (Lambert Conformal)')
    plt.savefig('../plots/grid_bathymetry_lambert.png', dpi=300)
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
