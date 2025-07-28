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

# Load the grid 
grid = xr.open_dataset(os.path.join(base_path, 'output', grid_nc))

# Load initialization data 
init_data = xr.open_dataset(os.path.join(forcing_datapath, 'GLORYS_data', 'GLORYS_January.nc'))

# === CREATE INITIAL CONDITIONS FILE ===
init_time = np.datetime64('2024-01-01T00:00:00')
time_idx = np.argmin(np.abs(init_data.time.values - init_time))
# Compute time in seconds since 2000-01-01 00:00:00
ref_time = np.datetime64('2000-01-01T00:00:00')
seconds_since_2000 = (init_time - ref_time) / np.timedelta64(1, 's')
print(f"Seconds since 2000-01-01 00:00:00: {seconds_since_2000:.6e}")

# Select first time step from GLORYS data
glorys_time_step = init_data.isel(time=time_idx)

# Get grid dimensions
eta_rho, xi_rho = grid.lat_rho.shape
eta_v, _ = grid.lat_v.shape
_, xi_u = grid.lat_u.shape
s_rho = len(grid.s_rho)
s_w = len(grid.s_w)
n_depth = len(init_data.depth)

# Prepare GLORYS source coordinates
# Note: GLORYS depths are typically positive downward, ROMS depths are negative upward
glorys_depth = init_data.depth.values  # Assuming positive depths
glorys_lat = init_data.latitude.values
glorys_lon = init_data.longitude.values

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

# Create interpolation tool instance
interp = interp_tools()

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


# Interpolate temperature using interp3d (no mask, no NaN checks)
# Output: (nz_tgt, eta_rho, xi_rho)
temp_interp = interp.interp3d(
    temp_data, glorys_lon_2d, glorys_lat_2d, glorys_depth, roms_lon_2d, roms_lat_2d, -z_rho, method='linear'
)

zind, yind, xind = np.where(temp_interp<0)
tmin = 0.1
temp_interp[zind, yind, xind] = tmin  # Set negative values to minimum temperature

# Mask temp_interp with grid.mask_rho (1=water, 0=land)
mask = grid.mask_rho.values.astype(bool)
temp_interp = np.where(mask, temp_interp, np.nan)

print(f"Interpolated temperature shape: {temp_interp.shape}")

# Plotting: Planar temperature maps at surface, mid-depth, and bottom
if 1:
    # === Plot multiple cross sections (source vs interpolated) ===

    cross_lats = [36, 38, 40, 42]  # Example latitudes for cross sections

    fig, axes = plt.subplots(len(cross_lats), 2, figsize=(14, 3 * len(cross_lats)))

    for i, lat_val in enumerate(cross_lats):
        # Find nearest latitude index in GLORYS and ROMS
        glorys_lat_idx = np.argmin(np.abs(glorys_lat - lat_val))
        roms_lat_idx = np.argmin(np.abs(roms_lat_2d[:, 0] - lat_val))

        # Source cross section
        cross_temp_glorys = temp_data[:, glorys_lat_idx, :]  # (depth, lon)
        pcm_src = axes[i, 0].pcolormesh(
            glorys_lon, glorys_depth, cross_temp_glorys, cmap='cmo.thermal', shading='auto'
        )
        axes[i, 0].set_ylim(0, 6000)
        axes[i, 0].invert_yaxis()
        axes[i, 0].set_title(f'GLORYS Cross Section (lat={lat_val}°)')
        axes[i, 0].set_xlabel('Longitude')
        axes[i, 0].set_ylabel('Depth (m)')
        pcm_src.set_clim(0,30)
        axes[i, 0].set_xlim(-80, -45)
        fig.colorbar(pcm_src, ax=axes[i, 0], label='Temperature (°C)')

        # Interpolated cross section
        zz = -z_rho[roms_lat_idx, :, :].T  # (s_rho, xi_rho)
        lonlon = np.tile(roms_lon_2d[roms_lat_idx, :], (z_rho.shape[2], 1))  # (s_rho, xi_rho)
        cross_temp_interp = temp_interp[:, roms_lat_idx, :]  # (s_rho, xi_rho)
        pcm_int = axes[i, 1].pcolormesh(
            lonlon, zz, cross_temp_interp, cmap='cmo.thermal', shading='auto'
        )
        axes[i, 1].set_ylim(0, 6000)
        axes[i, 1].invert_yaxis()
        axes[i, 1].set_title(f'Interpolated Cross Section (lat={lat_val}°, ROMS)')
        axes[i, 1].set_xlabel('Longitude')
        axes[i, 1].set_ylabel('Depth (m)')
        pcm_int.set_clim(0,30)
        axes[i, 1].set_xlim(-80, -45)
        fig.colorbar(pcm_int, ax=axes[i, 1], label='Temperature (°C)')

    plt.tight_layout()
    fig.savefig('../plots/temperature_cross_sections.png', dpi=300, bbox_inches='tight')

# === SALINITY INTERPOLATION AND PLOTTING ===
sal_data = glorys_time_step.so.values  # Shape: (n_depth, n_lat, n_lon)
if sal_data.shape[0] != len(glorys_depth):
    sal_data = np.transpose(sal_data, (2, 0, 1))

print("Interpolating salinity with interp3d...")
sal_interp = interp.interp3d(
    sal_data, glorys_lon_2d, glorys_lat_2d, glorys_depth, roms_lon_2d, roms_lat_2d, -z_rho, method='linear'
)

# Mask sal_interp with grid.mask_rho (1=water, 0=land)
sal_interp = np.where(mask, sal_interp, np.nan)

print(f"Interpolated salinity shape: {sal_interp.shape}")

# Plotting: Planar salinity maps at surface, mid-depth, and bottom
if 1:
    cross_lats = [36, 38, 40, 42]
    fig, axes = plt.subplots(len(cross_lats), 2, figsize=(14, 3 * len(cross_lats)))
    for i, lat_val in enumerate(cross_lats):
        glorys_lat_idx = np.argmin(np.abs(glorys_lat - lat_val))
        roms_lat_idx = np.argmin(np.abs(roms_lat_2d[:, 0] - lat_val))
        # Source cross section
        cross_sal_glorys = sal_data[:, glorys_lat_idx, :]
        pcm_src = axes[i, 0].pcolormesh(
            glorys_lon, glorys_depth, cross_sal_glorys, cmap='cmo.haline', shading='auto'
        )
        axes[i, 0].set_ylim(0, 6000)
        axes[i, 0].invert_yaxis()
        axes[i, 0].set_title(f'GLORYS Salinity Cross Section (lat={lat_val}°)')
        axes[i, 0].set_xlabel('Longitude')
        axes[i, 0].set_ylabel('Depth (m)')
        pcm_src.set_clim(33, 37)
        axes[i, 0].set_xlim(-80, -45)
        fig.colorbar(pcm_src, ax=axes[i, 0], label='Salinity (psu)')
        # Interpolated cross section
        zz = -z_rho[roms_lat_idx, :, :].T
        lonlon = np.tile(roms_lon_2d[roms_lat_idx, :], (z_rho.shape[2], 1))
        cross_sal_interp = sal_interp[:, roms_lat_idx, :]
        pcm_int = axes[i, 1].pcolormesh(
            lonlon, zz, cross_sal_interp, cmap='cmo.haline', shading='auto'
        )
        axes[i, 1].set_ylim(0, 6000)
        axes[i, 1].invert_yaxis()
        axes[i, 1].set_title(f'Interpolated Salinity Cross Section (lat={lat_val}°, ROMS)')
        axes[i, 1].set_xlabel('Longitude')
        axes[i, 1].set_ylabel('Depth (m)')
        pcm_int.set_clim(33, 37)
        axes[i, 1].set_xlim(-80, -45)
        fig.colorbar(pcm_int, ax=axes[i, 1], label='Salinity (psu)')
    plt.tight_layout()
    fig.savefig('../plots/salinity_cross_sections.png', dpi=300, bbox_inches='tight')

# === U AND V VELOCITY INTERPOLATION (STUB) ===
if 'uo' in glorys_time_step and 'vo' in glorys_time_step:
    print("Interpolating u and v velocities with interp3d...")
    u_data = glorys_time_step.uo.values
    v_data = glorys_time_step.vo.values
    if u_data.shape[0] != len(glorys_depth):
        u_data = np.transpose(u_data, (2, 0, 1))
    if v_data.shape[0] != len(glorys_depth):
        v_data = np.transpose(v_data, (2, 0, 1))

    # Get ROMS target coordinates
    roms_latu_2d = grid.lat_u.values  
    roms_lonu_2d = grid.lon_u.values  

    roms_latv_2d = grid.lat_v.values  
    roms_lonv_2d = grid.lon_v.values  

    u_interp = interp.interp3d(
        u_data, glorys_lon_2d, glorys_lat_2d, glorys_depth, roms_lonu_2d, roms_latu_2d, -z_rho, method='linear'
    )
    v_interp = interp.interp3d(
        v_data, glorys_lon_2d, glorys_lat_2d, glorys_depth, roms_lonv_2d, roms_latv_2d, -z_rho, method='linear'
    )
    # Mask with grid
    mask_u = grid.mask_u.values.astype(bool)
    mask_v = grid.mask_v.values.astype(bool)
    u_interp = np.where(mask_u, u_interp, np.nan)
    v_interp = np.where(mask_v, v_interp, np.nan)
    print(f"Interpolated u velocity shape: {u_interp.shape}")
    print(f"Interpolated v velocity shape: {v_interp.shape}")
    # Plotting code for u/v can be added similarly
    cross_lats = [36, 38, 40, 42]
    fig, axes = plt.subplots(len(cross_lats), 4, figsize=(28, 3 * len(cross_lats)))
    for i, lat_val in enumerate(cross_lats):
        glorys_lat_idx = np.argmin(np.abs(glorys_lat - lat_val))
        roms_lat_u_idx = np.argmin(np.abs(roms_latu_2d[:, 0] - lat_val))
        roms_lat_v_idx = np.argmin(np.abs(roms_latv_2d[:, 0] - lat_val))

        # GLORYS u cross section
        cross_u_glorys = u_data[:, glorys_lat_idx, :]
        pcm_u_src = axes[i, 0].pcolormesh(
            glorys_lon, glorys_depth, cross_u_glorys, cmap=cmocean.cm.balance, shading='auto'
        )
        axes[i, 0].set_ylim(0, 6000)
        axes[i, 0].invert_yaxis()
        axes[i, 0].set_title(f'GLORYS U Cross Section (lat={lat_val}°)')
        axes[i, 0].set_xlabel('Longitude')
        axes[i, 0].set_ylabel('Depth (m)')
        pcm_u_src.set_clim(-1, 1)
        axes[i, 0].set_xlim(-80, -45)
        fig.colorbar(pcm_u_src, ax=axes[i, 0], label='U Velocity (m/s)')

        # Interpolated u cross section
        # Interpolate z_rho to u-grid for correct vertical coordinates
        # Average z_rho between adjacent rho points in xi direction
        z_rho_u = 0.5 * (z_rho[:, :-1, :] + z_rho[:, 1:, :])  # shape: (eta_rho, xi_rho-1, s_rho)
        zz_u = -z_rho_u[roms_lat_u_idx, :, :].T  # shape: (s_rho, xi_u)
        lon_u = np.tile(roms_lonu_2d[roms_lat_u_idx, :], (z_rho.shape[2], 1))
        cross_u_interp = u_interp[:, roms_lat_u_idx, :]
        pcm_u_int = axes[i, 1].pcolormesh(
            lon_u, zz_u, cross_u_interp, cmap=cmocean.cm.balance, shading='auto'
        )
        axes[i, 1].set_ylim(0, 6000)
        axes[i, 1].invert_yaxis()
        axes[i, 1].set_title(f'Interpolated U Cross Section (lat={lat_val}°, ROMS)')
        axes[i, 1].set_xlabel('Longitude')
        axes[i, 1].set_ylabel('Depth (m)')
        pcm_u_int.set_clim(-1, 1)
        axes[i, 1].set_xlim(-80, -45)
        fig.colorbar(pcm_u_int, ax=axes[i, 1], label='U Velocity (m/s)')

        # GLORYS v cross section
        cross_v_glorys = v_data[:, glorys_lat_idx, :]
        pcm_v_src = axes[i, 2].pcolormesh(
            glorys_lon, glorys_depth, cross_v_glorys, cmap=cmocean.cm.balance, shading='auto'
        )
        axes[i, 2].set_ylim(0, 6000)
        axes[i, 2].invert_yaxis()
        axes[i, 2].set_title(f'GLORYS V Cross Section (lat={lat_val}°)')
        axes[i, 2].set_xlabel('Longitude')
        axes[i, 2].set_ylabel('Depth (m)')
        pcm_v_src.set_clim(-1, 1)
        axes[i, 2].set_xlim(-80, -45)
        fig.colorbar(pcm_v_src, ax=axes[i, 2], label='V Velocity (m/s)')

        # Interpolated v cross section
        # Interpolate z_rho to v-grid for correct vertical coordinates
        # Average z_rho between adjacent rho points in eta direction
        z_rho_v = 0.5 * (z_rho[:-1, :, :] + z_rho[1:, :, :])  # shape: (eta_rho-1, xi_rho, s_rho)
        zz_v = -z_rho_v[roms_lat_v_idx, :, :].T  # shape: (s_rho, xi_v)
        lon_v = np.tile(roms_lonv_2d[roms_lat_v_idx, :], (z_rho.shape[2], 1))
        cross_v_interp = v_interp[:, roms_lat_v_idx, :]
        pcm_v_int = axes[i, 3].pcolormesh(
            lon_v, zz_v, cross_v_interp, cmap=cmocean.cm.balance, shading='auto'
        )
        axes[i, 3].set_ylim(0, 6000)
        axes[i, 3].invert_yaxis()
        axes[i, 3].set_title(f'Interpolated V Cross Section (lat={lat_val}°, ROMS)')
        axes[i, 3].set_xlabel('Longitude')
        axes[i, 3].set_ylabel('Depth (m)')
        pcm_v_int.set_clim(-1, 1)
        axes[i, 3].set_xlim(-80, -45)
        fig.colorbar(pcm_v_int, ax=axes[i, 3], label='V Velocity (m/s)')

    plt.tight_layout()
    fig.savefig('../plots/uv_cross_sections.png', dpi=300, bbox_inches='tight')


output_nc = os.path.join(base_path, 'output', 'initial_conditions.nc')
ds = xr.Dataset(
    {
        'temp': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), temp_interp[np.newaxis, :, :, :]),
        'salt': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), sal_interp[np.newaxis, :, :, :]),
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