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
 
base_path = '/global/cfs/cdirs/m4304/enuss/model-tools'
grid_nc = 'roms_grid_1km_smoothed.nc' 
forcing_datapath = '/global/cfs/cdirs/m4304/enuss/US_East_Coast/Forcing_Data'

era5 = xr.open_dataset(os.path.join(forcing_datapath, 'ERA5_data.nc'))
era5_pair = xr.open_dataset(os.path.join(forcing_datapath, 'ERA5_Pair.nc'))
era5_rad = xr.open_dataset(os.path.join(forcing_datapath, 'ERA5_rad.nc'))
grid = xr.open_dataset(os.path.join(base_path, 'output', grid_nc))

## Set variable names 
time = 'time'
shortwave_net = 'ssr'
shortwave = 'ssrd'
longwave_net = 'str'
longwave = 'strd'
sst = 'sst'
airtemp = 't2m'
dewpoint = 'd2m'
precip = 'tp'
u10 = 'u10'
v10 = 'v10'
press = 'msl'

## time
time = era5['time']
start_time = np.datetime64('2024-12-01T00:00:00')
end_time = np.datetime64('2025-01-01T00:00:00')
time = time[(time >= start_time) & (time <= end_time)]
ref_time = np.datetime64('2000-01-01T00:00:00')
relative_time_days = (time.values - ref_time) / np.timedelta64(1, 'D')

output_file = os.path.join(base_path, 'output', 'surface_forcing_1km_dec.nc')

# Compute time delta in seconds from time variable
dt = (time[1] - time[0]).astype('timedelta64[s]').values
dt = dt.astype(float)

swdn = convert_tools.convert_to_flux_density(era5_rad[shortwave].values, dt)
swrad = swdn - swdn*0.06
lwrad = convert_tools.convert_to_flux_density(era5_rad[longwave_net].values, dt)
Tair = convert_tools.convert_K_to_C(era5[sst].values) 
Tair = np.nan_to_num(Tair, nan=np.nanmean(Tair))
qair = convert_tools.compute_relative_humidity(era5[airtemp].values, era5[dewpoint].values)
Pair = convert_tools.convert_Pa_to_mbar(era5_pair[press].values)
rain_rate = convert_tools.compute_rainfall_cm_per_day(era5[precip].values, dt.astype(float))
rain = [rate * 0.01 / 86400 for rate in rain_rate] # Convert cm/day to m/s
uwnd = convert_tools.calculate_surface_wind(era5[u10].values)
vwnd = convert_tools.calculate_surface_wind(era5[v10].values)

# === Interpolate computed surface forcing variables to ROMS grid ===
# Get ERA5 source grid
era5_lat = era5['latitude'].values
era5_lon = era5['longitude'].values
era5_lon_2d, era5_lat_2d = np.meshgrid(era5_lon, era5_lat, indexing='xy')

# Get ROMS target grid
roms_lat_2d = grid['lat_rho'].values
roms_lon_2d = grid['lon_rho'].values

# Interpolate each variable to ROMS grid
# Interpolate each variable for every time index
nt = len(time)
ny_rho, nx_rho = roms_lat_2d.shape

swrad_interp = np.empty((nt, ny_rho, nx_rho))
lwrad_interp = np.empty((nt, ny_rho, nx_rho))
Tair_interp = np.empty((nt, ny_rho, nx_rho))
qair_interp = np.empty((nt, ny_rho, nx_rho))
Pair_interp = np.empty((nt, ny_rho, nx_rho))
rain_interp = np.empty((nt, ny_rho, nx_rho))
uwnd_interp = np.empty((nt, ny_rho, nx_rho))
vwnd_interp = np.empty((nt, ny_rho, nx_rho))

for t in range(nt):
    swrad_interp[t] = interp_tools.interp2d(swrad[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    lwrad_interp[t] = interp_tools.interp2d(lwrad[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    Tair_interp[t] = interp_tools.interp2d(Tair[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    qair_interp[t] = interp_tools.interp2d(qair[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    Pair_interp[t] = interp_tools.interp2d(Pair[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    rain_interp[t] = interp_tools.interp2d(rain[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    uwnd_interp[t] = interp_tools.interp2d(uwnd[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    vwnd_interp[t] = interp_tools.interp2d(vwnd[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    if np.mod(t, 100) == 0:
        print(f"Interpolated time step {t+1}/{nt}")

time_days_since_2000 = (time.values - ref_time) / np.timedelta64(1, 'D')

# === Write interpolated surface forcing variables to NetCDF using xarray ===
# Prepare coordinates
eta_rho, xi_rho = ny_rho, nx_rho

ds = xr.Dataset(
    {
        'swrad': (['wind_time', 'eta_rho', 'xi_rho'], swrad_interp, {
            'long_name': 'net short-wave (solar) radiation',
            'units': 'W/m^2',
        }),
        'lwrad': (['wind_time', 'eta_rho', 'xi_rho'], lwrad_interp, {
            'long_name': 'net long-wave (thermal) radiation',
            'units': 'W/m^2',
        }),
        'Tair': (['wind_time', 'eta_rho', 'xi_rho'], Tair_interp, {
            'long_name': 'air temperature at 2m',
            'units': 'degrees Celsius',
        }),
        'qair': (['wind_time', 'eta_rho', 'xi_rho'], qair_interp, {
            'long_name': 'absolute humidity at 2m',
            'units': 'kg/kg',
        }),
        'rain': (['wind_time', 'eta_rho', 'xi_rho'], rain_interp, {
            'long_name': 'total precipitation',
            'units': 'cm/day',
        }),
        'Uwind': (['wind_time', 'eta_rho', 'xi_rho'], uwnd_interp, {
            'long_name': '10 meter wind in x-direction',
            'units': 'm/s',
        }),
        'Vwind': (['wind_time', 'eta_rho', 'xi_rho'], vwnd_interp, {
            'long_name': '10 meter wind in y-direction',
            'units': 'm/s',
        }),
        'Pair': (['wind_time', 'eta_rho', 'xi_rho'], Pair_interp, {
            'long_name': 'mean sea level pressure',
            'units': 'mbar',
        }),
        'bhflux': (['wind_time', 'eta_rho', 'xi_rho'], np.zeros((nt, eta_rho, xi_rho)), {
            'long_name': 'bottom heat flux',
            'units': 'W/m^2',
        }), 
        'EminusP': (['wind_time', 'eta_rho', 'xi_rho'], np.zeros((nt, eta_rho, xi_rho)), {
            'long_name': 'surface upward freshwater flux (evaporation minus precipitation)',
            'units': 'm/s',
        }),
        'bwflux': (['wind_time', 'eta_rho', 'xi_rho'], np.zeros((nt, eta_rho, xi_rho)), {
            'long_name': 'bottom freshwater flux',
            'units': 'm/s',
        }),
    },
    coords={
        'wind_time': ('wind_time', time_days_since_2000, {
            'units': 'days since 2000-01-01 00:00:00',
        }),
        'eta_rho': ('eta_rho', np.arange(eta_rho)),
        'xi_rho': ('xi_rho', np.arange(xi_rho)),
    },
    attrs={
        'title': 'ROMS surface forcing file created by model-tools',
        'start_time': str(time[0].values)[:10] + ' 00:00:00',
        'end_time': str(time[-1].values)[:10] + ' 00:00:00',
        'source': 'ERA5',
        'model_reference_date': str(ref_time),
    }
)

# Write to NetCDF 
ds.to_netcdf(output_file, format='NETCDF4', encoding={var: {'_FillValue': np.nan} for var in ds.data_vars})
print(f"Surface forcing NetCDF file written: {output_file}")


