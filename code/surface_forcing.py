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

era5 = xr.open_dataset(os.path.join(forcing_datapath, 'ERA5_data.nc'))
era5_pair = xr.open_dataset(os.path.join(forcing_datapath, 'ERA5_Pair.nc'))
grid = xr.open_dataset(os.path.join(base_path, 'output', grid_nc))

def compute_relative_humidity(t_air, t_dew):
    """
    Compute surface air relative humidity given 2-meter air temperature and dew point.
    
    Parameters:
        t_air (float): 2-meter air temperature in Kelvin
        t_dew (float): 2-meter dew point temperature in Kelvin
        
    Returns:
        float: Relative humidity in percentage
    """
    # Convert temperatures from Kelvin to Celsius
    t_air_c = t_air - 273.15
    t_dew_c = t_dew - 273.15
    
    # Calculate saturation vapor pressure for air temperature (in hPa)
    e_sat_air = 6.112 * np.exp((17.67 * t_air_c) / (t_air_c + 243.5))
    
    # Calculate saturation vapor pressure for dew point temperature (in hPa)
    e_sat_dew = 6.112 * np.exp((17.67 * t_dew_c) / (t_dew_c + 243.5))
    
    # Calculate relative humidity
    rh = (e_sat_dew / e_sat_air) * 100  # RH in percentage
    
    return rh

def convert_to_energy_density(radiation_wm2, time_interval):
    """
    Converts radiation from W/m² to J/m² given a time interval.
    
    Parameters:
        radiation_wm2 (array-like): Radiation values in W/m² (can be 1D, 2D, or 3D: [nt, ny, nx]).
        time_interval (float): Time interval in seconds over which the radiation is measured.
        
    Returns:
        array-like: Radiation values converted to J/m², same shape as input.
    """
    # Element-wise multiplication, works for arrays of any shape
    energy_density = np.asarray(radiation_wm2) * time_interval
    return energy_density

def calculate_surface_wind(u10_value, z0=0.0002, z_ref=10):
    """    Calculate surface wind speed at a reference height using the logarithmic wind profile.   
    
    Parameters:
        u10_value (float): Wind speed at 10 meters above ground level (m/s).
        z0 (float): Surface roughness length (m), default is 0.0002 m.
        z_ref (float): Reference height for the wind speed (m), default is 10 m.
    Returns:
        float: Wind speed at the reference height (m/s)."""
    return u10_value * (np.log(z0) / np.log(z_ref / z0))

def convert_Pa_to_mbar(pressure_pa):
    """
    Convert pressure from Pascals to millibars.
    
    Parameters:
        pressure_pa (float or array): Pressure in Pascals.
        
    Returns:
        float or array: Pressure in millibars.
    """
    return pressure_pa / 100.0  # 1 mbar = 100 Pa   

def convert_K_to_C(temperature_k):
    """
    Convert temperature from Kelvin to Celsius.
    
    Parameters:
        temperature_k (float or array): Temperature in Kelvin.
        
    Returns:
        float or array: Temperature in Celsius.
    """
    return temperature_k - 273.15  # K to C conversion  

def compute_rainfall_cm_per_day(total_precipitation_m, time_interval_seconds):
    """
    Convert total precipitation from meters to rainfall in cm/day.
    
    Parameters:
        total_precipitation_m (list or array): Total precipitation time series in meters.
        time_interval_seconds (float): Time interval of the data in seconds (e.g., 3600 for hourly data).
        
    Returns:
        list: Rainfall rate in cm/day.
    """
    # Seconds in a day
    seconds_per_day = 86400  # 24 hours * 60 minutes * 60 seconds
    
    # Convert each value in total precipitation from meters to cm/day
    rainfall_cm_per_day = [
        (precip * seconds_per_day) / time_interval_seconds * 100  # Convert m to cm
        for precip in total_precipitation_m
    ]
    
    return rainfall_cm_per_day

time = era5['time']

# Compute time delta in seconds from time variable
dt = (time[1] - time[0]).astype('timedelta64[s]').values

## swrad
swrad = convert_to_energy_density(era5['ssr'].values, dt)

## lwrad 
lwrad = convert_to_energy_density(era5['strd'].values, dt) 

## Tair 
Tair = convert_K_to_C(era5['sst'].values)

## qair 
qair = compute_relative_humidity(era5['t2m'].values, era5['d2m'].values)

## Pair
Pair = convert_Pa_to_mbar(era5_pair['msl'].values)

## rain 
rain = compute_rainfall_cm_per_day(era5['tp'].values, dt.astype(float))

## wind 
uwnd = calculate_surface_wind(era5['u10'].values)
vwnd = calculate_surface_wind(era5['v10'].values)

## time
ref_time = np.datetime64('2000-01-01T00:00:00')
relative_time_days = (time.values - ref_time) / np.timedelta64(1, 'D')

# === Interpolate computed surface forcing variables to ROMS grid ===
# Get ERA5 source grid
era5_lat = era5['latitude'].values
era5_lon = era5['longitude'].values
era5_lon_2d, era5_lat_2d = np.meshgrid(era5_lon, era5_lat, indexing='xy')

# Get ROMS target grid
roms_lat_2d = grid['lat_rho'].values
roms_lon_2d = grid['lon_rho'].values

# Interpolation tool
interp = interp_tools()

# Interpolate each variable to ROMS grid

# Interpolate each variable for every time index
nt = swrad.shape[0]
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
    swrad_interp[t] = interp.interp2d(swrad[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    lwrad_interp[t] = interp.interp2d(lwrad[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    Tair_interp[t] = interp.interp2d(Tair[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    qair_interp[t] = interp.interp2d(qair[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    Pair_interp[t] = interp.interp2d(Pair[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    rain_interp[t] = interp.interp2d(rain[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    uwnd_interp[t] = interp.interp2d(uwnd[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    vwnd_interp[t] = interp.interp2d(vwnd[t], era5_lon_2d, era5_lat_2d, roms_lon_2d, roms_lat_2d, method='linear')
    if np.mod(t, 100) == 0:
        print(f"Interpolated time step {t+1}/{nt}")

# === Write interpolated surface forcing variables to NetCDF using xarray ===

# Prepare coordinates
eta_rho, xi_rho = ny_rho, nx_rho
output_file = os.path.join(base_path, 'output', 'surface_forcing.nc')

ds = xr.Dataset(
    {
        'swrad': (['time', 'eta_rho', 'xi_rho'], swrad_interp, {
            '_FillValue': np.nan,
            'long_name': 'downward short-wave (solar) radiation',
            'units': 'W/m^2',
            'coordinates': 'abs_time expver number',
        }),
        'lwrad': (['time', 'eta_rho', 'xi_rho'], lwrad_interp, {
            '_FillValue': np.nan,
            'long_name': 'downward long-wave (thermal) radiation',
            'units': 'W/m^2',
            'coordinates': 'abs_time expver number',
        }),
        'Tair': (['time', 'eta_rho', 'xi_rho'], Tair_interp, {
            '_FillValue': np.nan,
            'long_name': 'air temperature at 2m',
            'units': 'degrees Celsius',
            'coordinates': 'abs_time expver number',
        }),
        'qair': (['time', 'eta_rho', 'xi_rho'], qair_interp, {
            '_FillValue': np.nan,
            'long_name': 'absolute humidity at 2m',
            'units': 'kg/kg',
            'coordinates': 'abs_time expver number',
        }),
        'rain': (['time', 'eta_rho', 'xi_rho'], rain_interp, {
            '_FillValue': np.nan,
            'long_name': 'total precipitation',
            'units': 'cm/day',
            'coordinates': 'abs_time expver number',
        }),
        'uwnd': (['time', 'eta_rho', 'xi_rho'], uwnd_interp, {
            '_FillValue': np.nan,
            'long_name': '10 meter wind in x-direction',
            'units': 'm/s',
            'coordinates': 'abs_time expver number',
        }),
        'vwnd': (['time', 'eta_rho', 'xi_rho'], vwnd_interp, {
            '_FillValue': np.nan,
            'long_name': '10 meter wind in y-direction',
            'units': 'm/s',
            'coordinates': 'abs_time expver number',
        }),
        'Pair': (['time', 'eta_rho', 'xi_rho'], Pair_interp, {
            '_FillValue': np.nan,
            'long_name': 'mean sea level pressure',
            'units': 'mbar',
            'coordinates': 'abs_time expver number',
        }),
        'time': (['time'], relative_time_days, {
            '_FillValue': np.nan,
            'long_name': 'relative time: days since 2000-01-01 00:00:00',
            'units': 'days',
        }),
    },
    coords={
        'time': ('time', np.arange(nt)),
        'eta_rho': ('eta_rho', np.arange(eta_rho)),
        'xi_rho': ('xi_rho', np.arange(xi_rho)),
    },
    attrs={
        'title': 'ROMS surface forcing file created by model-tools',
        'start_time': str(time[0].values)[:10] + ' 00:00:00',
        'end_time': str(time[-1].values)[:10] + ' 00:00:00',
        'source': 'ERA5',
        'model_reference_date': '2000-01-01 00:00:00',
    }
)

# Write to NetCDF
ds.to_netcdf(output_file, format='NETCDF4', encoding={var: {'_FillValue': np.nan} for var in ds.data_vars})
print(f"Surface forcing NetCDF file written: {output_file}")


