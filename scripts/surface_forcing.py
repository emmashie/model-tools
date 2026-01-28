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
from forcing import forcing_tools
import xarray as xr
import numpy as np
import os

# ============================================================================
# LOCATION-SPECIFIC PARAMETERS - Customize these for your domain
# ============================================================================

base_path = '/global/cfs/cdirs/m4304/enuss/model-tools'
grid_nc = 'roms_grid_1km_smoothed.nc' 
forcing_datapath = '/global/cfs/cdirs/m4304/enuss/US_East_Coast/Forcing_Data'

# ============================================================================
# DATA SOURCE CONFIGURATION
# Choose between NetCDF files or Copernicus Climate Data Store API
# ============================================================================

# Set to True to use CDS API, False to use local NetCDF files
USE_API = True

# If USE_API is False, specify paths to existing NetCDF files
# Can be a single file or dictionary for multiple files
NETCDF_PATHS = {
    'main': os.path.join(forcing_datapath, 'ERA5_data.nc'),
    'pair': os.path.join(forcing_datapath, 'ERA5_Pair.nc'),
    'rad': os.path.join(forcing_datapath, 'ERA5_rad.nc')
}
# Or use a single file: NETCDF_PATHS = 'era5_forcing.nc'

# If USE_API is True, these parameters define the download
API_LON_RANGE = None  # e.g., (-80.0, -60.0) or None to auto-detect from grid
API_LAT_RANGE = None  # e.g., (30.0, 50.0) or None to auto-detect from grid
API_HOURS = ['00:00', '06:00', '12:00', '18:00']  # Time resolution, None = all hours
API_INCLUDE_RADIATION = True  # Include radiation variables

# Time range for forcing
start_time = np.datetime64('2024-01-01T00:00:00')
end_time = np.datetime64('2024-01-02T00:00:00')

# Output file
output_file = os.path.join(base_path, 'output', 'surface_forcing_api.nc')

# ERA5 variable names
era5_vars = {
    'shortwave': 'ssrd',
    'shortwave_net' : 'ssrd',
    'longwave': 'strd',
    'longwave_net' : 'str',
    'sst': 'sst',
    'airtemp': 't2m',
    'dewpoint': 'd2m',
    'precip': 'tp',
    'u10': 'u10',
    'v10': 'v10',
    'press': 'msl'
}

# ============================================================================
# LOAD DATA
# ============================================================================

# Load grid
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

# Load ERA5 forcing data
if USE_API:
    print("\n=== Loading data via Copernicus Climate Data Store API ===")
    era5_data = forcing_tools.load_era5_data(
        time_range=(start_time, end_time),
        lon_range=API_LON_RANGE,
        lat_range=API_LAT_RANGE,
        use_api=True,
        hours=API_HOURS,
        include_radiation=API_INCLUDE_RADIATION
    )
    # API returns a single dataset with all variables
    era5_datasets = {'main': era5_data, 'pair': era5_data, 'rad': era5_data}
    era5 = era5_data
    era5_pair = era5_data
    era5_rad = era5_data
else:
    print("\n=== Loading data from NetCDF files ===")
    if isinstance(NETCDF_PATHS, dict):
        era5_data = forcing_tools.load_era5_data(
            time_range=(start_time, end_time),
            lon_range=None,
            lat_range=None,
            use_api=False,
            netcdf_paths=NETCDF_PATHS
        )
        era5_datasets = era5_data
        era5 = era5_data['main']
        era5_pair = era5_data.get('pair', era5_data['main'])
        era5_rad = era5_data.get('rad', era5_data['main'])
    else:
        # Single file
        era5_data = forcing_tools.load_era5_data(
            time_range=(start_time, end_time),
            lon_range=None,
            lat_range=None,
            use_api=False,
            netcdf_paths=NETCDF_PATHS
        )
        era5_datasets = {'main': era5_data}
        era5 = era5_data
        era5_pair = era5_data
        era5_rad = era5_data

# ============================================================================
# PROCESS ERA5 DATA - Convert variables and apply time filtering
# ============================================================================

# Get time coordinate and compute dt
# Note: Different ERA5 datasets may use different time dimension names
time_dim_main = 'time' if 'time' in era5.dims else 'valid_time'
time_dim_rad = 'time' if 'time' in era5_rad.dims else 'valid_time'
time_dim_pair = 'time' if 'time' in era5_pair.dims else 'valid_time'

time = era5[time_dim_main]
time = time[(time >= start_time) & (time <= end_time)]
dt = (time[1] - time[0]).astype('timedelta64[s]').values.astype(float)

# Process ERA5 variables with conversions
print("Processing ERA5 forcing data...")
swdn = convert_tools.convert_to_flux_density(
    era5_rad[era5_vars['shortwave']].sel({time_dim_rad: slice(start_time, end_time)}).values, dt
)
swrad = swdn - swdn * 0.06  # Apply albedo correction

lwrad = convert_tools.convert_to_flux_density(
    era5_rad[era5_vars['longwave_net']].sel({time_dim_rad: slice(start_time, end_time)}).values, dt
)

Tair = convert_tools.convert_K_to_C(
    era5[era5_vars['sst']].sel({time_dim_main: slice(start_time, end_time)}).values
)
Tair = np.nan_to_num(Tair, nan=np.nanmean(Tair))

qair = convert_tools.compute_relative_humidity(
    era5[era5_vars['airtemp']].sel({time_dim_main: slice(start_time, end_time)}).values,
    era5[era5_vars['dewpoint']].sel({time_dim_main: slice(start_time, end_time)}).values
)

Pair = convert_tools.convert_Pa_to_mbar(
    era5_pair[era5_vars['press']].sel({time_dim_pair: slice(start_time, end_time)}).values
)

rain_rate = convert_tools.compute_rainfall_cm_per_day(
    era5[era5_vars['precip']].sel({time_dim_main: slice(start_time, end_time)}).values, dt
)
# Convert to numpy array if it's a list, then convert cm/day to m/s
rain = np.array(rain_rate) * 0.01 / 86400

uwnd = convert_tools.calculate_surface_wind(
    era5[era5_vars['u10']].sel({time_dim_main: slice(start_time, end_time)}).values
)

vwnd = convert_tools.calculate_surface_wind(
    era5[era5_vars['v10']].sel({time_dim_main: slice(start_time, end_time)}).values
)

# ============================================================================
# INTERPOLATE TO ROMS GRID
# ============================================================================

# Prepare source and target coordinates
print("\nPreparing coordinates...")
source_coords = forcing_tools.prepare_forcing_coords(era5, lat_var='latitude', lon_var='longitude')
target_lon = grid['lon_rho'].values
target_lat = grid['lat_rho'].values

# Prepare variables dictionary for batch interpolation
variables_to_interp = {
    'swrad': swrad,
    'lwrad': lwrad,
    'Tair': Tair,
    'qair': qair,
    'Pair': Pair,
    'rain': rain,
    'Uwind': uwnd,
    'Vwind': vwnd
}

# Interpolate all variables at once
print("\nInterpolating forcing variables to ROMS grid...")
interpolated_vars = forcing_tools.interpolate_forcing_timeseries(
    variables_to_interp,
    source_coords['lon_2d'], source_coords['lat_2d'],
    target_lon, target_lat,
    interp_method='linear',
    verbose=True
)

# ============================================================================
# CREATE AND SAVE SURFACE FORCING DATASET
# ============================================================================

print("\nCreating surface forcing dataset...")
grid_dims = (target_lat.shape[0], target_lat.shape[1])

ds = forcing_tools.create_surface_forcing_dataset(
    interpolated_vars,
    time,
    grid_dims,
    ref_time='2000-01-01T00:00:00',
    source_name='ERA5',
    add_zero_fields=True
)

# Write to NetCDF 
print(f"\nWriting surface forcing file to: {output_file}")
ds.to_netcdf(output_file, format='NETCDF4', encoding={var: {'_FillValue': np.nan} for var in ds.data_vars})
print("Surface forcing file successfully created!")


