# Surface Forcing Script Usage

The `surface_forcing.py` script creates ROMS surface forcing files from ERA5 reanalysis data. It now supports two methods for accessing data:

## Quick Start

### Method 1: Using Existing NetCDF Files (Default)

If you already have ERA5 data downloaded:

```python
# In surface_forcing.py, set:
USE_API = False
NETCDF_PATHS = {
    'main': '/path/to/ERA5_data.nc',
    'pair': '/path/to/ERA5_Pair.nc',
    'rad': '/path/to/ERA5_rad.nc'
}
# Or for a single file with all variables:
# NETCDF_PATHS = '/path/to/era5_forcing.nc'
```

Then run:
```bash
python surface_forcing.py
```

### Method 2: Using Copernicus Climate Data Store API

To access data directly from the CDS:

1. Install the required package:
   ```bash
   pip install cdsapi
   ```

2. Set up authentication (one-time):
   - Register for a free account at https://cds.climate.copernicus.eu/
   - Get your API key from your profile page
   - Create `~/.cdsapirc` file:
     ```
     url: https://cds.climate.copernicus.eu/api/v2
     key: YOUR_UID:YOUR_API_KEY
     ```

3. Configure the script:
   ```python
   # In surface_forcing.py, set:
   USE_API = True
   API_LON_RANGE = None  # Auto-detect from grid, or set e.g., (-80.0, -60.0)
   API_LAT_RANGE = None  # Auto-detect from grid, or set e.g., (30.0, 50.0)
   API_HOURS = ['00:00', '06:00', '12:00', '18:00']  # Time resolution
   API_INCLUDE_RADIATION = True  # Include radiation variables
   ```

4. Run:
   ```bash
   python surface_forcing.py
   ```

## Configuration Parameters

### Common Parameters

- `base_path`: Base directory for model-tools
- `grid_nc`: ROMS grid filename
- `start_time`: Start date/time for forcing (e.g., `'2024-01-01T00:00:00'`)
- `end_time`: End date/time for forcing (e.g., `'2024-01-31T23:00:00'`)
- `output_file`: Output forcing filename

### NetCDF Method Parameters

- `NETCDF_PATHS`: Path to NetCDF file(s)
  - Dictionary for multiple files: `{'main': 'file1.nc', 'pair': 'file2.nc', 'rad': 'file3.nc'}`
  - Single file: `'era5_all_variables.nc'`

### API Method Parameters

- `API_LON_RANGE`: Longitude range `(min, max)` in degrees East, or `None` to auto-detect
- `API_LAT_RANGE`: Latitude range `(min, max)` in degrees North, or `None` to auto-detect
- `API_HOURS`: List of hours to download (e.g., `['00:00', '06:00', '12:00', '18:00']`)
  - Use `None` for all hours (24 per day)
  - Fewer hours = smaller downloads
- `API_INCLUDE_RADIATION`: Whether to include radiation variables (shortwave/longwave)

### ERA5 Variable Mapping

The script uses standard ERA5 variable names:

```python
era5_vars = {
    'shortwave': 'ssrd',      # Surface solar radiation downwards
    'longwave': 'strd',       # Surface thermal radiation downwards
    'shortwave_net': 'ssrd',  # (calculated with albedo)
    'longwave_net': 'str',    # Surface net thermal radiation
    'sst': 'sst',             # Sea surface temperature
    'airtemp': 't2m',         # 2m temperature
    'dewpoint': 'd2m',        # 2m dewpoint temperature
    'precip': 'tp',           # Total precipitation
    'u10': 'u10',             # 10m u-wind component
    'v10': 'v10',             # 10m v-wind component
    'press': 'msl'            # Mean sea level pressure
}
```

## Examples

### Example 1: Download and Cache (Recommended)

Download data once and reuse:

```python
from forcing import forcing_tools

# One-time download
forcing_tools.download_and_cache_era5(
    time_range=('2024-01-01', '2024-01-31'),
    lon_range=(-80.0, -60.0),
    lat_range=(30.0, 50.0),
    output_path='era5_jan2024.nc',
    hours=['00:00', '06:00', '12:00', '18:00'],
    include_radiation=True
)

# Then in surface_forcing.py:
USE_API = False
NETCDF_PATHS = 'era5_jan2024.nc'
```

### Example 2: Direct API Access

Quick testing without saving intermediate files:

```python
# In surface_forcing.py
USE_API = True
API_LON_RANGE = (-75.0, -70.0)  # Small region for testing
API_LAT_RANGE = (38.0, 42.0)
API_HOURS = ['00:00', '12:00']  # Just twice daily
start_time = np.datetime64('2024-01-01T00:00:00')
end_time = np.datetime64('2024-01-03T00:00:00')  # Just a few days
```

### Example 3: High-Resolution Forcing

For high-frequency forcing (e.g., storm events):

```python
USE_API = True
API_HOURS = None  # All 24 hours per day
API_INCLUDE_RADIATION = True
start_time = np.datetime64('2024-08-15T00:00:00')
end_time = np.datetime64('2024-08-20T00:00:00')  # Hurricane period
```

### Example 4: Long-Term Climatology

For annual simulations with reduced temporal resolution:

```python
USE_API = True
API_HOURS = ['00:00', '12:00']  # Twice daily sufficient
API_INCLUDE_RADIATION = True
start_time = np.datetime64('2024-01-01T00:00:00')
end_time = np.datetime64('2024-12-31T00:00:00')  # Full year
```

## Workflow Recommendations

### For Testing/Development

1. Use API directly with small domain and short time period
2. Set `API_HOURS = ['00:00', '12:00']` for faster downloads
3. Test with 1-3 days of data first

### For Production Runs

1. **Download and cache approach** (most efficient):
   ```python
   # Step 1: Download once
   forcing_tools.download_and_cache_era5(
       time_range=('2024-01-01', '2024-12-31'),
       lon_range=(-80.0, -60.0),
       lat_range=(30.0, 50.0),
       output_path='era5_2024_full.nc',
       hours=['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
   )
   
   # Step 2: Use cached file for all runs
   USE_API = False
   NETCDF_PATHS = 'era5_2024_full.nc'
   ```

2. Adjust time range in script as needed
3. Reuse cached file for multiple experiments

### For HPC Environments

- Download data on login node (network access)
- Copy to fast scratch storage if available
- Use cached NetCDF files on compute nodes (Method 1)
- Consider downloading larger time windows to minimize API calls

## Advanced Usage

### Custom Variable Selection

When using the API programmatically:

```python
from download import ERA5Downloader

downloader = ERA5Downloader()

# Download specific variables only
downloader.download_era5_data(
    output_path='era5_custom.nc',
    variables=[
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2m_temperature',
        'mean_sea_level_pressure'
    ],
    lon_range=(-80.0, -60.0),
    lat_range=(30.0, 50.0),
    time_range=('2024-01-01', '2024-01-31'),
    hours=['00:00', '12:00']
)
```

### Available ERA5 Variables

Standard surface forcing variables:
- `10m_u_component_of_wind` (u10)
- `10m_v_component_of_wind` (v10)
- `2m_temperature` (t2m)
- `2m_dewpoint_temperature` (d2m)
- `mean_sea_level_pressure` (msl)
- `total_precipitation` (tp)
- `sea_surface_temperature` (sst)
- `surface_solar_radiation_downwards` (ssrd)
- `surface_thermal_radiation_downwards` (strd)
- `surface_net_solar_radiation` (ssr)
- `surface_net_thermal_radiation` (str)

Additional variables available from ERA5:
- `significant_height_of_combined_wind_waves_and_swell`
- `10m_wind_speed`
- `surface_pressure`
- And many more...

Browse all variables: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels

## Troubleshooting

### "cdsapi not found"

```bash
pip install cdsapi
```

### "Authentication failed" or "Invalid API key"

1. Check your `~/.cdsapirc` file exists and has correct format:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_UID:YOUR_API_KEY
   ```

2. Get your API key from: https://cds.climate.copernicus.eu/user

3. Make sure you've accepted the ERA5 terms of use

### Slow downloads

ERA5 downloads can be slow during peak usage. To optimize:

- Reduce spatial extent (`API_LON_RANGE`, `API_LAT_RANGE`)
- Reduce temporal resolution (`API_HOURS`)
- Download in smaller time chunks
- Use fewer variables
- Download during off-peak hours (evening/night in Europe)

### "Request queued" messages

CDS queues requests during high demand:
- Be patient (can take minutes to hours for large requests)
- Download and cache for future use
- Consider splitting large requests into smaller chunks

### Memory issues

For large domains or long time periods:
- Process in chunks (e.g., month by month)
- Reduce temporal resolution (fewer hours per day)
- Limit spatial extent
- Use compute nodes with more memory on HPC

### Variable not found errors

Make sure variable names match ERA5 conventions:
- Use long names: `'10m_u_component_of_wind'`
- Or short names: `'u10'`
- Check available variables at CDS website
- Use `ERA5Downloader.get_era5_variable_mapping()` for reference

## Data Format Notes

### Time Coordinates

ERA5 data may use different time dimension names:
- `'time'` (standard)
- `'valid_time'` (some products)

The script handles both automatically.

### Spatial Coordinates

ERA5 uses:
- Longitude: 0-360° (script handles conversion to -180 to 180°)
- Latitude: 90° to -90° (North to South)

### Units

Variables are automatically converted to ROMS conventions:
- Temperature: Kelvin → Celsius
- Pressure: Pascal → millibar
- Radiation: Accumulated → flux density (W/m²)
- Precipitation: Meters → m/s

## Reusable Components

The API access functions in `code/download.py` and `code/forcing.py` are designed to be reused for:

- Different time periods
- Different spatial domains
- Custom variable selections
- Other atmospheric forcing needs

See detailed examples in `scripts/example_era5_forcing.py`.

## Support

For more information:
- Documentation: https://model-tools.readthedocs.io
- ERA5 Documentation: https://confluence.ecmwf.int/display/CKB/ERA5
- CDS API Guide: https://cds.climate.copernicus.eu/api-how-to
- ERA5 Data Access: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
