# Initialization Script Usage

The `initialize.py` script creates ROMS initial condition files from ocean reanalysis data. It now supports two methods for accessing data:

## Quick Start

### Method 1: Using Existing NetCDF Files (Default)

If you already have GLORYS or other ocean data downloaded:

```python
# In initialize.py, set:
USE_API = False
NETCDF_PATH = '/path/to/your/glorys_data.nc'
```

Then run:
```bash
python initialize.py
```

### Method 2: Using Copernicus Marine API

To access data directly from Copernicus Marine Service:

1. Install the required package:
   ```bash
   pip install copernicusmarine
   ```

2. Set up authentication (one-time):
   ```bash
   copernicusmarine login
   ```
   You'll need a free account from https://marine.copernicus.eu/

3. Configure the script:
   ```python
   # In initialize.py, set:
   USE_API = True
   API_LON_RANGE = None  # Auto-detect from grid, or set e.g., (-80.0, -60.0)
   API_LAT_RANGE = None  # Auto-detect from grid, or set e.g., (30.0, 50.0)
   API_TIME_BUFFER_DAYS = 1  # Download ±1 day around init_time
   ```

4. Run:
   ```bash
   python initialize.py
   ```

## Configuration Parameters

### Common Parameters

- `base_path`: Base directory for model-tools
- `grid_nc`: ROMS grid filename
- `output_nc`: Output initial conditions filename
- `init_time`: Initialization date/time (e.g., `'2024-01-01T00:00:00'`)

### NetCDF Method Parameters

- `NETCDF_PATH`: Full path to your pre-downloaded GLORYS NetCDF file

### API Method Parameters

- `API_LON_RANGE`: Longitude range `(min, max)` in degrees East, or `None` to auto-detect from grid
- `API_LAT_RANGE`: Latitude range `(min, max)` in degrees North, or `None` to auto-detect from grid
- `API_TIME_BUFFER_DAYS`: Number of days before/after `init_time` to download

### Source Data Variables

Map your data source variable names:

```python
source_vars = {
    'salt': 'so',        # Salinity
    'temp': 'thetao',    # Potential temperature
    'u': 'uo',           # Eastward velocity
    'v': 'vo',           # Northward velocity
    'zeta': 'zos',       # Sea surface height
    'depth': 'depth',    # Depth coordinate
    'lat': 'latitude',   # Latitude coordinate
    'lon': 'longitude'   # Longitude coordinate
}
```

### Fill Values and Constraints

```python
deep_ocean_fill_values = {
    'zos': np.nan,
    'uo': 0.0,
    'vo': 0.0,
    'so': 35.0,      # Deep ocean salinity
    'thetao': 1.0    # Deep ocean temperature
}

fill_values = {
    'temp': 5.0,     # Fill value for masked temperature
    'salt': 32.0,    # Fill value for masked salinity
    'u': 0.0,
    'v': 0.0,
    'zeta': 0.0
}

min_temp = 0.1  # Minimum temperature constraint (degrees C)
```

## Examples

See `example_data_sources.py` for detailed examples of:

1. Loading from existing NetCDF files
2. Remote access via API
3. Downloading and caching data
4. Advanced API usage
5. Handling multiple time periods

## Workflow Recommendations

### For Testing/Development

Use the API directly:
- Quick access without managing files
- Easy to adjust spatial/temporal extent
- Slower for repeated runs

### For Production Runs

Use the download-and-cache approach:
1. Download data once via API
2. Save to local NetCDF file
3. Reuse the file for multiple runs
4. Much faster for repeated initialization

Example workflow:
```python
from initialization import init_tools

# One-time download
init_tools.download_and_cache_glorys(
    init_time='2024-01-01',
    lon_range=(-80.0, -60.0),
    lat_range=(30.0, 50.0),
    output_path='glorys_jan2024.nc',
    time_buffer_days=7
)

# Then in initialize.py:
USE_API = False
NETCDF_PATH = 'glorys_jan2024.nc'
```

## Advanced Usage

### Using Different Data Sources

The framework is designed to work with any ocean model output that provides the required variables. To use a different source:

1. Download or access your data
2. Update `source_vars` dictionary to match your variable names
3. Adjust `deep_ocean_fill_values` and `fill_values` as needed
4. Run the script

### Custom Spatial/Temporal Extent

When using the API, you can customize exactly what data is downloaded:

```python
# Minimal spatial extent (faster downloads)
API_LON_RANGE = (-75.5, -70.5)  # Small region
API_LAT_RANGE = (38.0, 42.0)
API_TIME_BUFFER_DAYS = 0  # Just the init_time

# Larger extent for interpolation (more robust)
API_LON_RANGE = (-85.0, -65.0)  # Larger region
API_LAT_RANGE = (25.0, 50.0)
API_TIME_BUFFER_DAYS = 3  # ±3 days
```

### Reusable Components

The API access functions in `code/download.py` and `code/initialization.py` are designed to be reused for:

- Boundary conditions
- Surface forcing
- Climatology
- Any other ocean data needs

See the documentation in `docs/api/data_sources.rst` for details.

## Troubleshooting

### "copernicusmarine not found"

```bash
pip install copernicusmarine
```

### "Authentication failed"

```bash
copernicusmarine login
```

### "Dataset not found"

Check the dataset ID. GLORYS datasets:
- Daily: `cmems_mod_glo_phy_my_0.083deg_P1D-m`
- Monthly: `cmems_mod_glo_phy_my_0.083deg_P1M-m`

Browse available datasets: https://data.marine.copernicus.eu/products

### Slow downloads

- Reduce spatial extent (`API_LON_RANGE`, `API_LAT_RANGE`)
- Reduce temporal extent (`API_TIME_BUFFER_DAYS`)
- Download once and cache the file
- Use monthly means instead of daily data

### Memory issues

- Limit depth range in download functions
- Process smaller spatial domains
- Download data in chunks for very large domains

## Support

For more information:
- Documentation: https://model-tools.readthedocs.io
- Copernicus Marine: https://help.marine.copernicus.eu/
- GLORYS Product: https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/
