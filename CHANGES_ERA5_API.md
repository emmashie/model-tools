# Summary: ERA5 CDS API Integration

## Overview

Enhanced the model-tools package to support flexible data access for surface forcing, with support for both pre-downloaded NetCDF files and remote access via the Copernicus Climate Data Store (CDS) API for ERA5 data.

## Files Modified

### 1. `code/download.py`
**Added:** `ERA5Downloader` class with methods:
- `download_era5_data()`: Download ERA5 data with custom variable selection
- `download_era5_surface_forcing()`: Convenience method for standard forcing variables
- `get_era5_variable_mapping()`: Reference for ERA5 variable names (long/short forms)

**Key Features:**
- Spatial/temporal subsetting
- Temporal resolution control (hourly to daily)
- Support for all ERA5 single-level variables
- Automatic format handling (NetCDF/GRIB)
- Force download option

### 2. `code/forcing.py`
**Added:** Three new static methods to `forcing_tools` class:
- `load_era5_data()`: Unified interface for loading ERA5 from NetCDF or API
- `download_and_cache_era5()`: Download via API and save for reuse
- Enhanced with type hints and comprehensive documentation

**Key Features:**
- Auto-detection of spatial extent from ROMS grid
- Support for single or multiple NetCDF files
- Flexible temporal resolution
- Consistent interface regardless of data source

### 3. `scripts/surface_forcing.py`
**Modified:** Configuration and data loading sections:
- Added `USE_API` flag to toggle between NetCDF and API
- Added `NETCDF_PATHS` for local file specification (single or multiple files)
- Added API configuration parameters:
  - `API_LON_RANGE`, `API_LAT_RANGE`: Spatial extent
  - `API_HOURS`: Temporal resolution
  - `API_INCLUDE_RADIATION`: Control radiation variables
- Auto-detection of spatial extent from grid
- Updated data loading to use new `load_era5_data()` method

## Files Created

### 4. `scripts/README_SURFACE_FORCING.md`
Comprehensive user guide covering:
- Quick start for both methods
- Configuration parameter reference
- Workflow recommendations
- Advanced usage examples
- ERA5 variable reference
- Troubleshooting guide

### 5. `scripts/example_era5_forcing.py`
Six detailed examples demonstrating:
1. Loading from existing NetCDF files (single and multiple)
2. Remote access via CDS API
3. Download and cache workflow
4. Advanced API usage (custom variables, temporal resolution, long-term)
5. ERA5 variable mapping reference
6. Monthly chunk downloads for large datasets

### 6. `docs/api/era5_forcing.rst`
API documentation including:
- CDS API installation and setup
- Configuration methods
- Python API usage examples
- ERA5 variable reference
- Temporal resolution options
- Workflow recommendations
- Unit conversions
- Troubleshooting

### 7. Updated `README.md`
Enhanced main README with:
- Separated ocean and atmospheric data sections
- Quick setup for both APIs
- Links to detailed documentation

### 8. Updated `QUICK_REFERENCE.md`
Added ERA5 section with:
- Three data access methods
- Quick commands
- Parameter comparison table
- Troubleshooting for both APIs
- Links to examples

## Design Decisions

### 1. Consistency with GLORYS Implementation
- Same pattern as initialization (USE_API flag, auto-detection, etc.)
- Consistent function naming and parameters
- Similar workflow recommendations

### 2. Backward Compatibility
- Default behavior unchanged (USE_API = False)
- Existing NetCDF-based workflows continue to work
- Supports both single and multiple file configurations

### 3. Flexible File Organization
- Support for single NetCDF with all variables
- Support for multiple files (main, pair, rad)
- Automatic handling of different file structures

### 4. Temporal Resolution Control
- Users can specify which hours to download
- Balances download size vs. temporal resolution
- Default to 6-hourly (common for ocean modeling)

### 5. Radiation Variables
- Optional inclusion via `API_INCLUDE_RADIATION` flag
- Reduces download size if radiation not needed
- Includes both downward and net radiation

## Usage Examples

### Basic Usage (NetCDF)
```python
# In surface_forcing.py
USE_API = False
NETCDF_PATHS = '/path/to/era5_data.nc'
```

### Basic Usage (API)
```python
# In surface_forcing.py
USE_API = True
API_LON_RANGE = None  # Auto-detect
API_LAT_RANGE = None  # Auto-detect
API_HOURS = ['00:00', '06:00', '12:00', '18:00']
```

### Programmatic Usage
```python
from forcing import forcing_tools

# Load from file
ds = forcing_tools.load_era5_data(
    time_range=('2024-01-01', '2024-01-31'),
    lon_range=None,
    lat_range=None,
    use_api=False,
    netcdf_paths='era5_data.nc'
)

# Load via API
ds = forcing_tools.load_era5_data(
    time_range=('2024-01-01', '2024-01-31'),
    lon_range=(-80.0, -60.0),
    lat_range=(30.0, 50.0),
    use_api=True,
    hours=['00:00', '06:00', '12:00', '18:00']
)

# Download and cache
file_path = forcing_tools.download_and_cache_era5(
    time_range=('2024-01-01', '2024-12-31'),
    lon_range=(-80.0, -60.0),
    lat_range=(30.0, 50.0),
    output_path='era5_2024.nc',
    hours=['00:00', '06:00', '12:00', '18:00']
)
```

## ERA5 Variables

### Standard Variables (included by default)
- Wind: `u10`, `v10` (10m components)
- Temperature: `t2m` (2m air temperature)
- Humidity: `d2m` (2m dewpoint temperature)
- Pressure: `msl` (mean sea level)
- Precipitation: `tp` (total)
- Sea surface: `sst` (temperature)

### Radiation Variables (optional)
- `ssrd`: Surface solar radiation downwards
- `strd`: Surface thermal radiation downwards
- `ssr`: Surface net solar radiation
- `str`: Surface net thermal radiation

## Dependencies

### Required (unchanged)
- xarray
- numpy
- scipy
- netCDF4
- pandas (for date handling)

### Optional (new)
- cdsapi (for CDS API access)

Install with: `pip install cdsapi`

## CDS API Setup

1. Create account: https://cds.climate.copernicus.eu/
2. Get API key from user profile
3. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_UID:YOUR_API_KEY
   ```

## Future Applications

The ERA5 API access infrastructure can be reused for:
- Different atmospheric variables
- Multiple pressure levels
- Ensemble members
- Long-term climatologies
- Storm-specific high-resolution forcing
- Wave forcing data

## Performance Considerations

### Download Times
- CDS can queue requests during peak usage
- Large requests (annual, global) can take hours
- Recommended: Download during off-peak hours (evening/night Europe time)

### File Sizes
- 1 month, regional domain, 6-hourly ≈ 100-500 MB
- Annual, regional, 6-hourly ≈ 1-5 GB
- Hourly data ≈ 4x larger than 6-hourly

### Optimization Strategies
1. Use appropriate temporal resolution for your application
2. Download monthly chunks for long simulations
3. Exclude radiation if not needed
4. Use spatial subsetting to cover only your domain + buffer

## Testing Recommendations

1. Test NetCDF method (backward compatibility)
2. Test API method with small domain (1-2 days)
3. Test auto-detection of spatial extent
4. Test download and cache workflow
5. Test with different temporal resolutions
6. Verify error handling for missing credentials

## Comparison: GLORYS vs ERA5

| Feature | GLORYS (Ocean) | ERA5 (Atmospheric) |
|---------|----------------|-------------------|
| API Package | `copernicusmarine` | `cdsapi` |
| Authentication | `copernicusmarine login` | `~/.cdsapirc` file |
| Temporal Resolution | Daily/Monthly | Hourly (customizable) |
| Download Speed | Generally faster | Can be slow (queuing) |
| Data Volume | Larger (3D ocean) | Smaller (2D surface) |
| Typical Use | Initialization, BC | Surface forcing |

## Notes

- ERA5 provides hourly data globally at ~31 km resolution
- Data available from 1940 to near real-time
- CDS may queue requests during peak usage
- All unit conversions handled automatically
- Both short names (u10) and long names (10m_u_component_of_wind) supported
- Script handles different time dimension names (time, valid_time)

## Documentation Files

- `scripts/README_SURFACE_FORCING.md` - User guide
- `scripts/example_era5_forcing.py` - 6 practical examples
- `docs/api/era5_forcing.rst` - API reference
- `QUICK_REFERENCE.md` - Quick reference for both GLORYS and ERA5
- `README.md` - Updated with both data sources
