# Summary of Changes: Copernicus Marine API Integration

## Overview

Enhanced the model-tools package to support flexible data access for initialization, with support for both pre-downloaded NetCDF files and remote access via the Copernicus Marine Toolbox API.

## Files Modified

### 1. `code/download.py`
**Added:** `CopernicusMarineDownloader` class with reusable methods:
- `open_dataset()`: Open remote datasets via API
- `download_dataset()`: Download data to local NetCDF file
- `get_glorys_dataset()`: Convenience method for GLORYS data (remote access)
- `download_glorys_dataset()`: Convenience method for downloading GLORYS data

**Key Features:**
- Support for spatial/temporal subsetting
- Depth range filtering
- Daily or monthly data products
- Force download option for cache refresh

### 2. `code/initialization.py`
**Added:** Three new static methods to `init_tools` class:
- `load_glorys_data()`: Unified interface for loading data from either NetCDF or API
- `download_and_cache_glorys()`: Download data via API and save for reuse
- Enhanced with type hints and documentation

**Key Features:**
- Automatic spatial extent detection from ROMS grid
- Time buffer configuration for temporal subsetting
- Consistent interface regardless of data source

### 3. `scripts/initialize.py`
**Modified:** Configuration section to support both data access methods:
- Added `USE_API` flag to toggle between NetCDF and API
- Added `NETCDF_PATH` for local file specification
- Added API configuration parameters:
  - `API_LON_RANGE`: Spatial extent (longitude)
  - `API_LAT_RANGE`: Spatial extent (latitude)
  - `API_TIME_BUFFER_DAYS`: Temporal extent
- Auto-detection of spatial extent from grid when ranges are None
- Updated data loading logic to use new `load_glorys_data()` method

## Files Created

### 4. `docs/api/data_sources.rst`
Comprehensive documentation covering:
- Installation and configuration
- Usage examples for both methods
- API reference for all new functions
- Available GLORYS datasets and variables
- Troubleshooting guide
- Example workflows for boundary conditions

### 5. `scripts/example_data_sources.py`
Five detailed examples demonstrating:
1. Loading from existing NetCDF files
2. Remote access via API
3. Download and cache workflow
4. Advanced API usage (custom variables, depth ranges, monthly means)
5. Multiple time periods for boundary conditions

### 6. `scripts/README_INITIALIZATION.md`
User guide for the initialization script with:
- Quick start instructions
- Configuration parameter reference
- Workflow recommendations
- Advanced usage patterns
- Troubleshooting section

### 7. `README.md` (updated)
Enhanced main README with:
- Feature list highlighting flexible data access
- Quick start examples for both methods
- Data source overview
- Link to detailed documentation

## Design Decisions

### 1. Backward Compatibility
- Default behavior unchanged (USE_API = False)
- Existing workflows continue to work without modification
- NetCDF file method remains the default for performance

### 2. Reusable Components
- All API access functions are static methods in classes
- Can be imported and used for boundary conditions, forcing, etc.
- Consistent API design across all functions

### 3. User-Friendly Configuration
- Simple boolean flag to switch between methods
- Auto-detection of spatial extent from grid
- Sensible defaults for all parameters
- Clear error messages with installation instructions

### 4. Performance Considerations
- Caching strategy for download-once-use-many pattern
- Spatial/temporal subsetting to minimize data transfer
- Option to use monthly means for smaller file sizes
- Force download option for cache refresh

## Usage Examples

### Basic Usage (NetCDF)
```python
# In initialize.py
USE_API = False
NETCDF_PATH = '/path/to/glorys_data.nc'
```

### Basic Usage (API)
```python
# In initialize.py
USE_API = True
API_LON_RANGE = None  # Auto-detect from grid
API_LAT_RANGE = None  # Auto-detect from grid
```

### Programmatic Usage
```python
from initialization import init_tools

# Load from file
ds = init_tools.load_glorys_data(
    init_time='2024-01-01',
    lon_range=None,
    lat_range=None,
    use_api=False,
    netcdf_path='glorys_data.nc'
)

# Load via API
ds = init_tools.load_glorys_data(
    init_time='2024-01-01',
    lon_range=(-80.0, -60.0),
    lat_range=(30.0, 50.0),
    use_api=True
)

# Download and cache
file_path = init_tools.download_and_cache_glorys(
    init_time='2024-01-01',
    lon_range=(-80.0, -60.0),
    lat_range=(30.0, 50.0),
    output_path='cached_glorys.nc'
)
```

## Dependencies

### Required (unchanged)
- xarray
- numpy
- scipy
- netCDF4

### Optional (new)
- copernicusmarine (for API access)

Install with: `pip install copernicusmarine`

## Future Applications

The new API access infrastructure can be reused for:
- **Boundary Conditions**: Download time series for lateral boundaries
- **Surface Forcing**: Access atmospheric forcing data
- **Climatology**: Generate climatological forcing
- **Validation**: Compare model output with observations
- **Data Exploration**: Quick access to reanalysis data for testing

## Testing Recommendations

1. Test NetCDF method (ensure backward compatibility)
2. Test API method with small domain (verify API access)
3. Test auto-detection of spatial extent
4. Test download and cache workflow
5. Test with different time periods
6. Verify error handling for missing credentials

## Notes

- API access requires free Copernicus Marine account
- Authentication: `copernicusmarine login` (one-time)
- API method is slower for first access but enables exploration
- Download-and-cache recommended for production workflows
- All API functions include comprehensive docstrings and examples
