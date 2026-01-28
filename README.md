# model-tools
Python scripts to help generate model files for ROMS or REMORA

See documentation:
https://model-tools.readthedocs.io/en/latest/index.html
## Features

- **Grid Generation**: Create ROMS-compatible grids from bathymetry data
- **Initialization**: Generate initial conditions from ocean reanalysis data
- **Boundary Conditions**: Create boundary forcing files
- **Surface Forcing**: Process atmospheric forcing data
- **Flexible Data Access**: 
  - Use pre-downloaded NetCDF files
  - Access data remotely via Copernicus Marine API
  - Download and cache data for reuse

## Quick Start

### Installation

```bash
# Basic installation
pip install xarray numpy scipy netCDF4

# For remote data access (optional)
pip install copernicusmarine
```

### Using the Scripts

See `scripts/README_INITIALIZATION.md` for detailed initialization instructions.

Example: Initialize from Copernicus Marine API
```python
# In scripts/initialize.py, set:
USE_API = True
API_LON_RANGE = None  # Auto-detect from grid
API_LAT_RANGE = None  # Auto-detect from grid
init_time = np.datetime64('2024-01-01T00:00:00')

# Then run
python initialize.py
```

Example: Initialize from local NetCDF
```python
# In scripts/initialize.py, set:
USE_API = False
NETCDF_PATH = '/path/to/your/glorys_data.nc'
init_time = np.datetime64('2024-01-01T00:00:00')

# Then run
python initialize.py
```

## Data Sources

The package supports multiple methods for accessing ocean and atmospheric data:

### Ocean Data (GLORYS)

1. **Pre-downloaded NetCDF Files** - Fastest for repeated runs
2. **Copernicus Marine API** - Remote access, requires: `pip install copernicusmarine`
3. **Download and Cache** - Best of both worlds

See `docs/api/data_sources.rst` for ocean data details.

### Atmospheric Data (ERA5)

1. **Pre-downloaded NetCDF Files** - Fastest for repeated runs
2. **Copernicus Climate Data Store API** - Remote access, requires: `pip install cdsapi`
3. **Download and Cache** - Best of both worlds

See `docs/api/era5_forcing.rst` for atmospheric data details.

### Quick Setup

```bash
# For ocean data (GLORYS)
pip install copernicusmarine
copernicusmarine login

# For atmospheric data (ERA5)
pip install cdsapi
# Create ~/.cdsapirc with your credentials from https://cds.climate.copernicus.eu/user
```

## Documentation

Full documentation available at: https://model-tools.readthedocs.io/en/latest/index.html