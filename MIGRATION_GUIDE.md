# Migration Guide: Updating to New Data Access Methods

## For Existing Users

If you're already using the model-tools package, your existing workflows **will continue to work without any changes**. The new features are **opt-in** and backward compatible.

## What's New?

- ✅ Support for Copernicus Marine API (remote data access)
- ✅ Flexible data source configuration
- ✅ Reusable helper functions for boundary conditions and forcing
- ✅ Download-and-cache workflow

## Do I Need to Change Anything?

**No!** By default, the initialization script still uses local NetCDF files exactly as before.

### Your Current Workflow (Still Works)

```python
# scripts/initialize.py - works exactly as before
init_time = np.datetime64('2024-01-01T00:00:00')
grid = xr.open_dataset('roms_grid.nc')
init_data = xr.open_dataset('/path/to/glorys_data.nc')
# ... rest of your code
```

### New Configuration Style (Optional)

```python
# scripts/initialize.py - new style (optional)
USE_API = False  # Still using NetCDF files
NETCDF_PATH = '/path/to/glorys_data.nc'
init_time = np.datetime64('2024-01-01T00:00:00')

# Data is loaded automatically using:
init_data = init_tools.load_glorys_data(
    init_time=init_time,
    lon_range=None,
    lat_range=None,
    use_api=False,
    netcdf_path=NETCDF_PATH
)
```

## When Should I Upgrade My Workflow?

### Stay with Current Method If:
- ✅ Your workflow is working fine
- ✅ You have all data pre-downloaded
- ✅ You prefer explicit file paths
- ✅ You don't need remote data access

### Upgrade to New Method If:
- ✅ You want to explore different domains without downloading first
- ✅ You're setting up new simulations frequently
- ✅ You want to automate data downloads
- ✅ You're working on boundary conditions or forcing (reusable functions!)

## Step-by-Step Migration

### Option 1: Keep Using NetCDF Files (Easiest)

1. Update to latest code (just pull the new files)
2. In `scripts/initialize.py`, add these lines at the top of configuration:
   ```python
   USE_API = False
   NETCDF_PATH = '/your/existing/path/to/glorys_data.nc'
   ```
3. Comment out or remove your old `xr.open_dataset()` line
4. The rest works automatically!

### Option 2: Try the API (Recommended for New Work)

1. Install the API package:
   ```bash
   pip install copernicusmarine
   ```

2. Authenticate (one-time):
   ```bash
   copernicusmarine login
   ```

3. Update `scripts/initialize.py`:
   ```python
   USE_API = True
   API_LON_RANGE = None  # Auto-detect from grid
   API_LAT_RANGE = None  # Auto-detect from grid
   API_TIME_BUFFER_DAYS = 1
   ```

4. Run as normal:
   ```bash
   python scripts/initialize.py
   ```

### Option 3: Download-and-Cache (Best Practice)

1. Install and authenticate (same as Option 2)

2. Create a one-time download script:
   ```python
   from initialization import init_tools
   
   # Download once
   init_tools.download_and_cache_glorys(
       init_time='2024-01-01',
       lon_range=(-80.0, -60.0),
       lat_range=(30.0, 50.0),
       output_path='cached_glorys_jan2024.nc',
       time_buffer_days=3
   )
   ```

3. Then use the cached file:
   ```python
   USE_API = False
   NETCDF_PATH = 'cached_glorys_jan2024.nc'
   ```

## Breaking Changes

**None!** This update is 100% backward compatible.

## New Features You Might Want

### 1. Reusable Data Loading

Old way (direct xarray):
```python
init_data = xr.open_dataset('/path/to/glorys.nc')
```

New way (more flexible):
```python
init_data = init_tools.load_glorys_data(
    init_time='2024-01-01',
    lon_range=None,
    lat_range=None,
    use_api=False,
    netcdf_path='/path/to/glorys.nc'
)
```

Benefits:
- Same function works with API or files
- Can switch sources without changing downstream code
- Reusable for boundary conditions

### 2. Automatic Spatial Extent

Old way (manual):
```python
# Download data covering specific region
# Have to figure out lon/lat ranges manually
```

New way (automatic):
```python
USE_API = True
API_LON_RANGE = None  # Auto-detect from your grid!
API_LAT_RANGE = None  # Adds 10% buffer automatically
```

### 3. Download Helper for Boundary Conditions

Can now reuse for other workflows:
```python
from download import CopernicusMarineDownloader

downloader = CopernicusMarineDownloader()

# Download monthly data for full year
for month in range(1, 13):
    ds = downloader.get_glorys_dataset(
        lon_range=(-80.0, -60.0),
        lat_range=(30.0, 50.0),
        time_range=(f'2024-{month:02d}-01', f'2024-{month+1:02d}-01')
    )
    # Process for boundary conditions...
```

## Common Migration Scenarios

### Scenario 1: "I just want it to work like before"

**Action:** No changes needed! Your code still works.

**Optional:** Add two lines for clarity:
```python
USE_API = False
NETCDF_PATH = '/your/current/path.nc'
```

### Scenario 2: "I want to try the API but keep my files as backup"

**Action:**
```python
# Try API first
USE_API = True
API_LON_RANGE = None
API_LAT_RANGE = None

# If API fails, switch to:
# USE_API = False
# NETCDF_PATH = '/your/backup/file.nc'
```

### Scenario 3: "I'm starting a new project"

**Action:** Use download-and-cache workflow (see Option 3 above)

### Scenario 4: "I need this for boundary conditions too"

**Action:** Check out `scripts/example_data_sources.py` Example 5 for a complete workflow using the new reusable functions.

## Rollback Plan

If you want to revert to the old way:

1. The old code is still there - just don't use the new parameters
2. Or keep your old `initialize.py` script (just don't update it)
3. The core functions (`add_deep_ocean_layer`, etc.) are unchanged

## Getting Help

### Documentation
- **Quick Start:** `QUICK_REFERENCE.md`
- **Detailed Guide:** `scripts/README_INITIALIZATION.md`
- **API Reference:** `docs/api/data_sources.rst`
- **Examples:** `scripts/example_data_sources.py`

### Testing
Run the test suite to verify everything works:
```bash
python scripts/test_data_access.py
```

### Common Issues

| Issue | Solution |
|-------|----------|
| "Old code doesn't work anymore" | It should! Check if you accidentally changed something |
| "Can't import copernicusmarine" | It's optional! Only needed for API access |
| "Want to use API but don't have credentials" | Sign up free at https://marine.copernicus.eu/ |
| "Downloads are too slow" | Use download-and-cache, then switch to NetCDF mode |

## Summary

- ✅ **No action required** - existing code works as-is
- ✅ **Opt-in features** - use new functionality when you need it
- ✅ **Backward compatible** - no breaking changes
- ✅ **Enhanced capabilities** - remote access, automation, reusability

The update adds powerful new features while keeping everything you already have working perfectly!
