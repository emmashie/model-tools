# Quick Reference: Data Access Methods

## Ocean Data (GLORYS - Initialization & Boundary Conditions)

### Three Ways to Access GLORYS Data

### 1️⃣ Use Existing NetCDF File (Default & Fastest)

**When to use:** You already have the data downloaded

**Setup:**
```python
# In scripts/initialize.py
USE_API = False
NETCDF_PATH = '/path/to/your/glorys_data.nc'
```

**Pros:** ✅ Fast, ✅ No network needed, ✅ Works offline  
**Cons:** ❌ Must download data first

---

### 2️⃣ Remote API Access (No Download)

**When to use:** Quick exploration, testing different domains

**Setup:**
```bash
pip install copernicusmarine
copernicusmarine login  # One-time setup
```

```python
# In scripts/initialize.py
USE_API = True
API_LON_RANGE = None  # Auto-detect from grid
API_LAT_RANGE = None  # Auto-detect from grid
API_TIME_BUFFER_DAYS = 1
```

**Pros:** ✅ No local storage, ✅ Always up-to-date, ✅ Easy spatial/temporal control  
**Cons:** ❌ Slower, ❌ Requires network, ❌ Repeated downloads

---

### 3️⃣ Download & Cache (Best Practice)

**When to use:** Production runs, repeated initialization

**Setup:**
```bash
pip install copernicusmarine
copernicusmarine login  # One-time setup
```

**Workflow:**
```python
# Step 1: Download once (in Python or standalone script)
from initialization import init_tools

init_tools.download_and_cache_glorys(
    init_time='2024-01-01',
    lon_range=(-80.0, -60.0),
    lat_range=(30.0, 50.0),
    output_path='glorys_jan2024.nc',
    time_buffer_days=3
)

# Step 2: Use cached file (in scripts/initialize.py)
USE_API = False
NETCDF_PATH = 'glorys_jan2024.nc'
```

**Pros:** ✅ Fast after first download, ✅ Reusable, ✅ Best of both worlds  
**Cons:** ❌ Requires local storage

---

## Quick Commands

### Check if API is installed
```bash
python -c "import copernicusmarine; print('✅ API available')"
```

### Login to Copernicus Marine
```bash
copernicusmarine login
```

### Test API access
```python
from download import CopernicusMarineDownloader

downloader = CopernicusMarineDownloader()
ds = downloader.get_glorys_dataset(
    lon_range=(-75.0, -70.0),
    lat_range=(35.0, 40.0),
    time_range=('2024-01-01', '2024-01-02')
)
print(ds)
```

---

## Configuration Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `USE_API` | bool | Use API (True) or NetCDF (False) | `False` |
| `NETCDF_PATH` | str | Path to NetCDF file | `'data.nc'` |
| `API_LON_RANGE` | tuple or None | (min, max) longitude | `(-80, -60)` |
| `API_LAT_RANGE` | tuple or None | (min, max) latitude | `(30, 50)` |
| `API_TIME_BUFFER_DAYS` | int | Days ± init_time | `1` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: copernicusmarine` | `pip install copernicusmarine` |
| Authentication error | `copernicusmarine login` |
| Slow downloads | Reduce spatial/temporal extent |
| File already exists | Set `force_download=True` |
| Memory error | Use smaller domain or depth range |

---

## Recommended Workflows

### For Beginners
Start with API method (Method 2) for exploration, then switch to download & cache (Method 3) when domain is finalized.

### For Production
Always use download & cache method (Method 3):
1. Download data once with generous spatial/temporal buffer
2. Reuse cached file for all runs
3. Re-download only when updating time period

### For HPC Environments
Download data on login node (network access), then use cached NetCDF files on compute nodes (Method 1).

---

## Examples

See detailed examples in:
- `scripts/example_data_sources.py` - 5 complete examples
- `scripts/README_INITIALIZATION.md` - Full documentation
- `docs/api/data_sources.rst` - API reference

---

## Atmospheric Data (ERA5 - Surface Forcing)

### Three Ways to Access ERA5 Data

#### 1️⃣ Use Existing NetCDF File (Default & Fastest)

**When to use:** You already have the data downloaded

**Setup:**
```python
# In scripts/surface_forcing.py
USE_API = False
NETCDF_PATHS = {
    'main': '/path/to/ERA5_data.nc',
    'pair': '/path/to/ERA5_Pair.nc',
    'rad': '/path/to/ERA5_rad.nc'
}
# Or single file: NETCDF_PATHS = '/path/to/era5_all.nc'
```

**Pros:** ✅ Fast, ✅ No network needed, ✅ Works offline  
**Cons:** ❌ Must download data first

---

#### 2️⃣ Remote API Access (No Download)

**When to use:** Quick exploration, testing different domains

**Setup:**
```bash
pip install cdsapi
# Create ~/.cdsapirc with credentials from https://cds.climate.copernicus.eu/user
```

```python
# In scripts/surface_forcing.py
USE_API = True
API_LON_RANGE = None  # Auto-detect from grid
API_LAT_RANGE = None  # Auto-detect from grid
API_HOURS = ['00:00', '06:00', '12:00', '18:00']
API_INCLUDE_RADIATION = True
```

**Pros:** ✅ No local storage, ✅ Always up-to-date, ✅ Easy control  
**Cons:** ❌ Slower, ❌ Requires network, ❌ Can be queued

---

#### 3️⃣ Download & Cache (Best Practice)

**When to use:** Production runs, repeated forcing generation

**Workflow:**
```python
# Step 1: Download once
from forcing import forcing_tools

forcing_tools.download_and_cache_era5(
    time_range=('2024-01-01', '2024-01-31'),
    lon_range=(-80.0, -60.0),
    lat_range=(30.0, 50.0),
    output_path='era5_jan2024.nc',
    hours=['00:00', '06:00', '12:00', '18:00']
)

# Step 2: Use cached file (in scripts/surface_forcing.py)
USE_API = False
NETCDF_PATHS = 'era5_jan2024.nc'
```

**Pros:** ✅ Fast after first download, ✅ Reusable, ✅ Best of both worlds  
**Cons:** ❌ Requires local storage

---

## Quick Commands - ERA5

### Check if CDS API is installed
```bash
python -c "import cdsapi; print('✅ CDS API available')"
```

### Setup CDS credentials
```bash
# Create ~/.cdsapirc with:
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
```

Get your credentials from: https://cds.climate.copernicus.eu/user

### Test API access
```python
from download import ERA5Downloader

downloader = ERA5Downloader()
downloader.download_era5_surface_forcing(
    output_path='test_era5.nc',
    lon_range=(-75.0, -70.0),
    lat_range=(38.0, 42.0),
    time_range=('2024-01-01', '2024-01-02'),
    hours=['00:00', '12:00']
)
```

---

## Configuration Parameters Comparison

### Ocean Data (GLORYS)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `USE_API` | bool | Use Marine API | `False` |
| `NETCDF_PATH` | str | Path to NetCDF | `'glorys.nc'` |
| `API_LON_RANGE` | tuple | (min, max) longitude | `(-80, -60)` |
| `API_LAT_RANGE` | tuple | (min, max) latitude | `(30, 50)` |
| `API_TIME_BUFFER_DAYS` | int | Days ± init_time | `1` |

### Atmospheric Data (ERA5)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `USE_API` | bool | Use CDS API | `False` |
| `NETCDF_PATHS` | str/dict | Path(s) to NetCDF | `'era5.nc'` |
| `API_LON_RANGE` | tuple | (min, max) longitude | `(-80, -60)` |
| `API_LAT_RANGE` | tuple | (min, max) latitude | `(30, 50)` |
| `API_HOURS` | list | Hours to download | `['00:00', '12:00']` |
| `API_INCLUDE_RADIATION` | bool | Include radiation vars | `True` |

---

## Troubleshooting Both APIs

| Problem | Ocean (GLORYS) | Atmospheric (ERA5) |
|---------|----------------|-------------------|
| Package not found | `pip install copernicusmarine` | `pip install cdsapi` |
| Authentication | `copernicusmarine login` | Create `~/.cdsapirc` |
| Slow downloads | Reduce extent/time | Download off-peak hours |
| File exists error | Set `force_download=True` | Set `force_download=True` |

---

## Examples Documentation

See detailed examples in:

**Ocean Data:**
- `scripts/example_data_sources.py` - 5 GLORYS examples
- `scripts/README_INITIALIZATION.md` - Full documentation
- `docs/api/data_sources.rst` - API reference

**Atmospheric Data:**
- `scripts/example_era5_forcing.py` - 6 ERA5 examples
- `scripts/README_SURFACE_FORCING.md` - Full documentation
- `docs/api/era5_forcing.rst` - API reference

---

## Next Steps

1. ✅ Install dependencies:
   - Ocean: `pip install copernicusmarine` (optional)
   - Atmospheric: `pip install cdsapi` (optional)
2. ✅ Configure scripts with your preferred method
3. ✅ Run scripts:
   - Initialization: `python scripts/initialize.py`
   - Surface forcing: `python scripts/surface_forcing.py`
4. ✅ Reuse functions for boundary conditions and climatology

---

**Need help?** 
- Ocean data: Check `scripts/README_INITIALIZATION.md`
- Atmospheric data: Check `scripts/README_SURFACE_FORCING.md`
