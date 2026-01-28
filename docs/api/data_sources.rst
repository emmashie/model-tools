Data Source Access
==================

The model-tools package supports multiple methods for accessing ocean data for initialization, boundary conditions, and forcing.

Copernicus Marine Data Access
------------------------------

The package includes support for the Copernicus Marine Service, which provides access to various ocean reanalysis and forecast products, including GLORYS.

Installation
^^^^^^^^^^^^

To use the Copernicus Marine API, you need to install the ``copernicusmarine`` package::

    pip install copernicusmarine

You also need to create a free account at https://marine.copernicus.eu/ and configure your credentials::

    copernicusmarine login

Configuration Methods
^^^^^^^^^^^^^^^^^^^^^

The initialization script (``scripts/initialize.py``) supports two methods:

1. **Local NetCDF File** (Default)
   
   Use pre-downloaded NetCDF files::

       USE_API = False
       NETCDF_PATH = '/path/to/your/glorys_data.nc'

2. **Remote API Access**
   
   Access data directly from Copernicus Marine Service::

       USE_API = True
       API_LON_RANGE = (-80.0, -60.0)  # or None to auto-detect from grid
       API_LAT_RANGE = (30.0, 50.0)    # or None to auto-detect from grid
       API_TIME_BUFFER_DAYS = 1        # days before/after init_time

Using the API in Python
^^^^^^^^^^^^^^^^^^^^^^^

Basic usage for loading GLORYS data::

    from initialization import init_tools
    import numpy as np

    # Load data via API
    ds = init_tools.load_glorys_data(
        init_time='2024-01-01',
        lon_range=(-80.0, -60.0),
        lat_range=(30.0, 50.0),
        use_api=True,
        time_buffer_days=1
    )

    # Or load from existing NetCDF file
    ds = init_tools.load_glorys_data(
        init_time='2024-01-01',
        lon_range=None,
        lat_range=None,
        use_api=False,
        netcdf_path='/path/to/glorys_data.nc'
    )

Download and cache data for later use::

    from initialization import init_tools

    # Download data and save to file
    file_path = init_tools.download_and_cache_glorys(
        init_time='2024-01-01',
        lon_range=(-80.0, -60.0),
        lat_range=(30.0, 50.0),
        output_path='cached_glorys_jan2024.nc',
        time_buffer_days=3,
        force_download=False  # Set True to re-download
    )

    # Later, use the cached file
    ds = init_tools.load_glorys_data(
        init_time='2024-01-01',
        lon_range=None,
        lat_range=None,
        use_api=False,
        netcdf_path=file_path
    )

Direct API Access
^^^^^^^^^^^^^^^^^

For more control, use the CopernicusMarineDownloader class directly::

    from download import CopernicusMarineDownloader

    downloader = CopernicusMarineDownloader()

    # Open dataset remotely (no download)
    ds = downloader.open_dataset(
        dataset_id='cmems_mod_glo_phy_my_0.083deg_P1D-m',
        variables=['thetao', 'so', 'uo', 'vo', 'zos'],
        minimum_longitude=-80.0,
        maximum_longitude=-60.0,
        minimum_latitude=30.0,
        maximum_latitude=50.0,
        start_datetime='2024-01-01',
        end_datetime='2024-01-31',
        minimum_depth=0.0,
        maximum_depth=5000.0
    )

    # Or download to file
    file_path = downloader.download_dataset(
        dataset_id='cmems_mod_glo_phy_my_0.083deg_P1D-m',
        output_path='glorys_data.nc',
        variables=['thetao', 'so', 'uo', 'vo', 'zos'],
        minimum_longitude=-80.0,
        maximum_longitude=-60.0,
        minimum_latitude=30.0,
        maximum_latitude=50.0,
        start_datetime='2024-01-01',
        end_datetime='2024-01-31'
    )

    # Convenience function for GLORYS
    ds = downloader.get_glorys_dataset(
        lon_range=(-80.0, -60.0),
        lat_range=(30.0, 50.0),
        time_range=('2024-01-01', '2024-01-31'),
        use_daily=True  # or False for monthly means
    )

Available GLORYS Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^

The package supports two GLORYS product types:

- **Daily data**: ``cmems_mod_glo_phy_my_0.083deg_P1D-m``
  
  - Higher temporal resolution
  - Use for initialization and short-term forcing
  - Set ``use_daily=True``

- **Monthly means**: ``cmems_mod_glo_phy_my_0.083deg_P1M-m``
  
  - Climatological forcing
  - Smaller file sizes
  - Set ``use_daily=False``

Standard Variables
^^^^^^^^^^^^^^^^^^

The default variables used for initialization:

- ``thetao``: Potential temperature (°C)
- ``so``: Salinity (PSU)
- ``uo``: Eastward velocity (m/s)
- ``vo``: Northward velocity (m/s)
- ``zos``: Sea surface height (m)

Additional variables available in GLORYS:

- ``mlotst``: Mixed layer depth
- ``bottomT``: Bottom temperature
- ``siconc``: Sea ice concentration
- And many more...

Reusable Components for Boundary Conditions
--------------------------------------------

The API access functions in ``download.py`` and ``initialization.py`` are designed to be reusable for:

- Boundary condition generation
- Surface forcing
- Climatology creation
- Any other ocean data needs

Example workflow for boundary conditions::

    from download import CopernicusMarineDownloader
    import xarray as xr

    downloader = CopernicusMarineDownloader()

    # Download multiple months for boundary conditions
    for month in range(1, 13):
        start = f'2024-{month:02d}-01'
        if month == 12:
            end = '2024-12-31'
        else:
            end = f'2024-{month+1:02d}-01'
        
        ds = downloader.get_glorys_dataset(
            lon_range=(-80.0, -60.0),
            lat_range=(30.0, 50.0),
            time_range=(start, end)
        )
        
        # Process boundary conditions...
        # (boundary condition code here)

Troubleshooting
---------------

**Authentication errors**

Make sure you've logged in::

    copernicusmarine login

**Dataset not found errors**

Check the dataset ID is correct. You can browse available datasets at:
https://data.marine.copernicus.eu/products

**Slow downloads**

- Use smaller spatial/temporal ranges
- Consider downloading once and caching the file
- Use monthly means instead of daily data when appropriate

**Memory issues**

For large domains or long time periods:

- Download data in chunks
- Use the ``minimum_depth`` and ``maximum_depth`` parameters to limit vertical extent
- Process one time step at a time

References
----------

- Copernicus Marine Service: https://marine.copernicus.eu/
- Copernicus Marine Toolbox Documentation: https://help.marine.copernicus.eu/en/collections/4060068-copernicus-marine-toolbox
- GLORYS Product: https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/
