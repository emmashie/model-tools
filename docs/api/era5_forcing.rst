ERA5 Surface Forcing Data Access
==================================

The model-tools package supports multiple methods for accessing ERA5 reanalysis data for surface forcing.

Copernicus Climate Data Store (CDS) API
----------------------------------------

The package includes support for the Copernicus Climate Data Store, which provides access to ERA5 reanalysis data.

Installation
^^^^^^^^^^^^

To use the CDS API, you need to install the ``cdsapi`` package::

    pip install cdsapi

You also need to create a free account at https://cds.climate.copernicus.eu/ and configure your credentials.

Create a file ``~/.cdsapirc`` with your credentials::

    url: https://cds.climate.copernicus.eu/api/v2
    key: YOUR_UID:YOUR_API_KEY

Get your UID and API key from: https://cds.climate.copernicus.eu/user

Configuration Methods
^^^^^^^^^^^^^^^^^^^^^

The surface forcing script (``scripts/surface_forcing.py``) supports two methods:

1. **Local NetCDF Files** (Default)
   
   Use pre-downloaded NetCDF files::

       USE_API = False
       NETCDF_PATHS = {
           'main': '/path/to/ERA5_data.nc',
           'pair': '/path/to/ERA5_Pair.nc',
           'rad': '/path/to/ERA5_rad.nc'
       }
       # Or single file:
       # NETCDF_PATHS = '/path/to/era5_all.nc'

2. **Remote API Access**
   
   Access data directly from CDS::

       USE_API = True
       API_LON_RANGE = None  # Auto-detect from grid
       API_LAT_RANGE = None  # Auto-detect from grid
       API_HOURS = ['00:00', '06:00', '12:00', '18:00']
       API_INCLUDE_RADIATION = True

Using the API in Python
^^^^^^^^^^^^^^^^^^^^^^^

Basic usage for loading ERA5 data::

    from forcing import forcing_tools
    import numpy as np

    # Load data via API
    ds = forcing_tools.load_era5_data(
        time_range=('2024-01-01', '2024-01-31'),
        lon_range=(-80.0, -60.0),
        lat_range=(30.0, 50.0),
        use_api=True,
        hours=['00:00', '06:00', '12:00', '18:00'],
        include_radiation=True
    )

    # Or load from existing NetCDF file
    ds = forcing_tools.load_era5_data(
        time_range=('2024-01-01', '2024-01-31'),
        lon_range=None,
        lat_range=None,
        use_api=False,
        netcdf_paths='/path/to/era5_data.nc'
    )

Download and cache data for later use::

    from forcing import forcing_tools

    # Download data and save to file
    file_path = forcing_tools.download_and_cache_era5(
        time_range=('2024-01-01', '2024-01-31'),
        lon_range=(-80.0, -60.0),
        lat_range=(30.0, 50.0),
        output_path='cached_era5_jan2024.nc',
        hours=['00:00', '06:00', '12:00', '18:00'],
        include_radiation=True,
        force_download=False
    )

    # Later, use the cached file
    ds = forcing_tools.load_era5_data(
        time_range=('2024-01-01', '2024-01-31'),
        lon_range=None,
        lat_range=None,
        use_api=False,
        netcdf_paths=file_path
    )

Direct API Access
^^^^^^^^^^^^^^^^^

For more control, use the ERA5Downloader class directly::

    from download import ERA5Downloader

    downloader = ERA5Downloader()

    # Download specific variables
    downloader.download_era5_data(
        output_path='era5_custom.nc',
        variables=[
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            '2m_temperature',
            '2m_dewpoint_temperature',
            'mean_sea_level_pressure',
            'total_precipitation'
        ],
        lon_range=(-80.0, -60.0),
        lat_range=(30.0, 50.0),
        time_range=('2024-01-01', '2024-01-31'),
        hours=['00:00', '06:00', '12:00', '18:00']
    )

    # Convenience function for standard surface forcing
    downloader.download_era5_surface_forcing(
        output_path='era5_forcing.nc',
        lon_range=(-80.0, -60.0),
        lat_range=(30.0, 50.0),
        time_range=('2024-01-01', '2024-01-31'),
        hours=['00:00', '06:00', '12:00', '18:00'],
        include_radiation=True
    )

ERA5 Variables
^^^^^^^^^^^^^^

Standard surface forcing variables automatically downloaded:

- ``u10`` / ``10m_u_component_of_wind``: 10m u-wind component
- ``v10`` / ``10m_v_component_of_wind``: 10m v-wind component
- ``t2m`` / ``2m_temperature``: 2m air temperature
- ``d2m`` / ``2m_dewpoint_temperature``: 2m dewpoint temperature
- ``msl`` / ``mean_sea_level_pressure``: Mean sea level pressure
- ``tp`` / ``total_precipitation``: Total precipitation
- ``sst`` / ``sea_surface_temperature``: Sea surface temperature
- ``ssrd`` / ``surface_solar_radiation_downwards``: Shortwave radiation (downward)
- ``strd`` / ``surface_thermal_radiation_downwards``: Longwave radiation (downward)
- ``ssr`` / ``surface_net_solar_radiation``: Net shortwave radiation
- ``str`` / ``surface_net_thermal_radiation``: Net longwave radiation

Variable Naming
^^^^^^^^^^^^^^^

ERA5 uses two naming conventions:

- **Short names**: Used in NetCDF files (e.g., ``u10``, ``t2m``, ``msl``)
- **Long names**: Used in API requests (e.g., ``10m_u_component_of_wind``)

The package handles both automatically. Use ``ERA5Downloader.get_era5_variable_mapping()`` for a complete mapping.

Temporal Resolution
^^^^^^^^^^^^^^^^^^^

ERA5 provides hourly data. You can control the temporal resolution when downloading:

- **Hourly** (24 per day): ``hours=None``
- **6-hourly** (4 per day): ``hours=['00:00', '06:00', '12:00', '18:00']``
- **Twice daily** (2 per day): ``hours=['00:00', '12:00']``
- **Custom**: Specify any combination of hours

Higher temporal resolution = larger downloads but better representation of diurnal cycles and high-frequency events.

Workflow Recommendations
------------------------

For Testing/Development
^^^^^^^^^^^^^^^^^^^^^^^

Use the API directly with small domains and short time periods::

    USE_API = True
    API_LON_RANGE = (-75.0, -70.0)  # Small region
    API_LAT_RANGE = (38.0, 42.0)
    API_HOURS = ['00:00', '12:00']  # Twice daily
    start_time = np.datetime64('2024-01-01')
    end_time = np.datetime64('2024-01-03')  # Just 2 days

For Production Runs
^^^^^^^^^^^^^^^^^^^^

Use the download-and-cache approach::

    # Step 1: Download once
    forcing_tools.download_and_cache_era5(
        time_range=('2024-01-01', '2024-12-31'),
        lon_range=(-80.0, -60.0),
        lat_range=(30.0, 50.0),
        output_path='era5_2024_full.nc',
        hours=['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
    )
    
    # Step 2: Use cached file
    USE_API = False
    NETCDF_PATHS = 'era5_2024_full.nc'

For HPC Environments
^^^^^^^^^^^^^^^^^^^^

- Download data on login nodes (network access)
- Copy to fast scratch storage
- Use cached NetCDF files on compute nodes

Troubleshooting
---------------

**Authentication errors**

Make sure you've created ``~/.cdsapirc`` with your credentials::

    url: https://cds.climate.copernicus.eu/api/v2
    key: YOUR_UID:YOUR_API_KEY

**Request queued**

CDS queues requests during high demand. Be patient - large requests can take hours. Consider:

- Downloading during off-peak hours
- Splitting large requests into smaller chunks
- Using the download-and-cache workflow

**Slow downloads**

To optimize download speed:

- Reduce spatial extent
- Reduce temporal resolution (fewer hours per day)
- Download fewer variables
- Split into monthly chunks for long time series

**Variable not found**

Check variable names:

- Use long names in API requests: ``'10m_u_component_of_wind'``
- Check available variables: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels

**Memory issues**

For large downloads:

- Process in monthly chunks
- Reduce temporal resolution
- Limit spatial extent
- Use compute nodes with more memory

Unit Conversions
----------------

The package automatically handles unit conversions for ROMS:

- Temperature: Kelvin → Celsius
- Pressure: Pascal → millibar
- Radiation: Accumulated values → flux density (W/m²)
- Precipitation: Meters → m/s
- Wind: Already in m/s

References
----------

- Copernicus Climate Data Store: https://cds.climate.copernicus.eu/
- ERA5 Documentation: https://confluence.ecmwf.int/display/CKB/ERA5
- CDS API Guide: https://cds.climate.copernicus.eu/api-how-to
- ERA5 Single Levels: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
