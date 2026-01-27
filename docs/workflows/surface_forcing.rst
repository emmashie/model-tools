Surface Forcing
===============

This guide explains how to create ROMS surface forcing files from atmospheric reanalysis data.

Overview
--------

Surface forcing provides atmospheric conditions that drive ocean circulation:

- **Surface wind stress** (sustr, svstr): Momentum flux [N/m²]
- **Surface heat flux** (shflux): Net heat flux [W/m²]
- **Freshwater flux** (swflux): Net water flux [m/s or kg/m²/s]
- **Shortwave radiation** (swrad): Solar radiation [W/m²]
- **Air temperature** (Tair): 2m air temperature [°C]
- **Surface pressure** (Pair): Sea level pressure [mb]
- **Humidity** (Qair): Specific humidity [kg/kg]
- **Cloud cover** (cloud): Fractional cloud cover [0-1]
- **Precipitation** (rain): Rainfall rate [kg/m²/s]

Data Sources
------------

Common atmospheric reanalysis products:

- **ERA5**: ECMWF Reanalysis (0.25° hourly)
- **CFSR/CFSv2**: NCEP Climate Forecast System
- **MERRA-2**: NASA Modern-Era Retrospective Analysis

Workflow
--------

1. Load ROMS grid and ERA5 data
2. Process ERA5 variables (unit conversions)
3. Interpolate to ROMS grid points
4. Create surface forcing dataset

Step-by-Step Example
---------------------

1. Load Data
~~~~~~~~~~~~

.. code-block:: python

   from forcing import forcing_tools
   from grid import grid_tools
   import xarray as xr
   
   # Load ROMS grid
   grid = xr.open_dataset('roms_grid.nc')
   
   # Load ERA5 data
   era5_data = xr.open_dataset('era5_2023.nc')

2. Process ERA5 Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

ERA5 variables require unit conversions for ROMS:

.. code-block:: python

   # Get time dimension name (ERA5 uses 'valid_time' or 'time')
   time_dim = 'valid_time' if 'valid_time' in era5_data.dims else 'time'
   
   # Convert 2m temperature: K → °C
   tair = era5_data['t2m'] - 273.15
   
   # Convert 2m dewpoint to specific humidity
   dewpoint = era5_data['d2m'] - 273.15
   qair = forcing_tools.dewpoint_to_specific_humidity(
       dewpoint.values,
       era5_data['sp'].values  # Surface pressure [Pa]
   )
   
   # Convert surface pressure: Pa → mb
   pair = era5_data['sp'] / 100.0
   
   # Compute wind stress from 10m winds
   u10 = era5_data['u10']
   v10 = era5_data['v10']
   sustr, svstr = forcing_tools.compute_wind_stress(
       u10.values, v10.values
   )
   
   # Net shortwave radiation (already in W/m²)
   swrad = era5_data['msdwswrf']
   
   # Total cloud cover (0-1)
   cloud = era5_data['tcc']
   
   # Precipitation: m → kg/m²/s
   rain = era5_data['tp'] / 3600.0  # Assuming hourly accumulation

3. Prepare Forcing Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get ERA5 coordinates
   era5_lon = era5_data['longitude'].values
   era5_lat = era5_data['latitude'].values
   
   # Create 2D coordinate grids
   era5_lon_2d, era5_lat_2d = np.meshgrid(
       era5_lon, era5_lat, indexing='xy'
   )
   
   # Prepare coordinate dictionary
   source_coords = {
       'lon_2d': era5_lon_2d,
       'lat_2d': era5_lat_2d
   }

4. Interpolate to ROMS Grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from interpolate import interp_tools
   
   nt = len(era5_data[time_dim])
   eta_rho, xi_rho = grid.lon_rho.shape
   
   # Initialize arrays
   forcing_vars = {
       'tair': np.empty((nt, eta_rho, xi_rho)),
       'qair': np.empty((nt, eta_rho, xi_rho)),
       'pair': np.empty((nt, eta_rho, xi_rho)),
       'sustr': np.empty((nt, eta_rho, xi_rho)),
       'svstr': np.empty((nt, eta_rho, xi_rho)),
       'swrad': np.empty((nt, eta_rho, xi_rho)),
       'cloud': np.empty((nt, eta_rho, xi_rho)),
       'rain': np.empty((nt, eta_rho, xi_rho)),
   }
   
   # Interpolate each time step
   for t in range(nt):
       forcing_vars['tair'][t] = interp_tools.interp2d(
           tair[t].values,
           source_coords['lon_2d'],
           source_coords['lat_2d'],
           grid.lon_rho.values,
           grid.lat_rho.values
       )
       
       # Repeat for other variables...

5. Use Batch Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The forcing_tools provides a batch interpolation method:

.. code-block:: python

   # Prepare forcing coordinate dict
   forcing_coords = forcing_tools.prepare_forcing_coords(
       era5_lon, era5_lat
   )
   
   # Batch interpolate all variables
   roms_forcing = forcing_tools.interpolate_forcing_timeseries(
       forcing_vars={
           'tair': tair.values,
           'qair': qair,
           'pair': pair.values,
           'sustr': sustr,
           'svstr': svstr,
           'swrad': swrad.values,
           'cloud': cloud.values,
           'rain': rain,
       },
       source_coords=forcing_coords,
       target_lon=grid.lon_rho.values,
       target_lat=grid.lat_rho.values
   )

6. Create Forcing Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   forcing_ds = forcing_tools.create_surface_forcing_dataset(
       grid=grid,
       forcing_vars=roms_forcing,
       time_array=era5_data[time_dim].values,
       start_time='2023-01-01',
       end_time='2023-12-31',
       source_name='ERA5'
   )
   
   # Save to file
   forcing_ds.to_netcdf('surface_forcing.nc')

Using process_era5_forcing
---------------------------

For convenience, use the integrated processing function:

.. code-block:: python

   forcing_vars, time_days = forcing_tools.process_era5_forcing(
       era5_data=era5_data,
       grid=grid,
       start_time='2023-01-01',
       end_time='2023-12-31'
   )
   
   forcing_ds = forcing_tools.create_surface_forcing_dataset(
       grid, forcing_vars, time_days,
       '2023-01-01', '2023-12-31',
       source_name='ERA5'
   )

Using the Script
----------------

Use the pre-configured script:

.. code-block:: bash

   cd scripts/
   python surface_forcing.py

Edit location-specific parameters:

.. code-block:: python

   # LOCATION-SPECIFIC PARAMETERS
   
   # Time range
   start_time = '2023-01-01'
   end_time = '2023-12-31'
   
   # File paths
   grid_file = '../output/roms_grid.nc'
   era5_pattern = '/path/to/ERA5/era5_*.nc'
   output_file = '../output/surface_forcing.nc'
   
   # ERA5 variable names
   era5_vars = {
       't2m': 't2m',           # 2m temperature
       'd2m': 'd2m',           # 2m dewpoint
       'sp': 'sp',             # Surface pressure
       'u10': 'u10',           # 10m U wind
       'v10': 'v10',           # 10m V wind
       'msdwswrf': 'msdwswrf', # Shortwave radiation
       'tcc': 'tcc',           # Total cloud cover
       'tp': 'tp',             # Total precipitation
   }

Unit Conversions
----------------

Key unit conversions performed:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Variable
     - ERA5 Units
     - ROMS Units
   * - Temperature
     - K
     - °C (subtract 273.15)
   * - Pressure
     - Pa
     - mb (divide by 100)
   * - Precipitation
     - m/hour
     - kg/m²/s (divide by 3600)
   * - Wind stress
     - From u10, v10
     - N/m² (bulk formula)
   * - Humidity
     - Dewpoint (K)
     - Specific humidity (kg/kg)

Wind Stress Calculation
------------------------

Wind stress is computed using a bulk aerodynamic formula:

.. math::

   \\tau = \\rho_{air} C_D |\\mathbf{U}_{10}| \\mathbf{U}_{10}

Where:

- :math:`\\rho_{air} = 1.22` kg/m³ (air density)
- :math:`C_D` = drag coefficient (depends on wind speed)
- :math:`\\mathbf{U}_{10}` = 10m wind vector

Output Format
-------------

The surface forcing file contains:

.. code-block:: text

   Dimensions:
     sms_time: 8760  # Hourly for 1 year
     eta_rho: 500
     xi_rho: 300
   
   Variables:
     Tair(sms_time, eta_rho, xi_rho)    # Air temperature [°C]
     Qair(sms_time, eta_rho, xi_rho)    # Specific humidity [kg/kg]
     Pair(sms_time, eta_rho, xi_rho)    # Pressure [mb]
     sustr(sms_time, eta_rho, xi_rho)   # U wind stress [N/m²]
     svstr(sms_time, eta_rho, xi_rho)   # V wind stress [N/m²]
     swrad(sms_time, eta_rho, xi_rho)   # Shortwave radiation [W/m²]
     cloud(sms_time, eta_rho, xi_rho)   # Cloud fraction [0-1]
     rain(sms_time, eta_rho, xi_rho)    # Precipitation [kg/m²/s]
     sms_time(sms_time)                  # days since reference

API Reference
-------------

See :doc:`../api/forcing` for complete API documentation.
