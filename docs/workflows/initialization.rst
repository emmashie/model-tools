Initial Conditions
==================

This guide explains how to create ROMS initial condition files from ocean reanalysis data.

Overview
--------

Initial conditions specify the ocean state at the start of a ROMS simulation. This includes:

- **Temperature** (temp): Potential temperature [°C]
- **Salinity** (salt): Practical salinity [PSU]
- **U-velocity** (u): Eastward velocity [m/s]
- **V-velocity** (v): Northward velocity [m/s]
- **W-velocity** (w): Vertical velocity [m/s]
- **Barotropic velocities** (ubar, vbar): Depth-integrated velocities [m/s]
- **Sea surface height** (zeta): Free surface elevation [m]

Data Sources
------------

Common ocean reanalysis products:

- **GLORYS12v1**: Copernicus Marine Service global reanalysis (1/12°)
- **HYCOM**: Hybrid Coordinate Ocean Model output
- **SODA**: Simple Ocean Data Assimilation

Workflow
--------

1. Load ROMS grid and source data
2. Prepare source coordinates
3. Compute ROMS vertical coordinates
4. Interpolate variables to ROMS grid
5. Compute derived variables (ubar, vbar, w)
6. Create initial conditions dataset

Step-by-Step Example
---------------------

1. Load Data
~~~~~~~~~~~~

.. code-block:: python

   from initialization import init_tools
   from grid import grid_tools
   import xarray as xr
   
   # Load ROMS grid
   grid = xr.open_dataset('roms_grid.nc')
   
   # Load GLORYS data
   glorys_data = xr.open_dataset('glorys_2023-01-01.nc')

2. Prepare Coordinates
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prepare source coordinates
   source_coords = init_tools.prepare_source_coords(
       glorys_data,
       depth_var='depth',
       lat_var='latitude',
       lon_var='longitude'
   )
   
   # Returns dictionary with:
   # - 'depth': 1D depth array
   # - 'lat': 1D latitude array
   # - 'lon': 1D longitude array
   # - 'lat_2d': 2D latitude grid
   # - 'lon_2d': 2D longitude grid
   # - 'depth_3d': 3D depth array

3. Compute ROMS Vertical Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get grid dimensions
   eta_rho, xi_rho, eta_v, xi_u, s_rho, s_w = \
       grid_tools.get_grid_dims(grid)
   
   # Compute z-coordinates at rho-points
   z_rho = grid_tools.compute_z(
       grid.sigma_r.values,
       grid.hc,
       grid.Cs_r.values,
       grid.h.values,
       np.zeros((eta_rho, xi_rho, 1))  # zeta=0 for initial conditions
   )
   
   # Shape: (s_rho, eta_rho, xi_rho)
   z_rho = np.squeeze(z_rho)
   z_rho = np.transpose(z_rho, (1, 2, 0))
   
   # Convert to positive depths
   roms_depth_3d = np.abs(z_rho)

4. Interpolate Variables
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get ROMS target coordinates
   roms_coords = {
       'lon_rho': grid.lon_rho.values,
       'lat_rho': grid.lat_rho.values,
       'lon_u': grid.lon_u.values,
       'lat_u': grid.lat_u.values,
       'lon_v': grid.lon_v.values,
       'lat_v': grid.lat_v.values,
   }
   
   # Interpolate temperature
   temp_data = glorys_data['thetao'][0, :, :, :].values
   temp = init_tools.interpolate_and_mask_3d(
       temp_data,
       source_coords,
       roms_coords['lon_rho'],
       roms_coords['lat_rho'],
       roms_depth_3d,
       grid.mask_rho.values,
       fill_value=5.0
   )
   
   # Interpolate salinity
   salt_data = glorys_data['so'][0, :, :, :].values
   salt = init_tools.interpolate_and_mask_3d(
       salt_data,
       source_coords,
       roms_coords['lon_rho'],
       roms_coords['lat_rho'],
       roms_depth_3d,
       grid.mask_rho.values,
       fill_value=32.0
   )
   
   # Interpolate velocities (u, v) to u/v-points
   u_data = glorys_data['uo'][0, :, :, :].values
   u = init_tools.interpolate_and_mask_3d(
       u_data,
       source_coords,
       roms_coords['lon_u'],
       roms_coords['lat_u'],
       roms_depth_3d,
       grid.mask_u.values,
       fill_value=0.0
   )
   
   v_data = glorys_data['vo'][0, :, :, :].values
   v = init_tools.interpolate_and_mask_3d(
       v_data,
       source_coords,
       roms_coords['lon_v'],
       roms_coords['lat_v'],
       roms_depth_3d,
       grid.mask_v.values,
       fill_value=0.0
   )
   
   # Interpolate sea surface height (2D)
   zeta_data = glorys_data['zos'][0, :, :].values
   zeta = init_tools.interpolate_and_mask_2d(
       zeta_data,
       source_coords['lon_2d'],
       source_coords['lat_2d'],
       roms_coords['lon_rho'],
       roms_coords['lat_rho'],
       fill_value=0.0
   )

5. Compute Derived Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from conversions import convert_tools
   
   # Transpose z_rho for conversions
   z_rho_t = np.transpose(z_rho, (2, 0, 1))  # (s_rho, eta_rho, xi_rho)
   
   # Compute barotropic velocities
   ubar, vbar = convert_tools.compute_uvbar(u, v, z_rho_t)
   
   # Compute vertical velocity
   w = convert_tools.compute_w(
       u, v,
       grid.pm.values,
       grid.pn.values,
       z_rho_t
   )

6. Create Initial Conditions Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ic_ds = init_tools.create_initial_conditions_dataset(
       grid=grid,
       temp=temp,
       salt=salt,
       u=u,
       v=v,
       w=w,
       ubar=ubar,
       vbar=vbar,
       zeta=zeta,
       init_time='2023-01-01T00:00:00',
       source_name='GLORYS12v1'
   )
   
   # Save to file
   ic_ds.to_netcdf('initial_conditions.nc')

Adding Deep Ocean Layer
------------------------

For stability, you may want to add a deep ocean layer below the source data:

.. code-block:: python

   # Define fill values for deep layer
   fill_values = {
       'thetao': 2.0,   # Deep ocean temperature
       'so': 34.7,      # Deep ocean salinity
       'uo': 0.0,       # No deep current
       'vo': 0.0,
   }
   
   # Add layer at 6000m depth
   extended_data = init_tools.add_deep_ocean_layer(
       glorys_data,
       new_depth=6000.0,
       fill_values=fill_values
   )

Using the Script
----------------

Use the pre-configured script:

.. code-block:: bash

   cd scripts/
   python initialize.py

Edit location-specific parameters:

.. code-block:: python

   # LOCATION-SPECIFIC PARAMETERS
   
   # Initialization time
   init_time = '2023-01-01T00:00:00'
   
   # File paths
   grid_file = '../output/roms_grid.nc'
   glorys_file = '/path/to/glorys_2023-01-01.nc'
   output_file = '../output/initial_conditions.nc'
   
   # Source variable names
   source_vars = {
       'temp': 'thetao',
       'salt': 'so',
       'u': 'uo',
       'v': 'vo',
       'zeta': 'zos',
       'depth': 'depth',
       'lat': 'latitude',
       'lon': 'longitude'
   }

Time Reference
--------------

ROMS uses time in days since a reference date (typically 2000-01-01). The ``ocean_time`` variable is automatically calculated:

.. code-block:: python

   seconds_since, days_since = init_tools.compute_time_since_reference(
       init_time='2023-01-01T00:00:00',
       ref_time='2000-01-01T00:00:00'
   )

Output Format
-------------

The initial conditions file contains:

.. code-block:: text

   Dimensions:
     ocean_time: 1
     s_rho: 40
     eta_rho: 500
     xi_rho: 300
     eta_v: 499
     xi_u: 299
   
   Variables:
     temp(ocean_time, s_rho, eta_rho, xi_rho)
     salt(ocean_time, s_rho, eta_rho, xi_rho)
     u(ocean_time, s_rho, eta_rho, xi_u)
     v(ocean_time, s_rho, eta_v, xi_rho)
     w(ocean_time, s_rho, eta_rho, xi_rho)
     ubar(ocean_time, eta_rho, xi_u)
     vbar(ocean_time, eta_v, xi_rho)
     zeta(ocean_time, eta_rho, xi_rho)
     ocean_time(ocean_time)  # days since reference

API Reference
-------------

See :doc:`../api/initialization` for complete API documentation.
