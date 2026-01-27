Grid Generation
===============

This guide explains how to generate ROMS grids with bathymetry and coordinate systems.

Overview
--------

ROMS uses a curvilinear, orthogonal grid with an Arakawa C staggered grid system. The grid generation process includes:

1. Creating staggered grids (rho, u, v, psi points)
2. Computing grid metrics (pm, pn, angle)
3. Interpolating bathymetry
4. Creating land/sea masks
5. Applying iterative smoothing
6. Setting up vertical coordinates

Grid Types
----------

ROMS uses four grid types:

- **rho-points**: Centers of grid cells (temperature, salinity, tracers)
- **u-points**: Western faces of grid cells (u-velocity)
- **v-points**: Southern faces of grid cells (v-velocity)  
- **psi-points**: Corners of grid cells (vorticity)

Creating Staggered Grids
-------------------------

Use ``create_staggered_grids()`` to generate all grid types:

.. code-block:: python

   from grid import grid_tools
   
   grids = grid_tools.create_staggered_grids(
       lat_min=30.0,
       lat_max=45.0,
       lon_min=-80.0,
       lon_max=-65.0,
       dx_km=1.0,
       dy_km=1.0
   )
   
   # Access individual grids
   lon_rho = grids['lon_rho']  # (eta_rho, xi_rho)
   lat_rho = grids['lat_rho']  # (eta_rho, xi_rho)
   lon_u = grids['lon_u']      # (eta_rho, xi_u)
   lat_u = grids['lat_u']      # (eta_rho, xi_u)
   # ... etc

Computing Grid Metrics
----------------------

Calculate inverse grid spacings and rotation angles:

.. code-block:: python

   metrics = grid_tools.compute_grid_metrics(grids)
   
   pm = metrics['pm']        # 1/dx in xi direction (m^-1)
   pn = metrics['pn']        # 1/dy in eta direction (m^-1)
   dndx = metrics['dndx']    # d(pn)/dx
   dmde = metrics['dmde']    # d(pm)/dy
   angle = metrics['angle']  # Rotation angle (radians)

Interpolating Bathymetry
-------------------------

Interpolate bathymetry from source data to the ROMS grid:

.. code-block:: python

   import xarray as xr
   
   # Load bathymetry dataset
   bathy = xr.open_dataset('etopo_subset.nc')
   
   # Interpolate to rho-points
   h = grid_tools.interpolate_bathymetry(
       bathy_dataset=bathy,
       target_lon=grids['lon_rho'],
       target_lat=grids['lat_rho'],
       depth_var='elevation',
       lat_var='lat',
       lon_var='lon',
       min_depth=5.0,      # Minimum depth (m)
       max_depth=5500.0,   # Maximum depth (m)
       fill_value=10.0     # Default depth for land
   )

Creating Masks
--------------

Generate land/sea masks for all grid types:

.. code-block:: python

   masks = grid_tools.create_masks(h, grids)
   
   mask_rho = masks['mask_rho']  # 1=water, 0=land
   mask_u = masks['mask_u']
   mask_v = masks['mask_v']
   mask_psi = masks['mask_psi']

Iterative Smoothing
--------------------

Apply iterative smoothing to reduce steep bathymetric gradients:

.. code-block:: python

   h_smooth = grid_tools.iterative_smoothing(
       h=h,
       max_iterations=500,
       r_max=0.35,         # Maximum slope parameter
       use_log=True        # Smooth log(h) for better results
   )

The smoothing ensures the bathymetry satisfies ROMS stability criteria:

.. math::

   r = \\frac{|h_{i+1} - h_i|}{h_{i+1} + h_i} < r_{max}

Vertical Coordinates
--------------------

ROMS uses terrain-following vertical coordinates. Set up with:

.. code-block:: python

   # In create_roms_grid_dataset()
   grid_ds = grid_tools.create_roms_grid_dataset(
       grids=grids,
       h=h_smooth,
       theta_s=5.0,      # Surface stretching parameter
       theta_b=2.0,      # Bottom stretching parameter  
       hc=250.0,         # Critical depth (m)
       n_levels=40       # Number of vertical levels
   )

Parameters:

- **theta_s**: Controls surface resolution (0-10, higher = more surface levels)
- **theta_b**: Controls bottom resolution (0-4, higher = more bottom levels)
- **hc**: Critical depth where stretching transitions
- **n_levels**: Total number of vertical levels (s_rho)

Complete Example
----------------

Full workflow to generate a ROMS grid:

.. code-block:: python

   from grid import grid_tools
   import xarray as xr
   
   # 1. Create staggered grids
   grids = grid_tools.create_staggered_grids(
       lat_min=30.0, lat_max=45.0,
       lon_min=-80.0, lon_max=-65.0,
       dx_km=1.0, dy_km=1.0
   )
   
   # 2. Compute metrics
   metrics = grid_tools.compute_grid_metrics(grids)
   
   # 3. Interpolate bathymetry
   bathy = xr.open_dataset('etopo_subset.nc')
   h = grid_tools.interpolate_bathymetry(
       bathy, grids['lon_rho'], grids['lat_rho'],
       depth_var='elevation', min_depth=5.0, max_depth=5500.0
   )
   
   # 4. Apply smoothing
   h_smooth = grid_tools.iterative_smoothing(
       h, max_iterations=500, r_max=0.35, use_log=True
   )
   
   # 5. Create masks
   masks = grid_tools.create_masks(h_smooth, grids)
   
   # 6. Create full dataset
   grid_ds = grid_tools.create_roms_grid_dataset(
       grids, h_smooth,
       theta_s=5.0, theta_b=2.0, hc=250.0, n_levels=40
   )
   
   # 7. Save to file
   grid_ds.to_netcdf('roms_grid.nc')

Using the Script
----------------

Alternatively, use the pre-configured script:

.. code-block:: bash

   cd scripts/
   python grid_generation.py

Edit the location-specific parameters at the top of the script:

.. code-block:: python

   # Location-specific parameters
   lat_min, lat_max = 30.0, 45.0
   lon_min, lon_max = -80.0, -65.0
   dx_km, dy_km = 1.0, 1.0
   
   # Vertical coordinate parameters
   theta_s = 5.0
   theta_b = 2.0
   hc = 250.0
   n_levels = 40
   
   # File paths
   bathy_file = '../output/east_coast_bathy.nc'
   output_file = '../output/roms_grid.nc'

API Reference
-------------

See :doc:`../api/grid` for complete API documentation.
