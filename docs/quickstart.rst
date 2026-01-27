Quick Start Guide
=================

This guide walks through the typical workflow for preparing ROMS input files.

Workflow Overview
-----------------

The typical preprocessing workflow consists of four steps:

1. **Download bathymetry data** and subset to your domain
2. **Generate ROMS grid** with bathymetry and metrics
3. **Create initial conditions** from ocean reanalysis data
4. **Generate surface forcing** from atmospheric reanalysis
5. **Create boundary conditions** from ocean reanalysis data

Step 1: Download Bathymetry
----------------------------

Download and subset bathymetry data for your domain:

.. code-block:: python

   from download import Downloader
   
   downloader = Downloader()
   
   # Download ETOPO bathymetry
   downloader.download_file(
       'https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/netcdf/ETOPO1_Ice_g_gmt4.grd.gz',
       'etopo1.nc.gz'
   )
   
   # Subset to domain
   downloader.subset_dataset(
       'etopo1.nc.gz',
       'my_domain_bathy.nc',
       lat_range=(30, 45),
       lon_range=(-80, -65)
   )

Step 2: Generate ROMS Grid
---------------------------

Create a ROMS grid with bathymetry and metrics:

.. code-block:: python

   from grid import grid_tools
   import xarray as xr
   import numpy as np
   
   # Define grid parameters
   lat_min, lat_max = 30.0, 45.0
   lon_min, lon_max = -80.0, -65.0
   dx_km, dy_km = 1.0, 1.0
   
   # Create staggered grids
   grids = grid_tools.create_staggered_grids(
       lat_min, lat_max, lon_min, lon_max, dx_km, dy_km
   )
   
   # Interpolate bathymetry
   bathy_data = xr.open_dataset('my_domain_bathy.nc')
   h = grid_tools.interpolate_bathymetry(
       bathy_data, grids['lon_rho'], grids['lat_rho'],
       depth_var='elevation', min_depth=5.0, max_depth=5500.0
   )
   
   # Apply iterative smoothing
   h_smooth = grid_tools.iterative_smoothing(
       h, max_iterations=500, r_max=0.35
   )
   
   # Create full ROMS grid dataset
   grid_ds = grid_tools.create_roms_grid_dataset(
       grids, h_smooth,
       theta_s=5.0, theta_b=2.0, hc=250.0, n_levels=40
   )
   
   grid_ds.to_netcdf('roms_grid.nc')

Step 3: Create Initial Conditions
----------------------------------

Generate initial conditions from GLORYS ocean reanalysis:

.. code-block:: python

   from initialization import init_tools
   import xarray as xr
   
   # Load grid and data
   grid = xr.open_dataset('roms_grid.nc')
   glorys_data = xr.open_dataset('glorys_data.nc')
   
   # Prepare coordinates
   source_coords = init_tools.prepare_source_coords(
       glorys_data, 'depth', 'latitude', 'longitude'
   )
   
   # Interpolate variables (temp, salt, u, v, zeta)
   # ... (see initialization workflow for details)
   
   # Create initial conditions dataset
   ic_ds = init_tools.create_initial_conditions_dataset(
       grid, temp, salt, u, v, w, ubar, vbar, zeta,
       init_time='2023-01-01', source_name='GLORYS12v1'
   )
   
   ic_ds.to_netcdf('initial_conditions.nc')

Step 4: Generate Surface Forcing
---------------------------------

Process ERA5 atmospheric data for surface forcing:

.. code-block:: python

   from forcing import forcing_tools
   import xarray as xr
   
   # Load grid and ERA5 data
   grid = xr.open_dataset('roms_grid.nc')
   era5_data = xr.open_dataset('era5_data.nc')
   
   # Process ERA5 forcing
   forcing_vars = forcing_tools.process_era5_forcing(
       era5_data, grid,
       start_time='2023-01-01',
       end_time='2023-12-31'
   )
   
   # Create forcing dataset
   forcing_ds = forcing_tools.create_surface_forcing_dataset(
       grid, forcing_vars, 
       start_time='2023-01-01',
       end_time='2023-12-31'
   )
   
   forcing_ds.to_netcdf('surface_forcing.nc')

Step 5: Create Boundary Conditions
-----------------------------------

Generate boundary forcing from GLORYS:

.. code-block:: python

   from boundary import boundary_tools
   import xarray as xr
   
   # Specify active boundaries
   boundaries = {
       'west': False,
       'east': True,
       'north': True,
       'south': True
   }
   
   # Extract boundary transects
   boundary_transects = boundary_tools.extract_boundary_transects(
       temp, salt, u, v, ubar, vbar, zeta,
       grid, boundaries
   )
   
   # Create boundary dataset
   boundary_ds = boundary_tools.create_boundary_dataset(
       boundary_transects, time_days, grid,
       '2023-01-01', '2023-12-31'
   )
   
   boundary_ds.to_netcdf('boundary_forcing.nc')

Using Pre-configured Scripts
-----------------------------

Instead of using the API directly, you can use the pre-configured scripts in ``scripts/``:

.. code-block:: bash

   # 1. Download bathymetry
   python scripts/download_east_coast_bathy.py
   
   # 2. Generate grid
   python scripts/grid_generation.py
   
   # 3. Create initial conditions
   python scripts/initialize.py
   
   # 4. Generate surface forcing
   python scripts/surface_forcing.py
   
   # 5. Create boundary conditions
   python scripts/boundary_conditions.py

Each script has location-specific parameters at the top that you can customize for your domain.

Next Steps
----------

- Learn more about :doc:`workflows/grid_generation`
- Explore :doc:`workflows/initialization`
- Configure :doc:`workflows/surface_forcing`
- Set up :doc:`workflows/boundary_conditions`
