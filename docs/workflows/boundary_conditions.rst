Boundary Conditions
===================

This guide explains how to create ROMS boundary condition files from ocean reanalysis data.

Overview
--------

Boundary conditions specify the ocean state at open boundaries (edges of the domain). ROMS supports four boundaries:

- **Western boundary** (xi = 0)
- **Eastern boundary** (xi = end)
- **Southern boundary** (eta = 0)
- **Northern boundary** (eta = end)

Each boundary requires:

- Temperature and salinity (temp, salt)
- Velocities (u, v, ubar, vbar)
- Sea surface height (zeta)

Optionally, you can also create a climatology file with full 3D fields for nudging.

Workflow
--------

1. Load ROMS grid and source data
2. Specify which boundaries are active
3. Interpolate variables to ROMS grid (similar to initialization)
4. Extract boundary transects
5. Create boundary forcing dataset
6. Optionally create climatology dataset

Step-by-Step Example
---------------------

1. Specify Active Boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define which boundaries need forcing
   boundaries = {
       'west': False,   # Closed boundary
       'east': True,    # Open boundary
       'north': True,   # Open boundary
       'south': True,   # Open boundary
   }

2. Load and Interpolate Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from initialization import init_tools
   from conversions import convert_tools
   from boundary import boundary_tools
   import xarray as xr
   import numpy as np
   
   # Load grid and data
   grid = xr.open_dataset('roms_grid.nc')
   glorys_data = xr.open_mfdataset('glorys_2023_*.nc')
   
   # Subset to time range
   glorys_data = glorys_data.sel(
       time=slice('2023-01-01', '2023-12-31')
   )
   
   # Prepare coordinates
   source_coords = init_tools.prepare_source_coords(
       glorys_data, 'depth', 'latitude', 'longitude'
   )
   
   # Get ROMS vertical coordinates
   # ... (same as initialization workflow)
   
   # Interpolate variables for all time steps
   nt = len(glorys_data['time'])
   temp_interp = np.empty((nt, s_rho, eta_rho, xi_rho))
   salt_interp = np.empty((nt, s_rho, eta_rho, xi_rho))
   u_interp = np.empty((nt, s_rho, eta_rho, xi_u))
   v_interp = np.empty((nt, s_rho, eta_v, xi_rho))
   zeta_interp = np.empty((nt, eta_rho, xi_rho))
   
   for t in range(nt):
       # Interpolate each variable at time t
       # ... (same as initialization)
       pass
   
   # Compute derived variables
   ubar = np.empty((nt, eta_rho, xi_u))
   vbar = np.empty((nt, eta_v, xi_rho))
   w = np.empty((nt, s_rho, eta_rho, xi_rho))
   
   for t in range(nt):
       ubar[t], vbar[t] = convert_tools.compute_uvbar(
           u_interp[t], v_interp[t], z_rho_transposed
       )
       w[t] = convert_tools.compute_w(
           u_interp[t], v_interp[t],
           grid.pm.values, grid.pn.values, z_rho_transposed
       )

3. Extract Boundary Transects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   boundary_transects = boundary_tools.extract_boundary_transects(
       temp=temp_interp,
       salt=salt_interp,
       u=u_interp,
       v=v_interp,
       ubar=ubar,
       vbar=vbar,
       zeta=zeta_interp,
       grid=grid,
       boundaries=boundaries
   )
   
   # Returns dictionary with transects for active boundaries
   # e.g., boundary_transects['east']['temp'] has shape (nt, s_rho, eta_rho)

Boundary Transect Shapes
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Boundary
     - 3D Variables (temp, salt, u, v)
     - 2D Variables (zeta, ubar, vbar)
   * - West
     - (nt, s_rho, eta_rho)
     - (nt, eta_rho)
   * - East
     - (nt, s_rho, eta_rho)
     - (nt, eta_rho)
   * - South
     - (nt, s_rho, xi_rho)
     - (nt, xi_rho)
   * - North
     - (nt, s_rho, xi_rho)
     - (nt, xi_rho)

4. Create Boundary Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   
   # Compute time in days since reference
   ref_time = pd.Timestamp('2000-01-01')
   time_days = np.array([
       (pd.Timestamp(t) - ref_time).total_seconds() / 86400.0
       for t in glorys_data['time'].values
   ])
   
   # Create boundary forcing dataset
   boundary_ds = boundary_tools.create_boundary_dataset(
       boundary_transects=boundary_transects,
       time_days=time_days,
       grid=grid,
       start_time='2023-01-01',
       end_time='2023-12-31',
       source_name='GLORYS12v1'
   )
   
   # Save to file
   boundary_ds.to_netcdf('boundary_forcing.nc')

5. Create Climatology Dataset (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For nudging or reference, create a climatology file:

.. code-block:: python

   climatology_ds = boundary_tools.create_climatology_dataset(
       temp=temp_interp,
       salt=salt_interp,
       u=u_interp,
       v=v_interp,
       w=w,
       ubar=ubar,
       vbar=vbar,
       zeta=zeta_interp,
       time_days=time_days,
       grid=grid,
       source_name='GLORYS12v1'
   )
   
   climatology_ds.to_netcdf('climatology.nc')

Using the Script
----------------

Use the pre-configured script:

.. code-block:: bash

   cd scripts/
   python boundary_conditions.py

Edit location-specific parameters:

.. code-block:: python

   # LOCATION-SPECIFIC PARAMETERS
   
   # Specify which boundaries to include
   boundaries = {
       'west': False,
       'east': True,
       'north': True,
       'south': True
   }
   
   # Time range
   start_time = '2023-01-01'
   end_time = '2023-12-31'
   
   # Output options
   save_climatology = True   # Save climatology file
   save_boundary = True      # Save boundary forcing file
   
   # File paths
   grid_file = '../output/roms_grid.nc'
   glorys_pattern = '/path/to/GLORYS/glorys_*.nc'
   output_climatology = '../output/climatology_2023.nc'
   output_boundary = '../output/boundary_forcing_2023.nc'

Boundary Dataset Structure
---------------------------

The boundary forcing file has separate variables for each boundary:

.. code-block:: text

   Dimensions:
     bry_time: 365
     s_rho: 40
     eta_rho: 500
     xi_rho: 300
     eta_v: 499
     xi_u: 299
   
   Variables:
     # Eastern boundary
     temp_east(bry_time, s_rho, eta_rho)
     salt_east(bry_time, s_rho, eta_rho)
     u_east(bry_time, s_rho, eta_rho)
     v_east(bry_time, s_rho, eta_v)
     zeta_east(bry_time, eta_rho)
     ubar_east(bry_time, eta_rho)
     vbar_east(bry_time, eta_v)
     
     # Northern boundary
     temp_north(bry_time, s_rho, xi_rho)
     salt_north(bry_time, s_rho, xi_rho)
     # ... etc
     
     # Southern boundary
     temp_south(bry_time, s_rho, xi_rho)
     salt_south(bry_time, s_rho, xi_rho)
     # ... etc
     
     bry_time(bry_time)  # days since reference

Inactive boundaries have NaN-filled arrays.

Climatology vs Boundary Forcing
--------------------------------

**Climatology File:**

- Contains full 3D fields for entire domain
- Used for nudging interior points
- Smooths out high-frequency variability
- Typically monthly or seasonal averages

**Boundary Forcing File:**

- Contains only boundary transects
- Used for open boundary conditions
- Retains temporal variability
- Can be daily or sub-daily

When to use each:

- **Strong boundary forcing + weak nudging**: Use boundary forcing for boundaries, climatology for light interior nudging
- **Data assimilation**: Use climatology for strong interior nudging
- **Pure boundary forcing**: Use only boundary forcing file

Time Interpolation
------------------

ROMS linearly interpolates between time records. Provide adequate temporal resolution:

- **Slowly-varying**: Monthly boundary conditions may suffice
- **Energetic regions**: Daily or sub-daily recommended
- **Tides/inertial**: Hourly resolution needed

Computational Cost
------------------

Tips for handling large datasets:

1. **Process in chunks**: Interpolate smaller time ranges separately
2. **Parallel processing**: Use Dask for parallel interpolation
3. **Reduce resolution**: Start with coarser temporal resolution
4. **Subset variables**: Only process needed boundaries

.. code-block:: python

   # Example: Process in monthly chunks
   import pandas as pd
   
   months = pd.date_range('2023-01-01', '2023-12-31', freq='MS')
   
   for start in months:
       end = start + pd.DateOffset(months=1)
       chunk = glorys_data.sel(time=slice(start, end))
       # Process chunk...
       # Append to output file

API Reference
-------------

See :doc:`../api/boundary` for complete API documentation.
