ROMS Model Tools Documentation
===============================

**ROMS Model Tools** is a comprehensive Python toolkit for preprocessing and preparing input files for the Regional Ocean Modeling System (ROMS). This package provides modular, reusable tools for grid generation, initial conditions, surface forcing, and boundary conditions.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   workflows/grid_generation
   workflows/initialization
   workflows/surface_forcing
   workflows/boundary_conditions

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/grid
   api/initialization
   api/forcing
   api/boundary
   api/conversions
   api/interpolate
   api/download

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   contributing
   license

Overview
--------

The ROMS Model Tools package consists of:

- **Reusable Python classes** in the ``code/`` directory containing location-agnostic functions
- **Location-specific scripts** in the ``scripts/`` directory that call the reusable functions
- **Comprehensive interpolation tools** for regridding ocean and atmospheric data
- **Conversion utilities** for computing derived variables (e.g., barotropic velocities, vertical velocities)

Key Features
------------

* **Modular Design**: Separation of reusable code from location-specific parameters
* **GLORYS12v1 Support**: Interpolation from GLORYS global ocean reanalysis
* **ERA5 Integration**: Processing of ERA5 atmospheric forcing data
* **Grid Generation**: Bathymetry interpolation with iterative smoothing
* **Boundary Conditions**: Flexible boundary forcing with climatology options
* **Vertical Coordinates**: ROMS terrain-following coordinate system support

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
