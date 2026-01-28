NetCDF Handler API
==================

.. automodule:: netcdf_handler
   :members:
   :undoc-members:
   :show-inheritance:

NetCDFHandler Class
-------------------

.. autoclass:: netcdf_handler.NetCDFHandler
   :members:
   :undoc-members:
   :show-inheritance:

Methods
-------

create_netcdf_file
~~~~~~~~~~~~~~~~~~

.. automethod:: netcdf_handler.NetCDFHandler.create_netcdf_file

Overview
--------

The NetCDFHandler module provides utilities for creating and managing NetCDF files, particularly for converting and copying NetCDF datasets.

Example Usage
-------------

Basic NetCDF file creation::

    from netcdf_handler import NetCDFHandler
    
    # Create handler with custom output filename
    handler = NetCDFHandler(default_filename='processed_data.nc')
    
    # Create NetCDF from downloaded data
    handler.create_netcdf_file(
        data_path='downloaded_data.nc',
        filename='output_data.nc'
    )

Features
--------

The NetCDFHandler class provides:

- NetCDF file creation and copying
- Dimension and variable preservation
- Attribute copying
- Format conversion support
- Error handling for file operations
