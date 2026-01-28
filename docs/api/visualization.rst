Visualization Tools API
=======================

.. automodule:: visualization
   :members:
   :undoc-members:
   :show-inheritance:

Visualization Class
-------------------

.. autoclass:: visualization.Visualization
   :members:
   :undoc-members:
   :show-inheritance:

Methods
-------

plot_data
~~~~~~~~~

.. automethod:: visualization.Visualization.plot_data

Overview
--------

The Visualization module provides tools for plotting and visualizing ROMS data, particularly for comparing original and processed bathymetry data.

Example Usage
-------------

Basic plotting of bathymetry data::

    from visualization import Visualization
    import numpy as np
    
    # Create visualization object
    viz = Visualization(figsize=(12, 8))
    
    # Plot original vs smoothed bathymetry
    viz.plot_data(
        original_data=original_bathymetry,
        smoothed_data=smoothed_bathymetry
    )

The visualization tools are designed to help users:

- Compare original and processed datasets
- Validate smoothing and interpolation operations
- Generate publication-quality figures
- Inspect spatial patterns in bathymetry and other grid data
