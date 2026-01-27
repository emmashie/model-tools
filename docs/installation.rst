Installation
============

Requirements
------------

Python Dependencies
~~~~~~~~~~~~~~~~~~~

The following Python packages are required:

- Python >= 3.9
- xarray >= 2022.0
- numpy >= 1.20
- scipy >= 1.7
- pandas >= 1.3
- netCDF4 >= 1.5
- requests >= 2.25

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For visualization and analysis:

- matplotlib >= 3.4
- cartopy >= 0.20
- cmocean >= 2.0

Installation Steps
------------------

1. Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/your-org/model-tools.git
   cd model-tools

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

Using conda (recommended):

.. code-block:: bash

   conda create -n roms-tools python=3.11
   conda activate roms-tools
   conda install -c conda-forge xarray numpy scipy pandas netCDF4 requests

Using pip:

.. code-block:: bash

   pip install xarray numpy scipy pandas netCDF4 requests

3. Verify Installation
~~~~~~~~~~~~~~~~~~~~~~~

Test that you can import the modules:

.. code-block:: python

   import sys
   sys.path.insert(0, '/path/to/model-tools/code')
   
   from grid import grid_tools
   from initialization import init_tools
   from forcing import forcing_tools
   from boundary import boundary_tools

Directory Structure
-------------------

After installation, your directory should look like:

.. code-block:: text

   model-tools/
   ├── code/              # Reusable Python classes
   │   ├── boundary.py
   │   ├── conversions.py
   │   ├── download.py
   │   ├── forcing.py
   │   ├── grid.py
   │   ├── initialization.py
   │   └── interpolate.py
   ├── scripts/           # Location-specific scripts
   │   ├── boundary_conditions.py
   │   ├── download_east_coast_bathy.py
   │   ├── grid_generation.py
   │   ├── initialize.py
   │   └── surface_forcing.py
   ├── output/            # Generated output files
   └── docs/              # Documentation

Configuration
-------------

Path Setup
~~~~~~~~~~

The scripts automatically handle path configuration. When running from the ``scripts/`` directory, they add the ``code/`` directory to Python's path:

.. code-block:: python

   try:
       script_dir = os.path.dirname(os.path.abspath(__file__))
       code_dir = os.path.join(os.path.dirname(script_dir), 'code')
   except NameError:
       code_dir = '/global/cfs/cdirs/m4304/enuss/model-tools/code'
   
   if code_dir not in sys.path:
       sys.path.insert(0, code_dir)

For notebook environments, use the fallback path explicitly.

Next Steps
----------

See the :doc:`quickstart` guide to begin using ROMS Model Tools.
