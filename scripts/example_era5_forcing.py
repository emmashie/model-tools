"""
Example: Using different data sources for ERA5 surface forcing

This script demonstrates how to use both NetCDF files and the 
Copernicus Climate Data Store API for ERA5 atmospheric forcing data.
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(os.path.dirname(script_dir), 'code')
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from forcing import forcing_tools
from download import ERA5Downloader
import numpy as np
import xarray as xr


def example1_use_existing_netcdf():
    """
    Example 1: Load data from existing NetCDF file(s)
    
    This is the traditional method - fastest if you already have the data.
    """
    print("=" * 70)
    print("Example 1: Using existing NetCDF file(s)")
    print("=" * 70)
    
    time_range = ('2024-01-01', '2024-01-02')
    
    # Single file with all variables
    netcdf_path = '/path/to/your/era5_data.nc'
    
    # Load data from file
    ds = forcing_tools.load_era5_data(
        time_range=time_range,
        lon_range=None,  # Not needed for file loading
        lat_range=None,  # Not needed for file loading
        use_api=False,
        netcdf_paths=netcdf_path
    )
    
    print(f"\nLoaded dataset:")
    print(ds)
    print(f"\nAvailable variables: {list(ds.data_vars)}")
    
    # Or multiple files
    netcdf_paths = {
        'main': '/path/to/ERA5_data.nc',
        'pair': '/path/to/ERA5_Pair.nc',
        'rad': '/path/to/ERA5_rad.nc'
    }
    
    ds_dict = forcing_tools.load_era5_data(
        time_range=time_range,
        lon_range=None,
        lat_range=None,
        use_api=False,
        netcdf_paths=netcdf_paths
    )
    
    print(f"\nLoaded {len(ds_dict)} datasets:")
    for key in ds_dict.keys():
        print(f"  {key}: {list(ds_dict[key].data_vars)}")
    
    return ds


def example2_use_api_directly():
    """
    Example 2: Access data via Copernicus Climate Data Store API
    
    No local file needed - data is accessed remotely.
    Requires: pip install cdsapi
    """
    print("\n" + "=" * 70)
    print("Example 2: Using CDS API (remote access)")
    print("=" * 70)
    
    time_range = ('2024-01-01', '2024-01-02')
    lon_range = (-80.0, -60.0)  # East Coast US
    lat_range = (30.0, 50.0)
    
    # Load data via API - downloads to temporary file
    ds = forcing_tools.load_era5_data(
        time_range=time_range,
        lon_range=lon_range,
        lat_range=lat_range,
        use_api=True,
        hours=['00:00', '06:00', '12:00', '18:00'],  # 6-hourly data
        include_radiation=True
    )
    
    print(f"\nLoaded dataset:")
    print(ds)
    print(f"\nAvailable variables: {list(ds.data_vars)}")
    print(f"\nTime steps: {len(ds.time)}")
    
    return ds


def example3_download_and_cache():
    """
    Example 3: Download once, reuse many times
    
    Best practice: Download data via API and save to a local file,
    then use that file for multiple runs.
    """
    print("\n" + "=" * 70)
    print("Example 3: Download and cache data")
    print("=" * 70)
    
    time_range = ('2024-01-01', '2024-01-07')
    lon_range = (-80.0, -60.0)
    lat_range = (30.0, 50.0)
    cache_path = './cached_era5_jan2024.nc'
    
    # Download and cache (only downloads if file doesn't exist)
    print("\nStep 1: Download and cache data")
    file_path = forcing_tools.download_and_cache_era5(
        time_range=time_range,
        lon_range=lon_range,
        lat_range=lat_range,
        output_path=cache_path,
        hours=['00:00', '06:00', '12:00', '18:00'],
        include_radiation=True,
        force_download=False  # Skip if file exists
    )
    
    print(f"\nData cached to: {file_path}")
    
    # Now use the cached file (fast!)
    print("\nStep 2: Load from cached file")
    ds = forcing_tools.load_era5_data(
        time_range=time_range,
        lon_range=None,
        lat_range=None,
        use_api=False,
        netcdf_paths=file_path
    )
    
    print(f"\nLoaded dataset from cache:")
    print(ds)
    
    return ds


def example4_advanced_api_usage():
    """
    Example 4: Advanced API usage with custom parameters
    
    Use the ERA5Downloader class directly for more control.
    """
    print("\n" + "=" * 70)
    print("Example 4: Advanced API usage")
    print("=" * 70)
    
    downloader = ERA5Downloader()
    
    # Example 4a: Download specific variables only
    print("\n4a: Custom variable selection")
    downloader.download_era5_data(
        output_path='era5_winds_only.nc',
        variables=[
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'mean_sea_level_pressure'
        ],
        lon_range=(-80.0, -60.0),
        lat_range=(30.0, 50.0),
        time_range=('2024-01-01', '2024-01-02'),
        hours=['00:00', '12:00']
    )
    print("Downloaded winds and pressure only")
    
    # Example 4b: High temporal resolution for storm
    print("\n4b: High-frequency data for storm event")
    downloader.download_era5_surface_forcing(
        output_path='era5_storm_hourly.nc',
        lon_range=(-75.0, -70.0),
        lat_range=(38.0, 42.0),
        time_range=('2024-08-15', '2024-08-18'),
        hours=None,  # All hours (24 per day)
        include_radiation=True
    )
    print("Downloaded hourly data for storm period")
    
    # Example 4c: Long-term with reduced frequency
    print("\n4c: Long-term forcing with reduced temporal resolution")
    downloader.download_era5_surface_forcing(
        output_path='era5_annual_6hourly.nc',
        lon_range=(-80.0, -60.0),
        lat_range=(30.0, 50.0),
        time_range=('2024-01-01', '2024-12-31'),
        hours=['00:00', '06:00', '12:00', '18:00'],  # 6-hourly
        include_radiation=True
    )
    print("Downloaded annual forcing (6-hourly)")
    
    return None


def example5_variable_mapping():
    """
    Example 5: Understanding ERA5 variable names
    
    ERA5 uses specific variable naming conventions.
    """
    print("\n" + "=" * 70)
    print("Example 5: ERA5 variable mapping")
    print("=" * 70)
    
    # Get standard variable mapping
    mapping = ERA5Downloader.get_era5_variable_mapping()
    
    print("\nStandard ERA5 variable names:")
    print("\nShort names (used in NetCDF files):")
    for short_name in ['u10', 'v10', 't2m', 'd2m', 'msl', 'tp', 'sst', 'ssrd', 'strd']:
        print(f"  {short_name}: {short_name}")
    
    print("\nLong names (used in API requests):")
    long_names = [
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2m_temperature',
        '2m_dewpoint_temperature',
        'mean_sea_level_pressure',
        'total_precipitation',
        'sea_surface_temperature',
        'surface_solar_radiation_downwards',
        'surface_thermal_radiation_downwards'
    ]
    for long_name in long_names:
        short = mapping.get(long_name, 'N/A')
        print(f"  {long_name} → {short}")
    
    return mapping


def example6_monthly_chunks():
    """
    Example 6: Download data in monthly chunks
    
    Useful for very long time series or large domains.
    """
    print("\n" + "=" * 70)
    print("Example 6: Download in monthly chunks")
    print("=" * 70)
    
    downloader = ERA5Downloader()
    
    lon_range = (-80.0, -60.0)
    lat_range = (30.0, 50.0)
    
    # Download data for first quarter of 2024
    months = [
        ('2024-01-01', '2024-01-31', 'era5_jan2024.nc'),
        ('2024-02-01', '2024-02-29', 'era5_feb2024.nc'),
        ('2024-03-01', '2024-03-31', 'era5_mar2024.nc'),
    ]
    
    for start, end, filename in months:
        print(f"\nDownloading {start} to {end}")
        
        downloader.download_era5_surface_forcing(
            output_path=filename,
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=(start, end),
            hours=['00:00', '06:00', '12:00', '18:00'],
            include_radiation=True,
            force_download=False
        )
        
        print(f"  Saved to {filename}")
    
    print("\nAll months downloaded! You can now combine them if needed:")
    print("  ds = xr.open_mfdataset('era5_*2024.nc', combine='by_coords')")
    
    return None


def main():
    """
    Run examples based on user choice
    """
    print("\n" + "=" * 70)
    print("ERA5 Surface Forcing Examples for model-tools")
    print("=" * 70)
    print("\nAvailable examples:")
    print("  1: Use existing NetCDF file(s)")
    print("  2: Use CDS API (remote)")
    print("  3: Download and cache data")
    print("  4: Advanced API usage")
    print("  5: Variable mapping reference")
    print("  6: Download in monthly chunks")
    print("\nNote: Examples 2-6 require 'pip install cdsapi'")
    print("      and CDS API setup (see README_SURFACE_FORCING.md)")
    
    # Uncomment the example you want to run:
    
    # example1_use_existing_netcdf()
    # example2_use_api_directly()
    # example3_download_and_cache()
    # example4_advanced_api_usage()
    # example5_variable_mapping()
    # example6_monthly_chunks()
    
    print("\n" + "=" * 70)
    print("Uncomment an example in the main() function to run it")
    print("=" * 70)


if __name__ == '__main__':
    main()
