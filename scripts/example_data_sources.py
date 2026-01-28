"""
Example: Using different data sources for initialization

This script demonstrates how to use both NetCDF files and the 
Copernicus Marine API for ocean model initialization.
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(os.path.dirname(script_dir), 'code')
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from initialization import init_tools
from download import CopernicusMarineDownloader
import numpy as np
import xarray as xr


def example1_use_existing_netcdf():
    """
    Example 1: Load data from an existing NetCDF file
    
    This is the traditional method - fastest if you already have the data.
    """
    print("=" * 70)
    print("Example 1: Using existing NetCDF file")
    print("=" * 70)
    
    init_time = '2024-01-01'
    netcdf_path = '/path/to/your/glorys_data.nc'
    
    # Load data from file
    ds = init_tools.load_glorys_data(
        init_time=init_time,
        lon_range=None,  # Not needed for file loading
        lat_range=None,  # Not needed for file loading
        use_api=False,
        netcdf_path=netcdf_path
    )
    
    print(f"\nLoaded dataset:")
    print(ds)
    print(f"\nAvailable variables: {list(ds.data_vars)}")
    
    return ds


def example2_use_api_directly():
    """
    Example 2: Access data via Copernicus Marine API
    
    No local file needed - data is accessed remotely.
    Requires: pip install copernicusmarine
    """
    print("\n" + "=" * 70)
    print("Example 2: Using Copernicus Marine API (remote access)")
    print("=" * 70)
    
    init_time = '2024-01-01'
    lon_range = (-80.0, -60.0)  # East Coast US
    lat_range = (30.0, 50.0)
    
    # Load data via API
    ds = init_tools.load_glorys_data(
        init_time=init_time,
        lon_range=lon_range,
        lat_range=lat_range,
        use_api=True,
        time_buffer_days=1  # Download ±1 day around init_time
    )
    
    print(f"\nLoaded dataset:")
    print(ds)
    print(f"\nAvailable variables: {list(ds.data_vars)}")
    
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
    
    init_time = '2024-01-01'
    lon_range = (-80.0, -60.0)
    lat_range = (30.0, 50.0)
    cache_path = './cached_glorys_jan2024.nc'
    
    # Download and cache (only downloads if file doesn't exist)
    print("\nStep 1: Download and cache data")
    file_path = init_tools.download_and_cache_glorys(
        init_time=init_time,
        lon_range=lon_range,
        lat_range=lat_range,
        output_path=cache_path,
        time_buffer_days=3,  # Get a few days worth
        force_download=False  # Skip if file exists
    )
    
    print(f"\nData cached to: {file_path}")
    
    # Now use the cached file (fast!)
    print("\nStep 2: Load from cached file")
    ds = init_tools.load_glorys_data(
        init_time=init_time,
        lon_range=None,
        lat_range=None,
        use_api=False,
        netcdf_path=file_path
    )
    
    print(f"\nLoaded dataset from cache:")
    print(ds)
    
    return ds


def example4_advanced_api_usage():
    """
    Example 4: Advanced API usage with custom parameters
    
    Use the CopernicusMarineDownloader class directly for more control.
    """
    print("\n" + "=" * 70)
    print("Example 4: Advanced API usage")
    print("=" * 70)
    
    downloader = CopernicusMarineDownloader()
    
    # Example 4a: Get GLORYS data with custom variables and depth range
    print("\n4a: Custom variables and depth range")
    ds = downloader.open_dataset(
        dataset_id='cmems_mod_glo_phy_my_0.083deg_P1D-m',
        variables=['thetao', 'so', 'mlotst'],  # Include mixed layer depth
        minimum_longitude=-80.0,
        maximum_longitude=-60.0,
        minimum_latitude=30.0,
        maximum_latitude=50.0,
        start_datetime='2024-01-01',
        end_datetime='2024-01-07',
        minimum_depth=0.0,
        maximum_depth=1000.0  # Only upper 1000m
    )
    
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Depth range: {ds.depth.min().values} to {ds.depth.max().values} m")
    
    # Example 4b: Download monthly means (smaller files)
    print("\n4b: Using monthly mean data")
    ds_monthly = downloader.open_dataset(
        dataset_id='cmems_mod_glo_phy_my_0.083deg_P1M-m',  # Monthly means
        variables=['thetao', 'so'],
        minimum_longitude=-80.0,
        maximum_longitude=-60.0,
        minimum_latitude=30.0,
        maximum_latitude=50.0,
        start_datetime='2024-01-01',
        end_datetime='2024-12-31'
    )
    
    print(f"Time steps: {len(ds_monthly.time)}")
    print(f"Times: {ds_monthly.time.values}")
    
    return ds, ds_monthly


def example5_multiple_time_periods():
    """
    Example 5: Download data for multiple time periods
    
    Useful for boundary conditions or long-term forcing.
    """
    print("\n" + "=" * 70)
    print("Example 5: Multiple time periods for boundary conditions")
    print("=" * 70)
    
    downloader = CopernicusMarineDownloader()
    
    # Download data for each month
    lon_range = (-80.0, -60.0)
    lat_range = (30.0, 50.0)
    
    datasets = []
    
    for month in range(1, 4):  # Just first 3 months for demo
        start = f'2024-{month:02d}-01'
        if month == 12:
            end = '2024-12-31'
        else:
            # Get first day of next month
            next_month = month + 1
            end = f'2024-{next_month:02d}-01'
        
        print(f"\nDownloading month {month}: {start} to {end}")
        
        ds = downloader.get_glorys_dataset(
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=(start, end),
            use_daily=True
        )
        
        datasets.append(ds)
        print(f"  Got {len(ds.time)} time steps")
    
    # Combine all months
    combined = xr.concat(datasets, dim='time')
    print(f"\nCombined dataset has {len(combined.time)} time steps")
    print(f"Date range: {combined.time.min().values} to {combined.time.max().values}")
    
    return combined


def main():
    """
    Run examples based on user choice
    """
    print("\n" + "=" * 70)
    print("Data Source Examples for model-tools")
    print("=" * 70)
    print("\nAvailable examples:")
    print("  1: Use existing NetCDF file")
    print("  2: Use Copernicus Marine API (remote)")
    print("  3: Download and cache data")
    print("  4: Advanced API usage")
    print("  5: Multiple time periods")
    print("\nNote: Examples 2-5 require 'pip install copernicusmarine'")
    print("      and authentication with 'copernicusmarine login'")
    
    # Uncomment the example you want to run:
    
    # example1_use_existing_netcdf()
    # example2_use_api_directly()
    # example3_download_and_cache()
    # example4_advanced_api_usage()
    # example5_multiple_time_periods()
    
    print("\n" + "=" * 70)
    print("Uncomment an example in the main() function to run it")
    print("=" * 70)


if __name__ == '__main__':
    main()
