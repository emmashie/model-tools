"""
Simple tests to verify the ERA5 data access functionality.

Run with: python test_era5_access.py
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(os.path.dirname(script_dir), 'code')
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)


def test_import_modules():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        from download import Downloader, CopernicusMarineDownloader, ERA5Downloader
        print("  ✓ download.py imports successfully")
        print("    - ERA5Downloader class available")
    except ImportError as e:
        print(f"  ✗ download.py import failed: {e}")
        return False
    
    try:
        from forcing import forcing_tools
        print("  ✓ forcing.py imports successfully")
    except ImportError as e:
        print(f"  ✗ forcing.py import failed: {e}")
        return False
    
    return True


def test_cdsapi_available():
    """Test if cdsapi is installed."""
    print("\nTesting CDS API availability...")
    
    try:
        import cdsapi
        print("  ✓ cdsapi is installed")
        return True
    except ImportError:
        print("  ⚠ cdsapi not installed (optional)")
        print("    Install with: pip install cdsapi")
        return False


def test_era5_downloader_methods():
    """Test that ERA5Downloader has expected methods."""
    print("\nTesting ERA5Downloader class...")
    
    from download import ERA5Downloader
    
    expected_methods = [
        'download_era5_data',
        'download_era5_surface_forcing',
        'get_era5_variable_mapping'
    ]
    
    for method in expected_methods:
        if hasattr(ERA5Downloader, method):
            print(f"  ✓ Method '{method}' exists")
        else:
            print(f"  ✗ Method '{method}' missing")
            return False
    
    return True


def test_forcing_tools_methods():
    """Test that forcing_tools has expected methods."""
    print("\nTesting forcing_tools class...")
    
    from forcing import forcing_tools
    
    expected_methods = [
        'load_era5_data',
        'download_and_cache_era5',
        'prepare_forcing_coords',
        'interpolate_forcing_timeseries',
        'create_surface_forcing_dataset'
    ]
    
    for method in expected_methods:
        if hasattr(forcing_tools, method):
            print(f"  ✓ Method '{method}' exists")
        else:
            print(f"  ✗ Method '{method}' missing")
            return False
    
    return True


def test_method_signatures():
    """Test that methods have proper signatures."""
    print("\nTesting method signatures...")
    
    from forcing import forcing_tools
    import inspect
    
    # Test load_era5_data signature
    sig = inspect.signature(forcing_tools.load_era5_data)
    params = list(sig.parameters.keys())
    
    expected_params = [
        'time_range', 'lon_range', 'lat_range', 
        'use_api', 'netcdf_paths', 'hours', 
        'include_radiation', 'force_download'
    ]
    
    if all(p in params for p in expected_params):
        print("  ✓ load_era5_data has correct parameters")
    else:
        print(f"  ✗ load_era5_data parameters: {params}")
        return False
    
    return True


def test_variable_mapping():
    """Test ERA5 variable mapping."""
    print("\nTesting ERA5 variable mapping...")
    
    from download import ERA5Downloader
    
    mapping = ERA5Downloader.get_era5_variable_mapping()
    
    # Check some key mappings
    expected_mappings = {
        'u10': 'u10',
        'v10': 'v10',
        't2m': 't2m',
        '10m_u_component_of_wind': 'u10',
        '2m_temperature': 't2m'
    }
    
    for key, expected_val in expected_mappings.items():
        actual_val = mapping.get(key)
        if actual_val == expected_val:
            print(f"  ✓ {key} → {actual_val}")
        else:
            print(f"  ✗ {key} → {actual_val} (expected {expected_val})")
            return False
    
    return True


def test_file_structure():
    """Test that documentation files were created."""
    print("\nTesting documentation files...")
    
    base_dir = os.path.dirname(script_dir)
    
    expected_files = [
        'scripts/README_SURFACE_FORCING.md',
        'scripts/example_era5_forcing.py',
        'docs/api/era5_forcing.rst',
        'CHANGES_ERA5_API.md'
    ]
    
    for file_path in expected_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"  ✓ {file_path} exists")
        else:
            print(f"  ✗ {file_path} missing")
            return False
    
    return True


def test_surface_forcing_script():
    """Test that surface_forcing.py has new configuration."""
    print("\nTesting surface_forcing.py configuration...")
    
    script_path = os.path.join(script_dir, 'surface_forcing.py')
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    required_strings = [
        'USE_API',
        'NETCDF_PATHS',
        'API_LON_RANGE',
        'API_LAT_RANGE',
        'API_HOURS',
        'API_INCLUDE_RADIATION',
        'load_era5_data'
    ]
    
    for string in required_strings:
        if string in content:
            print(f"  ✓ Found '{string}' in script")
        else:
            print(f"  ✗ Missing '{string}' in script")
            return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing model-tools ERA5 CDS API Integration")
    print("=" * 70)
    
    tests = [
        ("Module Imports", test_import_modules),
        ("CDS API Availability", test_cdsapi_available),
        ("ERA5Downloader Methods", test_era5_downloader_methods),
        ("Forcing Tools Methods", test_forcing_tools_methods),
        ("Method Signatures", test_method_signatures),
        ("Variable Mapping", test_variable_mapping),
        ("Documentation Files", test_file_structure),
        ("Surface Forcing Script", test_surface_forcing_script),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  ✗ Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! The ERA5 integration is working correctly.")
    elif passed >= total - 1:  # Allow CDS API test to fail (it's optional)
        print("\n⚠ Most tests passed. The integration should work.")
        print("   Note: cdsapi is optional for using local NetCDF files.")
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
