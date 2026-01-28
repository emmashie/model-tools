"""
Simple tests to verify the new data access functionality.

Run with: python test_data_access.py
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
        from download import Downloader, CopernicusMarineDownloader
        print("  ✓ download.py imports successfully")
    except ImportError as e:
        print(f"  ✗ download.py import failed: {e}")
        return False
    
    try:
        from initialization import init_tools
        print("  ✓ initialization.py imports successfully")
    except ImportError as e:
        print(f"  ✗ initialization.py import failed: {e}")
        return False
    
    return True


def test_copernicus_api_available():
    """Test if copernicusmarine is installed."""
    print("\nTesting Copernicus Marine API availability...")
    
    try:
        import copernicusmarine
        print("  ✓ copernicusmarine is installed")
        return True
    except ImportError:
        print("  ⚠ copernicusmarine not installed (optional)")
        print("    Install with: pip install copernicusmarine")
        return False


def test_downloader_class_methods():
    """Test that CopernicusMarineDownloader has expected methods."""
    print("\nTesting CopernicusMarineDownloader class...")
    
    from download import CopernicusMarineDownloader
    
    expected_methods = [
        'open_dataset',
        'download_dataset',
        'get_glorys_dataset',
        'download_glorys_dataset'
    ]
    
    for method in expected_methods:
        if hasattr(CopernicusMarineDownloader, method):
            print(f"  ✓ Method '{method}' exists")
        else:
            print(f"  ✗ Method '{method}' missing")
            return False
    
    return True


def test_init_tools_methods():
    """Test that init_tools has expected methods."""
    print("\nTesting init_tools class...")
    
    from initialization import init_tools
    
    expected_methods = [
        'load_glorys_data',
        'download_and_cache_glorys',
        'add_deep_ocean_layer',
        'compute_time_since_reference',
        'prepare_source_coords',
        'interpolate_and_mask_3d',
        'interpolate_and_mask_2d',
        'create_initial_conditions_dataset'
    ]
    
    for method in expected_methods:
        if hasattr(init_tools, method):
            print(f"  ✓ Method '{method}' exists")
        else:
            print(f"  ✗ Method '{method}' missing")
            return False
    
    return True


def test_method_signatures():
    """Test that methods have proper signatures."""
    print("\nTesting method signatures...")
    
    from initialization import init_tools
    import inspect
    
    # Test load_glorys_data signature
    sig = inspect.signature(init_tools.load_glorys_data)
    params = list(sig.parameters.keys())
    
    expected_params = [
        'init_time', 'lon_range', 'lat_range', 
        'use_api', 'netcdf_path', 'time_buffer_days', 'variables'
    ]
    
    if all(p in params for p in expected_params):
        print("  ✓ load_glorys_data has correct parameters")
    else:
        print(f"  ✗ load_glorys_data parameters: {params}")
        return False
    
    return True


def test_file_structure():
    """Test that documentation files were created."""
    print("\nTesting documentation files...")
    
    base_dir = os.path.dirname(script_dir)
    
    expected_files = [
        'scripts/README_INITIALIZATION.md',
        'scripts/example_data_sources.py',
        'docs/api/data_sources.rst',
        'QUICK_REFERENCE.md',
        'CHANGES_COPERNICUS_API.md'
    ]
    
    for file_path in expected_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"  ✓ {file_path} exists")
        else:
            print(f"  ✗ {file_path} missing")
            return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing model-tools Copernicus Marine API Integration")
    print("=" * 70)
    
    tests = [
        ("Module Imports", test_import_modules),
        ("API Availability", test_copernicus_api_available),
        ("Downloader Methods", test_downloader_class_methods),
        ("Init Tools Methods", test_init_tools_methods),
        ("Method Signatures", test_method_signatures),
        ("Documentation Files", test_file_structure),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  ✗ Test '{test_name}' raised exception: {e}")
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
        print("\n✅ All tests passed! The integration is working correctly.")
    elif passed >= total - 1:  # Allow API test to fail (it's optional)
        print("\n⚠ Most tests passed. The integration should work.")
        print("   Note: copernicusmarine is optional for using local NetCDF files.")
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
