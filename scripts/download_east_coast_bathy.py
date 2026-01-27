## download_east_coast_bathy.py
import os
import sys
import xarray as xr

# Add parent directory to path to import from code/
# Handle both script execution and interactive use
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.insert(0, os.path.join(script_dir, '..', 'code'))

from download import Downloader
from netcdf_handler import NetCDFHandler
import cmocean

def download_east_coast_bathymetry(output_dir="."):
    url = "https://topex.ucsd.edu/pub/global_topo_1min/topo_20.1.nc"
    full_nc = os.path.join(output_dir, "topo_1min.nc")
    subset_nc = os.path.join(output_dir, "east_coast_bathy.nc")
    lat_range = (30, 50)
    lon_range = (-80, -40)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use Downloader class methods
    downloader = Downloader()
    downloader.download_file(url, full_nc)
    downloader.subset_dataset(full_nc, subset_nc, lat_range, lon_range)

    # Optionally, use your NetCDFHandler to copy/standardize the file
    final_nc = os.path.join(output_dir, "east_coast_bathy_final.nc")
    handler = NetCDFHandler(default_filename=final_nc)
    handler.create_netcdf_file(subset_nc)

if __name__ == "__main__":
    download_east_coast_bathymetry()

    def plot_bathymetry(nc_file):
        import matplotlib.pyplot as plt

        ds = xr.open_dataset(nc_file)
        plt.figure(figsize=(12, 8))
        ds.z.plot(
            cmap=cmocean.cm.topo,
            cbar_kwargs={'label': 'Depth (m)'}
        )
        plt.title('East Coast Bathymetry')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig('east_coast_bathymetry.png', dpi=300, bbox_inches='tight')
        plt.close()
        ds.close()

    if __name__ == "__main__":
        download_east_coast_bathymetry()
        plot_bathymetry("east_coast_bathy_final.nc")