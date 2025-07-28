## download_east_coast_bathy.py
import os
import requests
import xarray as xr
from netcdf_handler import NetCDFHandler
import cmocean

def download_full_bathy_file(url, local_nc):
    if not os.path.exists(local_nc):
        print(f"Downloading {url} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_nc, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded to {local_nc}")
    else:
        print(f"{local_nc} already exists, skipping download.")

def subset_bathy(input_nc, output_nc, lat_range, lon_range):
    print(f"Subsetting region: lat {lat_range}, lon {lon_range}")
    ds = xr.open_dataset(input_nc)
    # Adjust longitude if needed (e.g., 0-360 to -180 to 180)
    if ds.lon.max() > 180:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
    ds_sub = ds.sel(
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1])
    )
    ds_sub.to_netcdf(output_nc)
    ds.close()
    print(f"Subset saved to {output_nc}")

def download_east_coast_bathymetry(output_dir="."):
    url = "https://topex.ucsd.edu/pub/global_topo_1min/topo_20.1.nc"
    full_nc = os.path.join(output_dir, "topo_1min.nc")
    subset_nc = os.path.join(output_dir, "east_coast_bathy.nc")
    lat_range = (30, 50)
    lon_range = (-80, -40)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    download_full_bathy_file(url, full_nc)
    subset_bathy(full_nc, subset_nc, lat_range, lon_range)

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