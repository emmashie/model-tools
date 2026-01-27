## download.py

import os
import requests
import xarray as xr
from typing import Tuple

class Downloader:
    """Class responsible for downloading and processing bathymetry data."""

    def __init__(self, base_url: str = "https://example.com/srtm15plus"):
        """Initializes the Downloader with a base URL for data download.

        Args:
            base_url (str): The base URL for downloading data. Defaults to a placeholder URL.
        """
        self.base_url = base_url

    def download_data(self, lat_range: Tuple[float, float], lon_range: Tuple[float, float]) -> str:
        """Downloads data for the specified latitude and longitude range.

        Args:
            lat_range (Tuple[float, float]): Latitude range as a tuple (min_lat, max_lat).
            lon_range (Tuple[float, float]): Longitude range as a tuple (min_lon, max_lon).

        Returns:
            str: Path to the downloaded data file.
        """
        # Construct the URL with the given latitude and longitude range
        url = f"{self.base_url}?lat_min={lat_range[0]}&lat_max={lat_range[1]}&lon_min={lon_range[0]}&lon_max={lon_range[1]}"
        
        # Define the local file path to save the downloaded data
        local_filename = "srtm15plus_data.nc"
        
        try:
            # Send a GET request to download the data
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad responses

            # Write the content to a local file
            with open(local_filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Data successfully downloaded to {local_filename}")
            return local_filename

        except requests.exceptions.RequestException as e:
            print(f"An error occurred while downloading data: {e}")
            return ""

    def download_file(self, url: str, local_path: str) -> None:
        """Downloads a file from a URL to a local path.

        Args:
            url (str): The URL to download from.
            local_path (str): The local file path to save to.
        """
        if not os.path.exists(local_path):
            print(f"Downloading {url} ...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded to {local_path}")
        else:
            print(f"{local_path} already exists, skipping download.")

    def subset_dataset(self, input_nc: str, output_nc: str, 
                      lat_range: Tuple[float, float], 
                      lon_range: Tuple[float, float]) -> None:
        """Subsets a NetCDF dataset to a specified lat/lon range.

        Args:
            input_nc (str): Path to input NetCDF file.
            output_nc (str): Path to output NetCDF file.
            lat_range (Tuple[float, float]): Latitude range as (min, max).
            lon_range (Tuple[float, float]): Longitude range as (min, max).
        """
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

