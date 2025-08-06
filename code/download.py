## download.py

import requests
from typing import Tuple

class Downloader:
    """Class responsible for downloading SRTM15+ bathymetry data."""

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

