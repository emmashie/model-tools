## download.py

import os
import requests
import xarray as xr
from typing import Tuple, Optional, Dict, List
import numpy as np

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


class CopernicusMarineDownloader:
    """
    Helper class for accessing Copernicus Marine data using the Marine Toolbox API.
    Provides reusable functions for downloading GLORYS and other marine datasets.
    """
    
    def __init__(self):
        """Initialize the Copernicus Marine downloader."""
        self._check_api_availability()
    
    @staticmethod
    def _check_api_availability():
        """Check if copernicusmarine package is available."""
        try:
            import copernicusmarine
        except ImportError:
            raise ImportError(
                "copernicusmarine package is required but not installed. "
                "Install it with: pip install copernicusmarine"
            )
    
    @staticmethod
    def open_dataset(
        dataset_id: str,
        variables: Optional[List[str]] = None,
        minimum_longitude: Optional[float] = None,
        maximum_longitude: Optional[float] = None,
        minimum_latitude: Optional[float] = None,
        maximum_latitude: Optional[float] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        minimum_depth: Optional[float] = None,
        maximum_depth: Optional[float] = None,
    ) -> xr.Dataset:
        """
        Open a remote Copernicus Marine dataset using the API.
        
        Args:
            dataset_id: Product dataset ID (e.g., 'cmems_mod_glo_phy_my_0.083deg_P1D-m')
            variables: List of variable names to retrieve
            minimum_longitude: Minimum longitude (-180 to 180)
            maximum_longitude: Maximum longitude (-180 to 180)
            minimum_latitude: Minimum latitude (-90 to 90)
            maximum_latitude: Maximum latitude (-90 to 90)
            start_datetime: Start date as string (e.g., '2024-01-01')
            end_datetime: End date as string (e.g., '2024-01-31')
            minimum_depth: Minimum depth (meters, positive down)
            maximum_depth: Maximum depth (meters, positive down)
            
        Returns:
            xarray.Dataset with requested data
            
        Example:
            >>> downloader = CopernicusMarineDownloader()
            >>> ds = downloader.open_dataset(
            ...     dataset_id='cmems_mod_glo_phy_my_0.083deg_P1D-m',
            ...     variables=['thetao', 'so', 'uo', 'vo', 'zos'],
            ...     minimum_longitude=-80.0,
            ...     maximum_longitude=-60.0,
            ...     minimum_latitude=30.0,
            ...     maximum_latitude=50.0,
            ...     start_datetime='2024-01-01',
            ...     end_datetime='2024-01-31'
            ... )
        """
        import copernicusmarine
        
        ds = copernicusmarine.open_dataset(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=minimum_longitude,
            maximum_longitude=maximum_longitude,
            minimum_latitude=minimum_latitude,
            maximum_latitude=maximum_latitude,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            minimum_depth=minimum_depth,
            maximum_depth=maximum_depth,
        )
        
        return ds
    
    @staticmethod
    def download_dataset(
        dataset_id: str,
        output_path: str,
        variables: Optional[List[str]] = None,
        minimum_longitude: Optional[float] = None,
        maximum_longitude: Optional[float] = None,
        minimum_latitude: Optional[float] = None,
        maximum_latitude: Optional[float] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        minimum_depth: Optional[float] = None,
        maximum_depth: Optional[float] = None,
        force_download: bool = False
    ) -> str:
        """
        Download Copernicus Marine dataset to a local NetCDF file.
        
        Args:
            dataset_id: Product dataset ID
            output_path: Path to save the downloaded NetCDF file
            variables: List of variable names to retrieve
            minimum_longitude: Minimum longitude (-180 to 180)
            maximum_longitude: Maximum longitude (-180 to 180)
            minimum_latitude: Minimum latitude (-90 to 90)
            maximum_latitude: Maximum latitude (-90 to 90)
            start_datetime: Start date as string (e.g., '2024-01-01')
            end_datetime: End date as string (e.g., '2024-01-31')
            minimum_depth: Minimum depth (meters, positive down)
            maximum_depth: Maximum depth (meters, positive down)
            force_download: If True, download even if file exists
            
        Returns:
            Path to the downloaded file
            
        Example:
            >>> downloader = CopernicusMarineDownloader()
            >>> file_path = downloader.download_dataset(
            ...     dataset_id='cmems_mod_glo_phy_my_0.083deg_P1D-m',
            ...     output_path='glorys_data.nc',
            ...     variables=['thetao', 'so', 'uo', 'vo', 'zos'],
            ...     minimum_longitude=-80.0,
            ...     maximum_longitude=-60.0,
            ...     minimum_latitude=30.0,
            ...     maximum_latitude=50.0,
            ...     start_datetime='2024-01-01',
            ...     end_datetime='2024-01-31'
            ... )
        """
        import copernicusmarine
        
        if os.path.exists(output_path) and not force_download:
            print(f"File {output_path} already exists. Use force_download=True to re-download.")
            return output_path
        
        print(f"Downloading data from Copernicus Marine Service to {output_path}...")
        
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=minimum_longitude,
            maximum_longitude=maximum_longitude,
            minimum_latitude=minimum_latitude,
            maximum_latitude=maximum_latitude,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            minimum_depth=minimum_depth,
            maximum_depth=maximum_depth,
            output_filename=output_path,
            force_download=force_download
        )
        
        print(f"Download complete: {output_path}")
        return output_path
    
    @staticmethod
    def get_glorys_dataset(
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        time_range: Tuple[str, str],
        variables: Optional[List[str]] = None,
        depth_range: Optional[Tuple[float, float]] = None,
        use_daily: bool = True
    ) -> xr.Dataset:
        """
        Convenience function to access GLORYS reanalysis data remotely.
        
        Args:
            lon_range: (min_lon, max_lon) in degrees East (-180 to 180)
            lat_range: (min_lat, max_lat) in degrees North
            time_range: (start_date, end_date) as strings (e.g., '2024-01-01')
            variables: List of variables. Defaults to ['thetao', 'so', 'uo', 'vo', 'zos']
            depth_range: Optional (min_depth, max_depth) in meters. None = all depths
            use_daily: If True, use daily data. If False, use monthly means
            
        Returns:
            xarray.Dataset with GLORYS data
            
        Example:
            >>> downloader = CopernicusMarineDownloader()
            >>> ds = downloader.get_glorys_dataset(
            ...     lon_range=(-80.0, -60.0),
            ...     lat_range=(30.0, 50.0),
            ...     time_range=('2024-01-01', '2024-01-31')
            ... )
        """
        if variables is None:
            variables = ['thetao', 'so', 'uo', 'vo', 'zos']
        
        # GLORYS dataset IDs
        if use_daily:
            dataset_id = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
        else:
            dataset_id = 'cmems_mod_glo_phy_my_0.083deg_P1M-m'
        
        kwargs = {
            'dataset_id': dataset_id,
            'variables': variables,
            'minimum_longitude': lon_range[0],
            'maximum_longitude': lon_range[1],
            'minimum_latitude': lat_range[0],
            'maximum_latitude': lat_range[1],
            'start_datetime': time_range[0],
            'end_datetime': time_range[1],
        }
        
        if depth_range is not None:
            kwargs['minimum_depth'] = depth_range[0]
            kwargs['maximum_depth'] = depth_range[1]
        
        return CopernicusMarineDownloader.open_dataset(**kwargs)
    
    @staticmethod
    def download_glorys_dataset(
        output_path: str,
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        time_range: Tuple[str, str],
        variables: Optional[List[str]] = None,
        depth_range: Optional[Tuple[float, float]] = None,
        use_daily: bool = True,
        force_download: bool = False
    ) -> str:
        """
        Convenience function to download GLORYS reanalysis data to a file.
        
        Args:
            output_path: Path to save the downloaded NetCDF file
            lon_range: (min_lon, max_lon) in degrees East (-180 to 180)
            lat_range: (min_lat, max_lat) in degrees North
            time_range: (start_date, end_date) as strings (e.g., '2024-01-01')
            variables: List of variables. Defaults to ['thetao', 'so', 'uo', 'vo', 'zos']
            depth_range: Optional (min_depth, max_depth) in meters. None = all depths
            use_daily: If True, use daily data. If False, use monthly means
            force_download: If True, download even if file exists
            
        Returns:
            Path to the downloaded file
            
        Example:
            >>> downloader = CopernicusMarineDownloader()
            >>> file_path = downloader.download_glorys_dataset(
            ...     output_path='glorys_january.nc',
            ...     lon_range=(-80.0, -60.0),
            ...     lat_range=(30.0, 50.0),
            ...     time_range=('2024-01-01', '2024-01-31')
            ... )
        """
        if variables is None:
            variables = ['thetao', 'so', 'uo', 'vo', 'zos']
        
        # GLORYS dataset IDs
        if use_daily:
            dataset_id = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
        else:
            dataset_id = 'cmems_mod_glo_phy_my_0.083deg_P1M-m'
        
        kwargs = {
            'dataset_id': dataset_id,
            'output_path': output_path,
            'variables': variables,
            'minimum_longitude': lon_range[0],
            'maximum_longitude': lon_range[1],
            'minimum_latitude': lat_range[0],
            'maximum_latitude': lat_range[1],
            'start_datetime': time_range[0],
            'end_datetime': time_range[1],
            'force_download': force_download
        }
        
        if depth_range is not None:
            kwargs['minimum_depth'] = depth_range[0]
            kwargs['maximum_depth'] = depth_range[1]
        
        return CopernicusMarineDownloader.download_dataset(**kwargs)


class ERA5Downloader:
    """
    Helper class for accessing ERA5 reanalysis data using the Copernicus Climate Data Store (CDS) API.
    Provides reusable functions for downloading ERA5 atmospheric forcing datasets.
    """
    
    def __init__(self):
        """Initialize the ERA5 downloader."""
        self._check_api_availability()
    
    @staticmethod
    def _check_api_availability():
        """Check if cdsapi package is available."""
        try:
            import cdsapi
        except ImportError:
            raise ImportError(
                "cdsapi package is required but not installed. "
                "Install it with: pip install cdsapi"
            )
    
    @staticmethod
    def download_era5_data(
        output_path: str,
        variables: List[str],
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        time_range: Tuple[str, str],
        hours: Optional[List[str]] = None,
        data_format: str = 'netcdf',
        force_download: bool = False
    ) -> str:
        """
        Download ERA5 single-level reanalysis data from CDS.
        
        Args:
            output_path: Path to save the downloaded NetCDF file
            variables: List of ERA5 variable names (e.g., ['10m_u_component_of_wind', '2m_temperature'])
            lon_range: (west, east) longitude range (-180 to 180)
            lat_range: (south, north) latitude range (-90 to 90)
            time_range: (start_date, end_date) as strings (e.g., '2024-01-01', '2024-01-31')
            hours: List of hours as strings (e.g., ['00:00', '06:00', '12:00', '18:00']). None = all hours
            data_format: Output format ('netcdf' or 'grib')
            force_download: If True, download even if file exists
            
        Returns:
            Path to the downloaded file
            
        Example:
            >>> downloader = ERA5Downloader()
            >>> file_path = downloader.download_era5_data(
            ...     output_path='era5_forcing.nc',
            ...     variables=['10m_u_component_of_wind', '10m_v_component_of_wind', 
            ...                '2m_temperature', '2m_dewpoint_temperature',
            ...                'mean_sea_level_pressure', 'total_precipitation'],
            ...     lon_range=(-80.0, -60.0),
            ...     lat_range=(30.0, 50.0),
            ...     time_range=('2024-01-01', '2024-01-31'),
            ...     hours=['00:00', '06:00', '12:00', '18:00']
            ... )
        """
        import cdsapi
        import zipfile
        
        # Check if file exists and is already a valid NetCDF (not a zip)
        if os.path.exists(output_path) and not force_download:
            # If it's a zip file, we need to extract it
            if not zipfile.is_zipfile(output_path):
                print(f"File {output_path} already exists. Use force_download=True to re-download.")
                return output_path
            else:
                print(f"Found cached zip file: {output_path}, will extract it...")
        
        # Download if file doesn't exist or force_download is True
        if (not os.path.exists(output_path) or force_download) and not zipfile.is_zipfile(output_path):
            print(f"Downloading ERA5 data from Copernicus Climate Data Store to {output_path}...")
            print(f"  Variables: {variables}")
            print(f"  Time range: {time_range[0]} to {time_range[1]}")
            print(f"  Lon range: {lon_range}")
            print(f"  Lat range: {lat_range}")
            
            # Parse date range (ensure Python strings for pandas compatibility)
            start_date = str(time_range[0])
            end_date = str(time_range[1])
            
            # Generate year, month, day lists
            import pandas as pd
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            years = sorted(list(set(date_range.strftime('%Y').tolist())))
            months = sorted(list(set(date_range.strftime('%m').tolist())))
            days = sorted(list(set(date_range.strftime('%d').tolist())))
            
            # Default to all hours if not specified
            if hours is None:
                hours = [f'{h:02d}:00' for h in range(24)]
            
            # Build request
            request = {
                'product_type': ['reanalysis'],
                'variable': variables,
                'year': years,
                'month': months,
                'day': days,
                'time': hours,
                'area': [lat_range[1], lon_range[0], lat_range[0], lon_range[1]],  # North, West, South, East
                'data_format': data_format,
            }
            
            # Download
            try:
                client = cdsapi.Client()
            except Exception as e:
                if "Missing/incomplete configuration file" in str(e):
                    raise RuntimeError(
                        "CDS API credentials not configured.\n"
                        "\n"
                        "To set up authentication:\n"
                        "1. Register for a free account at https://cds.climate.copernicus.eu/\n"
                        "2. Get your API key from https://cds.climate.copernicus.eu/user\n"
                        "3. Create a file ~/.cdsapirc with the following content:\n"
                        "\n"
                        "   url: https://cds.climate.copernicus.eu/api/v2\n"
                        "   key: YOUR_UID:YOUR_API_KEY\n"
                        "\n"
                        "Replace YOUR_UID:YOUR_API_KEY with your actual credentials from step 2.\n"
                        "\n"
                        "For detailed instructions, see: https://cds.climate.copernicus.eu/how-to-api\n"
                    ) from e
                else:
                    raise
            
            client.retrieve(
                'reanalysis-era5-single-levels',
                request
            ).download(output_path)
            
            # Verify the file was created
            if not os.path.exists(output_path):
                raise RuntimeError(f"Download appeared to complete but file not found at: {output_path}")
            
            file_size = os.path.getsize(output_path)
            print(f"Download complete: {output_path} ({file_size / (1024**2):.2f} MB)")
        
        # If the download is a zip file, extract it
        import zipfile
        if zipfile.is_zipfile(output_path):
            print(f"Downloaded file is a zip archive, extracting...")
            extract_dir = os.path.dirname(output_path) or '.'
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                # Get list of files in the archive
                nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                if not nc_files:
                    raise RuntimeError(f"No NetCDF files found in downloaded archive: {output_path}")
                
                # Extract all NetCDF files
                for nc_file in nc_files:
                    zip_ref.extract(nc_file, extract_dir)
                    print(f"  Extracted: {nc_file}")
                
                # If there's only one NetCDF file, return its path
                # Otherwise, we'll need to merge them
                if len(nc_files) == 1:
                    extracted_path = os.path.join(extract_dir, nc_files[0])
                    # Remove the zip file
                    os.remove(output_path)
                    print(f"Using extracted file: {extracted_path}")
                    return extracted_path
                else:
                    # Multiple files - merge them
                    print(f"Multiple NetCDF files found, merging...")
                    import xarray as xr
                    datasets = []
                    for nc_file in nc_files:
                        nc_path = os.path.join(extract_dir, nc_file)
                        datasets.append(xr.open_dataset(nc_path))
                    
                    # Merge all datasets
                    merged = xr.merge(datasets)
                    
                    # Save to the original output path (without .zip extension if present)
                    final_path = output_path.replace('.zip', '') if output_path.endswith('.zip') else output_path
                    if final_path == output_path:
                        final_path = output_path.replace('.nc', '_merged.nc')
                    
                    merged.to_netcdf(final_path)
                    
                    # Clean up extracted files and zip
                    for nc_file in nc_files:
                        os.remove(os.path.join(extract_dir, nc_file))
                    os.remove(output_path)
                    
                    print(f"Merged data saved to: {final_path}")
                    return final_path
        
        return output_path
    
    @staticmethod
    def download_era5_surface_forcing(
        output_path: str,
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        time_range: Tuple[str, str],
        hours: Optional[List[str]] = None,
        include_radiation: bool = True,
        force_download: bool = False
    ) -> str:
        """
        Convenience function to download standard ERA5 variables for ROMS surface forcing.
        
        Downloads variables commonly needed for ocean model surface forcing:
        - Wind components (10m u, v)
        - Temperature (2m)
        - Humidity (2m dewpoint temperature)
        - Pressure (mean sea level)
        - Precipitation (total)
        - Radiation (shortwave and longwave, if include_radiation=True)
        
        Args:
            output_path: Path to save the downloaded NetCDF file
            lon_range: (west, east) longitude range
            lat_range: (south, north) latitude range
            time_range: (start_date, end_date) as strings
            hours: List of hours. None = all hours
            include_radiation: Whether to include radiation variables
            force_download: If True, download even if file exists
            
        Returns:
            Path to the downloaded file
            
        Example:
            >>> downloader = ERA5Downloader()
            >>> file_path = downloader.download_era5_surface_forcing(
            ...     output_path='era5_surface_forcing_jan2024.nc',
            ...     lon_range=(-80.0, -60.0),
            ...     lat_range=(30.0, 50.0),
            ...     time_range=('2024-01-01', '2024-01-31'),
            ...     hours=['00:00', '06:00', '12:00', '18:00']
            ... )
        """
        # Standard surface forcing variables
        variables = [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            '2m_temperature',
            '2m_dewpoint_temperature',
            'mean_sea_level_pressure',
            'total_precipitation',
            'sea_surface_temperature',
        ]
        
        if include_radiation:
            variables.extend([
                'surface_solar_radiation_downwards',
                'surface_thermal_radiation_downwards',
                'surface_net_solar_radiation',
                'surface_net_thermal_radiation',
            ])
        
        return ERA5Downloader.download_era5_data(
            output_path=output_path,
            variables=variables,
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=time_range,
            hours=hours,
            force_download=force_download
        )
    
    @staticmethod
    def get_era5_variable_mapping():
        """
        Get standard mapping between ROMS forcing variable names and ERA5 variable names.
        
        Returns:
            Dictionary mapping ROMS names to ERA5 names
            
        Example:
            >>> mapping = ERA5Downloader.get_era5_variable_mapping()
            >>> print(mapping['u10'])  # '10m_u_component_of_wind'
        """
        return {
            'u10': 'u10',  # Short name (alternative to long name)
            'v10': 'v10',
            't2m': 't2m',
            'd2m': 'd2m',
            'msl': 'msl',
            'tp': 'tp',
            'sst': 'sst',
            'ssrd': 'ssrd',  # Surface solar radiation downwards
            'strd': 'strd',  # Surface thermal radiation downwards
            'ssr': 'ssr',    # Surface net solar radiation
            'str': 'str',    # Surface net thermal radiation
            # Long names (for reference)
            '10m_u_component_of_wind': 'u10',
            '10m_v_component_of_wind': 'v10',
            '2m_temperature': 't2m',
            '2m_dewpoint_temperature': 'd2m',
            'mean_sea_level_pressure': 'msl',
            'total_precipitation': 'tp',
            'sea_surface_temperature': 'sst',
            'surface_solar_radiation_downwards': 'ssrd',
            'surface_thermal_radiation_downwards': 'strd',
            'surface_net_solar_radiation': 'ssr',
            'surface_net_thermal_radiation': 'str',
        }
