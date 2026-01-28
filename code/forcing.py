import numpy as np
import xarray as xr
from typing import Optional, Dict, Tuple, Union, List
import os

class forcing_tools:
    """Tools for creating ROMS surface forcing files."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def load_era5_data(
        time_range: Tuple[Union[str, np.datetime64], Union[str, np.datetime64]],
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        use_api: bool = False,
        netcdf_paths: Optional[Union[str, Dict[str, str]]] = None,
        hours: Optional[List[str]] = None,
        include_radiation: bool = True,
        force_download: bool = False
    ) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
        """
        Load ERA5 data from either NetCDF file(s) or via CDS API.
        
        Args:
            time_range: (start_time, end_time) as datetime64 or strings
            lon_range: (min_lon, max_lon) in degrees East
            lat_range: (min_lat, max_lat) in degrees North
            use_api: If True, use CDS API. If False, use netcdf_paths
            netcdf_paths: Path to NetCDF file or dict mapping dataset names to paths
                         (e.g., {'main': 'era5.nc', 'rad': 'era5_rad.nc'})
            hours: List of hours for API download (e.g., ['00:00', '06:00', '12:00', '18:00'])
            include_radiation: Whether to include radiation variables (for API only)
            force_download: If True, re-download even if file exists (for API only)
            
        Returns:
            xarray.Dataset or dict of datasets
            
        Example:
            >>> # Load from NetCDF file
            >>> ds = forcing_tools.load_era5_data(
            ...     time_range=('2024-01-01', '2024-01-31'),
            ...     lon_range=(-80.0, -60.0),
            ...     lat_range=(30.0, 50.0),
            ...     use_api=False,
            ...     netcdf_paths='era5_data.nc'
            ... )
            >>> 
            >>> # Load via API
            >>> ds = forcing_tools.load_era5_data(
            ...     time_range=('2024-01-01', '2024-01-31'),
            ...     lon_range=(-80.0, -60.0),
            ...     lat_range=(30.0, 50.0),
            ...     use_api=True,
            ...     hours=['00:00', '06:00', '12:00', '18:00']
            ... )
        """
        start_time = np.datetime64(time_range[0]) if not isinstance(time_range[0], np.datetime64) else time_range[0]
        end_time = np.datetime64(time_range[1]) if not isinstance(time_range[1], np.datetime64) else time_range[1]
        
        # Convert to strings for API (convert numpy.str_ to Python str for pandas compatibility)
        start_str = str(np.datetime_as_string(start_time, unit='D'))
        end_str = str(np.datetime_as_string(end_time, unit='D'))
        
        if use_api:
            from download import ERA5Downloader
            
            print(f"Loading ERA5 data via Copernicus Climate Data Store API...")
            print(f"  Time range: {start_str} to {end_str}")
            print(f"  Lon range: {lon_range}")
            print(f"  Lat range: {lat_range}")
            
            # Use a temporary file name if netcdf_paths not provided
            if netcdf_paths is None:
                netcdf_paths = 'era5_forcing_temp.nc'
            
            if isinstance(netcdf_paths, dict):
                output_path = netcdf_paths.get('main', 'era5_forcing_temp.nc')
            else:
                output_path = netcdf_paths
            
            downloader = ERA5Downloader()
            file_path = downloader.download_era5_surface_forcing(
                output_path=output_path,
                lon_range=lon_range,
                lat_range=lat_range,
                time_range=(start_str, end_str),
                hours=hours,
                include_radiation=include_radiation,
                force_download=force_download
            )
            
            print(f"Successfully downloaded data via API to: {file_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Absolute file path: {os.path.abspath(file_path)}")
            print(f"File exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                import subprocess
                file_type = subprocess.run(['file', file_path], capture_output=True, text=True).stdout.strip()
                print(f"File type: {file_type}")
                print(f"File size: {os.path.getsize(file_path) / (1024**2):.2f} MB")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Downloaded file not found at: {file_path}")
            
            # Check if it's a zip file that needs extraction
            import zipfile
            if zipfile.is_zipfile(file_path):
                raise RuntimeError(
                    f"File {file_path} is still a zip archive and was not extracted. "
                    f"This shouldn't happen - please report this bug. "
                    f"Try running with force_download=True to re-download."
                )
            ds = xr.open_dataset(file_path)
            
        else:
            if netcdf_paths is None:
                raise ValueError("netcdf_paths is required when use_api=False")
            
            if isinstance(netcdf_paths, dict):
                print(f"Loading ERA5 data from multiple NetCDF files...")
                ds = {}
                for key, path in netcdf_paths.items():
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"NetCDF file not found: {path}")
                    print(f"  Loading {key}: {path}")
                    ds[key] = xr.open_dataset(path)
            else:
                if not os.path.exists(netcdf_paths):
                    raise FileNotFoundError(f"NetCDF file not found: {netcdf_paths}")
                print(f"Loading ERA5 data from NetCDF file: {netcdf_paths}")
                ds = xr.open_dataset(netcdf_paths)
            
            print(f"Successfully loaded data from file(s)")
        
        return ds
    
    @staticmethod
    def download_and_cache_era5(
        time_range: Tuple[Union[str, np.datetime64], Union[str, np.datetime64]],
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        output_path: str,
        hours: Optional[List[str]] = None,
        include_radiation: bool = True,
        force_download: bool = False
    ) -> str:
        """
        Download ERA5 data via API and cache it as a NetCDF file.
        
        Args:
            time_range: (start_time, end_time) as datetime64 or strings
            lon_range: (min_lon, max_lon) in degrees East
            lat_range: (min_lat, max_lat) in degrees North
            output_path: Path to save the NetCDF file
            hours: List of hours (e.g., ['00:00', '06:00', '12:00', '18:00']). None = all hours
            include_radiation: Whether to include radiation variables
            force_download: If True, re-download even if file exists
            
        Returns:
            Path to the cached NetCDF file
            
        Example:
            >>> file_path = forcing_tools.download_and_cache_era5(
            ...     time_range=('2024-01-01', '2024-01-31'),
            ...     lon_range=(-80.0, -60.0),
            ...     lat_range=(30.0, 50.0),
            ...     output_path='cached_era5_jan2024.nc',
            ...     hours=['00:00', '06:00', '12:00', '18:00']
            ... )
        """
        from download import ERA5Downloader
        
        start_time = np.datetime64(time_range[0]) if not isinstance(time_range[0], np.datetime64) else time_range[0]
        end_time = np.datetime64(time_range[1]) if not isinstance(time_range[1], np.datetime64) else time_range[1]
        
        # Convert to Python strings for pandas compatibility
        start_str = str(np.datetime_as_string(start_time, unit='D'))
        end_str = str(np.datetime_as_string(end_time, unit='D'))
        
        print(f"Downloading ERA5 data to {output_path}...")
        print(f"  Time range: {start_str} to {end_str}")
        print(f"  Lon range: {lon_range}")
        print(f"  Lat range: {lat_range}")
        
        downloader = ERA5Downloader()
        file_path = downloader.download_era5_surface_forcing(
            output_path=output_path,
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=(start_str, end_str),
            hours=hours,
            include_radiation=include_radiation,
            force_download=force_download
        )
        
        return file_path
    
    @staticmethod
    def prepare_forcing_coords(dataset, lat_var='latitude', lon_var='longitude'):
        """
        Prepare 2D coordinate arrays from atmospheric forcing dataset.
        
        Args:
            dataset: xarray Dataset with atmospheric forcing data
            lat_var: Name of latitude variable
            lon_var: Name of longitude variable
            
        Returns:
            Dictionary with 'lat', 'lon', 'lon_2d', 'lat_2d'
        """
        lat = dataset[lat_var].values
        lon = dataset[lon_var].values
        
        # Create 2D lat/lon grids
        lon_2d, lat_2d = np.meshgrid(lon, lat, indexing='xy')
        
        return {
            'lat': lat,
            'lon': lon,
            'lon_2d': lon_2d,
            'lat_2d': lat_2d
        }
    
    @staticmethod
    def interpolate_forcing_timeseries(variables_dict, source_lon_2d, source_lat_2d, 
                                      target_lon, target_lat, interp_method='linear',
                                      verbose=True):
        """
        Interpolate multiple atmospheric forcing variables over time.
        
        Args:
            variables_dict: Dictionary mapping variable names to 3D arrays (time, lat, lon)
            source_lon_2d: 2D source longitude
            source_lat_2d: 2D source latitude
            target_lon: 2D target longitude
            target_lat: 2D target latitude
            interp_method: Interpolation method
            verbose: Print progress
            
        Returns:
            Dictionary mapping variable names to interpolated 3D arrays
        """
        from interpolate import interp_tools
        
        # Get dimensions
        first_var = next(iter(variables_dict.values()))
        nt = first_var.shape[0]
        ny_tgt, nx_tgt = target_lon.shape
        
        results = {}
        
        for var_name, var_data in variables_dict.items():
            if verbose:
                print(f"Interpolating {var_name}...")
            
            interp_data = np.empty((nt, ny_tgt, nx_tgt))
            
            for t in range(nt):
                interp_data[t] = interp_tools.interp2d(
                    var_data[t], source_lon_2d, source_lat_2d, 
                    target_lon, target_lat, method=interp_method
                )
                
                if verbose and np.mod(t, 100) == 0:
                    print(f"  Time step {t+1}/{nt}")
            
            results[var_name] = interp_data
        
        return results
    
    @staticmethod
    def process_era5_forcing(era5_datasets, var_mapping, time_range=None):
        """
        Process ERA5 data and apply necessary conversions.
        
        Args:
            era5_datasets: Dictionary of ERA5 datasets (e.g., {'main': ds1, 'pair': ds2, 'rad': ds3})
            var_mapping: Dictionary mapping ROMS variable names to ERA5 variable names
            time_range: Tuple of (start_time, end_time) as datetime64 or None for all times
            
        Returns:
            Dictionary with processed forcing variables and time array
        """
        from conversions import convert_tools
        
        # Get main dataset for time reference
        main_ds = era5_datasets.get('main', next(iter(era5_datasets.values())))
        
        # Extract time
        time = main_ds['time']
        if time_range is not None:
            start_time, end_time = time_range
            time = time[(time >= start_time) & (time <= end_time)]
        
        # Compute time delta in seconds
        dt = (time[1] - time[0]).astype('timedelta64[s]').values.astype(float)
        
        processed_vars = {}
        
        # Process each variable based on mapping
        for roms_var, era5_info in var_mapping.items():
            if isinstance(era5_info, dict):
                ds_key = era5_info.get('dataset', 'main')
                var_name = era5_info.get('variable')
                conversion = era5_info.get('conversion', None)
            else:
                # Simple string mapping
                ds_key = 'main'
                var_name = era5_info
                conversion = None
            
            # Get the dataset
            ds = era5_datasets.get(ds_key, main_ds)
            
            # Extract data
            if time_range is not None:
                data = ds[var_name].sel(time=slice(time_range[0], time_range[1])).values
            else:
                data = ds[var_name].values
            
            # Apply conversion if specified
            if conversion == 'flux_density':
                data = convert_tools.convert_to_flux_density(data, dt)
            elif conversion == 'K_to_C':
                data = convert_tools.convert_K_to_C(data)
            elif conversion == 'Pa_to_mbar':
                data = convert_tools.convert_Pa_to_mbar(data)
            elif conversion == 'rainfall':
                data = convert_tools.compute_rainfall_cm_per_day(data, dt)
                data = data * 0.01 / 86400  # Convert cm/day to m/s
            elif conversion == 'relative_humidity':
                # Need two variables for this
                if 'dewpoint_var' in era5_info:
                    dewpoint = ds[era5_info['dewpoint_var']].values
                    data = convert_tools.compute_relative_humidity(data, dewpoint)
            elif conversion == 'surface_wind':
                data = convert_tools.calculate_surface_wind(data)
            
            # Handle NaN values
            if np.any(np.isnan(data)):
                data = np.nan_to_num(data, nan=np.nanmean(data))
            
            processed_vars[roms_var] = data
        
        return {
            'variables': processed_vars,
            'time': time
        }
    
    @staticmethod
    def create_surface_forcing_dataset(variables, time, grid_dims, ref_time='2000-01-01T00:00:00',
                                      source_name='Unknown', add_zero_fields=True):
        """
        Create a complete ROMS surface forcing xarray Dataset.
        
        Args:
            variables: Dictionary mapping ROMS variable names to interpolated 3D arrays
            time: Time coordinate (datetime64 array)
            grid_dims: Tuple of (eta_rho, xi_rho)
            ref_time: Reference time as string or datetime64
            source_name: Name of data source
            add_zero_fields: Whether to add zero-valued fields (bhflux, bwflux, EminusP)
            
        Returns:
            xarray Dataset ready to save as ROMS surface forcing file
        """
        ref_time = np.datetime64(ref_time)
        time_days = (time.values - ref_time) / np.timedelta64(1, 'D')
        
        eta_rho, xi_rho = grid_dims
        nt = len(time)
        
        # Define variable attributes
        var_attrs = {
            'swrad': {'long_name': 'net short-wave (solar) radiation', 'units': 'W/m^2'},
            'lwrad': {'long_name': 'net long-wave (thermal) radiation', 'units': 'W/m^2'},
            'Tair': {'long_name': 'air temperature at 2m', 'units': 'degrees Celsius'},
            'qair': {'long_name': 'absolute humidity at 2m', 'units': 'kg/kg'},
            'rain': {'long_name': 'total precipitation', 'units': 'm/s'},
            'Uwind': {'long_name': '10 meter wind in x-direction', 'units': 'm/s'},
            'Vwind': {'long_name': '10 meter wind in y-direction', 'units': 'm/s'},
            'Pair': {'long_name': 'mean sea level pressure', 'units': 'mbar'},
            'bhflux': {'long_name': 'bottom heat flux', 'units': 'W/m^2'},
            'EminusP': {'long_name': 'surface upward freshwater flux (evaporation minus precipitation)', 'units': 'm/s'},
            'bwflux': {'long_name': 'bottom freshwater flux', 'units': 'm/s'},
        }
        
        # Create data variables dictionary
        data_vars = {}
        for var_name, var_data in variables.items():
            attrs = var_attrs.get(var_name, {'long_name': var_name, 'units': 'unknown'})
            data_vars[var_name] = (['wind_time', 'eta_rho', 'xi_rho'], var_data, attrs)
        
        # Add zero-valued fields if requested
        if add_zero_fields:
            zero_fields = ['bhflux', 'EminusP', 'bwflux']
            for field in zero_fields:
                if field not in data_vars:
                    attrs = var_attrs.get(field, {'long_name': field, 'units': 'unknown'})
                    data_vars[field] = (
                        ['wind_time', 'eta_rho', 'xi_rho'], 
                        np.zeros((nt, eta_rho, xi_rho)), 
                        attrs
                    )
        
        # Create dataset
        ds = xr.Dataset(
            data_vars,
            coords={
                'wind_time': ('wind_time', time_days, {
                    'units': 'days since 2000-01-01 00:00:00',
                }),
                'eta_rho': ('eta_rho', np.arange(eta_rho)),
                'xi_rho': ('xi_rho', np.arange(xi_rho)),
            },
            attrs={
                'title': 'ROMS surface forcing file created by model-tools',
                'start_time': str(time[0].values)[:10] + ' 00:00:00',
                'end_time': str(time[-1].values)[:10] + ' 00:00:00',
                'source': source_name,
                'model_reference_date': str(ref_time),
            }
        )
        
        return ds
