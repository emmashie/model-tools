import numpy as np
import xarray as xr

class forcing_tools:
    """Tools for creating ROMS surface forcing files."""
    
    def __init__(self):
        pass
    
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
