import numpy as np
import xarray as xr

class init_tools:
    """Tools for creating ROMS initial condition files."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def add_deep_ocean_layer(dataset, new_depth, fill_values, time_var='time', 
                            depth_var='depth', lat_var='latitude', lon_var='longitude'):
        """
        Add a deep ocean layer to the dataset with specified fill values.
        
        Args:
            dataset: xarray Dataset with ocean data
            new_depth: Depth value for new layer (positive, meters)
            fill_values: Dictionary mapping variable names to fill values
            time_var: Name of time dimension
            depth_var: Name of depth dimension
            lat_var: Name of latitude dimension
            lon_var: Name of longitude dimension
            
        Returns:
            xarray Dataset with added deep layer
        """
        new_data_vars = {}
        
        for var_name, fill_val in fill_values.items():
            if var_name not in dataset:
                continue
                
            # Create array with same shape as original but with depth dimension = 1
            shape = (len(dataset[time_var]), 1, len(dataset[lat_var]), len(dataset[lon_var]))
            data = np.full(shape, fill_val, dtype=np.float32)
            
            new_data_vars[var_name] = xr.DataArray(
                data,
                coords={
                    time_var: dataset[time_var],
                    depth_var: [new_depth],
                    lat_var: dataset[lat_var],
                    lon_var: dataset[lon_var]
                },
                dims=[time_var, depth_var, lat_var, lon_var]
            )
        
        # Create new dataset and concatenate with original
        new_layer = xr.Dataset(new_data_vars, attrs=dataset.attrs)
        extended_dataset = xr.concat([dataset, new_layer], dim=depth_var)
        
        return extended_dataset
    
    @staticmethod
    def compute_time_since_reference(init_time, ref_time='2000-01-01T00:00:00'):
        """
        Compute time in seconds/days since reference time.
        
        Args:
            init_time: numpy datetime64 or string
            ref_time: Reference time as string or datetime64
            
        Returns:
            Tuple of (seconds_since_ref, days_since_ref)
        """
        init_time = np.datetime64(init_time)
        ref_time = np.datetime64(ref_time)
        
        seconds_since = (init_time - ref_time) / np.timedelta64(1, 's')
        days_since = seconds_since / 86400.0
        
        return seconds_since, days_since
    
    @staticmethod
    def prepare_source_coords(dataset, depth_var, lat_var, lon_var):
        """
        Prepare 2D and 3D coordinate arrays from source dataset.
        
        Args:
            dataset: xarray Dataset with source data
            depth_var: Name of depth variable
            lat_var: Name of latitude variable
            lon_var: Name of longitude variable
            
        Returns:
            Dictionary with 'depth', 'lat', 'lon', 'lon_2d', 'lat_2d', 'depth_3d'
        """
        depth = dataset[depth_var].values
        lat = dataset[lat_var].values
        lon = dataset[lon_var].values
        
        # Create 2D lat/lon grids
        lon_2d, lat_2d = np.meshgrid(lon, lat, indexing='xy')
        
        # Create 3D depth array (depth varies with depth dimension, constant across lat/lon)
        depth_3d = np.broadcast_to(
            depth[np.newaxis, np.newaxis, :], 
            (len(lat), len(lon), len(depth))
        )
        
        return {
            'depth': depth,
            'lat': lat,
            'lon': lon,
            'lon_2d': lon_2d,
            'lat_2d': lat_2d,
            'depth_3d': depth_3d
        }
    
    @staticmethod
    def interpolate_and_mask_3d(data, source_coords, target_lon, target_lat, target_depth, 
                                mask, fill_value, interp_method='linear', min_value=None):
        """
        Interpolate 3D data and apply masking with fill values.
        
        Args:
            data: 3D array (nz, ny, nx)
            source_coords: Dict with 'lon_2d', 'lat_2d', 'depth'
            target_lon: 2D target longitude
            target_lat: 2D target latitude
            target_depth: 3D target depth
            mask: 2D boolean mask (True=water, False=land)
            fill_value: Value to fill masked/NaN regions
            interp_method: Interpolation method
            min_value: Minimum allowed value (optional)
            
        Returns:
            Interpolated and masked 3D array
        """
        from interpolate import interp_tools
        
        # Ensure data has correct shape
        if data.shape[0] != len(source_coords['depth']):
            data = np.transpose(data, (2, 0, 1))
        
        # Interpolate
        result = interp_tools.interp3d(
            data, 
            source_coords['lon_2d'], 
            source_coords['lat_2d'], 
            source_coords['depth'],
            target_lon, 
            target_lat, 
            target_depth, 
            method=interp_method
        )
        
        # Apply minimum value constraint if specified
        if min_value is not None:
            result = np.where(result < min_value, min_value, result)
        
        # Apply mask and fill NaNs
        result = np.where(mask, result, np.nan)
        result = np.where(np.isnan(result), fill_value, result)
        
        return result
    
    @staticmethod
    def interpolate_and_mask_2d(data, source_lon_2d, source_lat_2d, 
                                target_lon, target_lat, fill_value, interp_method='linear'):
        """
        Interpolate 2D data and fill NaN values.
        
        Args:
            data: 2D array (ny, nx)
            source_lon_2d: 2D source longitude
            source_lat_2d: 2D source latitude
            target_lon: 2D target longitude
            target_lat: 2D target latitude
            fill_value: Value to fill NaN regions
            interp_method: Interpolation method
            
        Returns:
            Interpolated 2D array
        """
        from interpolate import interp_tools
        
        result = interp_tools.interp2d(
            data, source_lon_2d, source_lat_2d, 
            target_lon, target_lat, 
            method=interp_method
        )
        
        result = np.where(np.isnan(result), fill_value, result)
        
        return result
    
    @staticmethod
    def create_initial_conditions_dataset(grid, temp, salt, u, v, w, ubar, vbar, zeta, 
                                         ocean_time_days, init_time, source_name='Unknown'):
        """
        Create a complete ROMS initial conditions xarray Dataset.
        
        Args:
            grid: xarray Dataset with ROMS grid
            temp: 3D temperature array (s_rho, eta_rho, xi_rho)
            salt: 3D salinity array (s_rho, eta_rho, xi_rho)
            u: 3D u-velocity array (s_rho, eta_rho, xi_u)
            v: 3D v-velocity array (s_rho, eta_v, xi_rho)
            w: 3D w-velocity array (s_rho, eta_rho, xi_rho)
            ubar: 2D depth-averaged u-velocity (eta_rho, xi_u)
            vbar: 2D depth-averaged v-velocity (eta_v, xi_rho)
            zeta: 2D sea surface height (eta_rho, xi_rho)
            ocean_time_days: Time in days since reference
            init_time: Initialization time as string or datetime64
            source_name: Name of data source
            
        Returns:
            xarray Dataset ready to save as ROMS initial conditions file
        """
        from grid import grid_tools
        
        # Get grid dimensions
        eta_rho, xi_rho, eta_v, xi_u, s_rho, s_w = grid_tools.get_grid_dims(grid)
        
        ds = xr.Dataset(
            {
                'temp': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), temp[np.newaxis, :, :, :]),
                'salt': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), salt[np.newaxis, :, :, :]),
                'u': (('ocean_time', 's_rho', 'eta_rho', 'xi_u'), u[np.newaxis, :, :, :]),
                'v': (('ocean_time', 's_rho', 'eta_v', 'xi_rho'), v[np.newaxis, :, :, :]),
                'w': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), w[np.newaxis, :, :, :]),
                'Cs_r': (('ocean_time', 's_rho'), grid.Cs_r.values[np.newaxis, :]),
                'Cs_w': (('ocean_time', 's_w'), grid.Cs_w.values[np.newaxis, :]),
                'zeta': (('ocean_time', 'eta_rho', 'xi_rho'), zeta[np.newaxis, :, :]),
                'ubar': (('ocean_time', 'eta_rho', 'xi_u'), ubar[np.newaxis, :, :]),
                'vbar': (('ocean_time', 'eta_v', 'xi_rho'), vbar[np.newaxis, :, :]),
            },
            coords={
                'ocean_time': [ocean_time_days],
                's_rho': np.arange(s_rho),
                'eta_rho': np.arange(eta_rho),
                'xi_rho': np.arange(xi_rho),
                'xi_u': np.arange(xi_u),
                'eta_v': np.arange(eta_v),
                's_w': np.arange(s_w),
            },
            attrs={
                'title': "ROMS initial conditions file created by model-tools",
                'roms_tools_version': "2.4.0",
                'ini_time': str(init_time),
                'model_reference_date': "2000-01-01 00:00:00",
                'source': source_name,
                'theta_s': grid.theta_s,
                'theta_b': grid.theta_b,
                'hc': grid.hc,
            }
        )
        
        # Add variable attributes
        ds['temp'].attrs = dict(long_name="potential temperature", units="degrees Celsius", coordinates="ocean_time")
        ds['salt'].attrs = dict(long_name="salinity", units="PSU", coordinates="ocean_time")
        ds['u'].attrs = dict(long_name="u-flux component", units="m/s", coordinates="ocean_time")
        ds['v'].attrs = dict(long_name="v-flux component", units="m/s", coordinates="ocean_time")
        ds['w'].attrs = dict(long_name="w-flux component", units="m/s", coordinates="ocean_time")
        ds['Cs_r'].attrs = dict(long_name="Vertical stretching function at rho-points", units="nondimensional", coordinates="ocean_time")
        ds['Cs_w'].attrs = dict(long_name="Vertical stretching function at w-points", units="nondimensional", coordinates="ocean_time")
        ds['zeta'].attrs = dict(long_name="sea surface height", units="m", coordinates="ocean_time")
        ds['ubar'].attrs = dict(long_name="vertically integrated u-flux component", units="m/s", coordinates="ocean_time")
        ds['vbar'].attrs = dict(long_name="vertically integrated v-flux component", units="m/s", coordinates="ocean_time")
        ds['ocean_time'].attrs = dict(long_name='relative time: days since 2000-01-01 00:00:00', units='days')
        
        return ds
