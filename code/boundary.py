import numpy as np
import xarray as xr

class boundary_tools:
    """Tools for creating ROMS boundary condition and climatology files."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def extract_boundary_transects(temp, salt, u, v, ubar, vbar, zeta, grid, boundaries):
        """
        Extract boundary transects from interpolated full-grid arrays.
        
        Args:
            temp: Temperature array (nt, s_rho, eta_rho, xi_rho)
            salt: Salinity array (nt, s_rho, eta_rho, xi_rho)
            u: U-velocity array (nt, s_rho, eta_rho, xi_u)
            v: V-velocity array (nt, s_rho, eta_v, xi_rho)
            ubar: Barotropic U-velocity (nt, eta_rho, xi_u)
            vbar: Barotropic V-velocity (nt, eta_v, xi_rho)
            zeta: Sea surface height (nt, eta_rho, xi_rho)
            grid: xarray Dataset with ROMS grid
            boundaries: Dictionary with boundary flags {'west': bool, 'east': bool, ...}
            
        Returns:
            Dictionary with boundary transect data for each active boundary
        """
        boundary_transects = {}
        
        roms_lon_2d = grid.lon_rho.values
        roms_lat_2d = grid.lat_rho.values
        
        if boundaries.get('west', False):
            # Western boundary: first xi index (xi=0), all eta
            boundary_transects['west'] = {
                'temp': temp[:, :, :, 0],
                'salt': salt[:, :, :, 0],
                'u': u[:, :, :, 0],
                'v': v[:, :, :, 0],
                'ubar': ubar[:, :, 0],
                'vbar': vbar[:, :, 0],
                'zeta': zeta[:, :, 0],
                'lon': roms_lon_2d[:, 0],
                'lat': roms_lat_2d[:, 0]
            }
        
        if boundaries.get('east', False):
            # Eastern boundary: last xi index (xi=-1), all eta
            boundary_transects['east'] = {
                'temp': temp[:, :, :, -1],
                'salt': salt[:, :, :, -1],
                'u': u[:, :, :, -1],
                'v': v[:, :, :, -1],
                'ubar': ubar[:, :, -1],
                'vbar': vbar[:, :, -1],
                'zeta': zeta[:, :, -1],
                'lon': roms_lon_2d[:, -1],
                'lat': roms_lat_2d[:, -1]
            }
        
        if boundaries.get('south', False):
            # Southern boundary: first eta index (eta=0), all xi
            boundary_transects['south'] = {
                'temp': temp[:, :, 0, :],
                'salt': salt[:, :, 0, :],
                'u': u[:, :, 0, :],
                'v': v[:, :, 0, :],
                'ubar': ubar[:, 0, :],
                'vbar': vbar[:, 0, :],
                'zeta': zeta[:, 0, :],
                'lon': roms_lon_2d[0, :],
                'lat': roms_lat_2d[0, :]
            }
        
        if boundaries.get('north', False):
            # Northern boundary: last eta index (eta=-1), all xi
            boundary_transects['north'] = {
                'temp': temp[:, :, -1, :],
                'salt': salt[:, :, -1, :],
                'u': u[:, :, -1, :],
                'v': v[:, :, -1, :],
                'ubar': ubar[:, -1, :],
                'vbar': vbar[:, -1, :],
                'zeta': zeta[:, -1, :],
                'lon': roms_lon_2d[-1, :],
                'lat': roms_lat_2d[-1, :]
            }
        
        return boundary_transects
    
    @staticmethod
    def create_climatology_dataset(temp, salt, u, v, w, ubar, vbar, zeta, 
                                   time_days, grid, source_name='Unknown'):
        """
        Create ROMS climatology dataset.
        
        Args:
            temp: Temperature array (nt, s_rho, eta_rho, xi_rho)
            salt: Salinity array (nt, s_rho, eta_rho, xi_rho)
            u: U-velocity array (nt, s_rho, eta_rho, xi_u)
            v: V-velocity array (nt, s_rho, eta_v, xi_rho)
            w: W-velocity array (nt, s_rho, eta_rho, xi_rho)
            ubar: Barotropic U-velocity (nt, eta_rho, xi_u)
            vbar: Barotropic V-velocity (nt, eta_v, xi_rho)
            zeta: Sea surface height (nt, eta_rho, xi_rho)
            time_days: Time array in days since reference
            grid: xarray Dataset with ROMS grid
            source_name: Name of data source
            
        Returns:
            xarray Dataset ready to save as ROMS climatology file
        """
        from grid import grid_tools
        
        eta_rho, xi_rho, eta_v, xi_u, s_rho, s_w = grid_tools.get_grid_dims(grid)
        
        ds = xr.Dataset(
            {
                'temp': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), temp),
                'salt': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), salt),
                'u': (('ocean_time', 's_rho', 'eta_rho', 'xi_u'), u),
                'v': (('ocean_time', 's_rho', 'eta_v', 'xi_rho'), v),
                'w': (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), w),
                'Cs_r': (('s_rho'), grid.Cs_r.values),
                'Cs_w': (('s_w'), grid.Cs_w.values),
                'zeta': (('ocean_time', 'eta_rho', 'xi_rho'), zeta),
                'ubar': (('ocean_time', 'eta_rho', 'xi_u'), ubar),
                'vbar': (('ocean_time', 'eta_v', 'xi_rho'), vbar),
            },
            coords={
                'ocean_time': time_days,
                's_rho': np.arange(s_rho),
                'eta_rho': np.arange(eta_rho),
                'xi_rho': np.arange(xi_rho),
                'xi_u': np.arange(xi_u),
                'eta_v': np.arange(eta_v),
                's_w': np.arange(s_w),
            },
            attrs={
                'title': "ROMS climatology file created by model-tools",
                'model_reference_date': "2000-01-01 00:00:00",
                'source': source_name,
                'theta_s': grid.theta_s,
                'theta_b': grid.theta_b,
                'hc': grid.hc,
            }
        )
        
        # Variable attributes
        ds['temp'].attrs = dict(long_name="potential temperature", units="degrees Celsius", coordinates="ocean_time")
        ds['salt'].attrs = dict(long_name="salinity", units="PSU", coordinates="ocean_time")
        ds['u'].attrs = dict(long_name="u-flux component", units="m/s", coordinates="ocean_time")
        ds['v'].attrs = dict(long_name="v-flux component", units="m/s", coordinates="ocean_time")
        ds['w'].attrs = dict(long_name="w-flux component", units="m/s", coordinates="ocean_time")
        ds['Cs_r'].attrs = dict(long_name="Vertical stretching function at rho-points", units="nondimensional")
        ds['Cs_w'].attrs = dict(long_name="Vertical stretching function at w-points", units="nondimensional")
        ds['zeta'].attrs = dict(long_name="sea surface height", units="m", coordinates="ocean_time")
        ds['ubar'].attrs = dict(long_name="vertically integrated u-flux component", units="m/s", coordinates="ocean_time")
        ds['vbar'].attrs = dict(long_name="vertically integrated v-flux component", units="m/s", coordinates="ocean_time")
        ds['ocean_time'].attrs = dict(long_name='relative time: days since 2000-01-01 00:00:00', units='days')
        
        # Swap dimensions for temp and salt to use their own time coordinates
        ds['temp'] = ds['temp'].swap_dims({'ocean_time': 'temp_time'})
        ds['salt'] = ds['salt'].swap_dims({'ocean_time': 'salt_time'})
        
        return ds
    
    @staticmethod
    def create_boundary_dataset(boundary_transects, time_days, grid, start_time, end_time, 
                               source_name='Unknown'):
        """
        Create ROMS boundary forcing dataset from boundary transects.
        
        Args:
            boundary_transects: Dictionary with boundary data from extract_boundary_transects
            time_days: Time array in days since reference
            grid: xarray Dataset with ROMS grid
            start_time: Start time as string or datetime
            end_time: End time as string or datetime
            source_name: Name of data source
            
        Returns:
            xarray Dataset ready to save as ROMS boundary forcing file
        """
        from grid import grid_tools
        
        eta_rho, xi_rho, eta_v, xi_u, s_rho, s_w = grid_tools.get_grid_dims(grid)
        nt = len(time_days)
        
        # Helper to get boundary or fill with NaN if not present
        def get_bry(var, b, shape):
            if b in boundary_transects:
                return boundary_transects[b][var]
            else:
                return np.full(shape, np.nan)
        
        ds_bry = xr.Dataset(
            {
                # South boundary
                'u_south': (['bry_time', 's_rho', 'xi_u'], get_bry('u', 'south', (nt, s_rho, xi_u))),
                'v_south': (['bry_time', 's_rho', 'xi_rho'], get_bry('v', 'south', (nt, s_rho, xi_rho))),
                'temp_south': (['bry_time', 's_rho', 'xi_rho'], get_bry('temp', 'south', (nt, s_rho, xi_rho))),
                'salt_south': (['bry_time', 's_rho', 'xi_rho'], get_bry('salt', 'south', (nt, s_rho, xi_rho))),
                'zeta_south': (['bry_time', 'xi_rho'], get_bry('zeta', 'south', (nt, xi_rho))),
                'ubar_south': (['bry_time', 'xi_u'], get_bry('ubar', 'south', (nt, xi_u))),
                'vbar_south': (['bry_time', 'xi_rho'], get_bry('vbar', 'south', (nt, xi_rho))),
                
                # East boundary
                'u_east': (['bry_time', 's_rho', 'eta_rho'], get_bry('u', 'east', (nt, s_rho, eta_rho))),
                'v_east': (['bry_time', 's_rho', 'eta_v'], get_bry('v', 'east', (nt, s_rho, eta_v))),
                'temp_east': (['bry_time', 's_rho', 'eta_rho'], get_bry('temp', 'east', (nt, s_rho, eta_rho))),
                'salt_east': (['bry_time', 's_rho', 'eta_rho'], get_bry('salt', 'east', (nt, s_rho, eta_rho))),
                'zeta_east': (['bry_time', 'eta_rho'], get_bry('zeta', 'east', (nt, eta_rho))),
                'ubar_east': (['bry_time', 'eta_rho'], get_bry('ubar', 'east', (nt, eta_rho))),
                'vbar_east': (['bry_time', 'eta_v'], get_bry('vbar', 'east', (nt, eta_v))),
                
                # North boundary
                'u_north': (['bry_time', 's_rho', 'xi_u'], get_bry('u', 'north', (nt, s_rho, xi_u))),
                'v_north': (['bry_time', 's_rho', 'xi_rho'], get_bry('v', 'north', (nt, s_rho, xi_rho))),
                'temp_north': (['bry_time', 's_rho', 'xi_rho'], get_bry('temp', 'north', (nt, s_rho, xi_rho))),
                'salt_north': (['bry_time', 's_rho', 'xi_rho'], get_bry('salt', 'north', (nt, s_rho, xi_rho))),
                'zeta_north': (['bry_time', 'xi_rho'], get_bry('zeta', 'north', (nt, xi_rho))),
                'ubar_north': (['bry_time', 'xi_u'], get_bry('ubar', 'north', (nt, xi_u))),
                'vbar_north': (['bry_time', 'xi_rho'], get_bry('vbar', 'north', (nt, xi_rho))),
                
                # West boundary
                'u_west': (['bry_time', 's_rho', 'eta_rho'], get_bry('u', 'west', (nt, s_rho, eta_rho))),
                'v_west': (['bry_time', 's_rho', 'eta_v'], get_bry('v', 'west', (nt, s_rho, eta_v))),
                'temp_west': (['bry_time', 's_rho', 'eta_rho'], get_bry('temp', 'west', (nt, s_rho, eta_rho))),
                'salt_west': (['bry_time', 's_rho', 'eta_rho'], get_bry('salt', 'west', (nt, s_rho, eta_rho))),
                'zeta_west': (['bry_time', 'eta_rho'], get_bry('zeta', 'west', (nt, eta_rho))),
                'ubar_west': (['bry_time', 'eta_rho'], get_bry('ubar', 'west', (nt, eta_rho))),
                'vbar_west': (['bry_time', 'eta_v'], get_bry('vbar', 'west', (nt, eta_v))),
            },
            coords={
                'bry_time': ('bry_time', time_days),
                's_rho': ('s_rho', np.arange(s_rho)),
                'xi_u': ('xi_u', np.arange(xi_u)),
                'xi_rho': ('xi_rho', np.arange(xi_rho)),
                'eta_rho': ('eta_rho', np.arange(eta_rho)),
                'eta_v': ('eta_v', np.arange(eta_v)),
            },
            attrs={
                'title': 'ROMS boundary forcing file created by model-tools',
                'start_time': str(start_time),
                'end_time': str(end_time),
                'source': source_name,
                'model_reference_date': '2000-01-01 00:00:00',
                'theta_s': float(grid.theta_s),
                'theta_b': float(grid.theta_b),
                'hc': float(grid.hc),
            }
        )
        
        # Set variable attributes
        for boundary in ['south', 'east', 'north', 'west']:
            ds_bry[f'u_{boundary}'].attrs = dict(
                long_name=f"{boundary} boundary u-flux component", 
                units="m/s", coordinates="bry_time"
            )
            ds_bry[f'v_{boundary}'].attrs = dict(
                long_name=f"{boundary} boundary v-flux component", 
                units="m/s", coordinates="bry_time"
            )
            ds_bry[f'temp_{boundary}'].attrs = dict(
                long_name=f"{boundary} boundary potential temperature", 
                units="degrees Celsius", coordinates="bry_time"
            )
            ds_bry[f'salt_{boundary}'].attrs = dict(
                long_name=f"{boundary} boundary salinity", 
                units="PSU", coordinates="bry_time"
            )
            ds_bry[f'zeta_{boundary}'].attrs = dict(
                long_name=f"{boundary} boundary sea surface height", 
                units="m", coordinates="bry_time"
            )
            ds_bry[f'ubar_{boundary}'].attrs = dict(
                long_name=f"{boundary} boundary barotropic u-velocity", 
                units="m/s", coordinates="bry_time"
            )
            ds_bry[f'vbar_{boundary}'].attrs = dict(
                long_name=f"{boundary} boundary barotropic v-velocity", 
                units="m/s", coordinates="bry_time"
            )
        
        ds_bry['bry_time'].attrs = dict(
            long_name='relative time: days since 2000-01-01 00:00:00', 
            units='days'
        )
        
        return ds_bry
