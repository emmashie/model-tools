import numpy as np
 
class convert_tools:

    @staticmethod
    def compute_relative_humidity(t_air, t_dew):
        """
        Compute surface air relative humidity given 2-meter air temperature and dew point.
        Parameters:
            t_air (float): 2-meter air temperature in Kelvin
            t_dew (float): 2-meter dew point temperature in Kelvin
        Returns:
            float: Relative humidity in percentage
        """
        t_air_c = t_air - 273.15
        t_dew_c = t_dew - 273.15
        e_sat_air = 6.112 * np.exp((17.67 * t_air_c) / (t_air_c + 243.5))
        e_sat_dew = 6.112 * np.exp((17.67 * t_dew_c) / (t_dew_c + 243.5))
        rh = (e_sat_dew / e_sat_air) * 100
        return rh

    @staticmethod
    def convert_to_energy_density(radiation_wm2, time_interval):
        """
        Converts radiation from W/m² to J/m² given a time interval.
        Parameters:
            radiation_wm2 (array-like): Radiation values in W/m² (can be 1D, 2D, or 3D: [nt, ny, nx]).
            time_interval (float): Time interval in seconds over which the radiation is measured.
        Returns:
            array-like: Radiation values converted to J/m², same shape as input.
        """
        energy_density = np.asarray(radiation_wm2) * time_interval
        return energy_density

    @staticmethod
    def calculate_surface_wind(u10_value, z0=0.0002, z_ref=10):
        """Calculate surface wind speed at a reference height using the logarithmic wind profile."""
        return u10_value * (np.log(z0) / np.log(z_ref / z0))

    @staticmethod
    def convert_Pa_to_mbar(pressure_pa):
        """Convert pressure from Pascals to millibars."""
        return pressure_pa / 100.0

    @staticmethod
    def convert_K_to_C(temperature_k):
        """Convert temperature from Kelvin to Celsius."""
        return temperature_k - 273.15

    @staticmethod
    def compute_rainfall_cm_per_day(total_precipitation_m, time_interval_seconds):
        """
        Convert total precipitation from meters to rainfall in cm/day.
        Parameters:
            total_precipitation_m (list or array): Total precipitation time series in meters.
            time_interval_seconds (float): Time interval of the data in seconds (e.g., 3600 for hourly data).
        Returns:
            list: Rainfall rate in cm/day.
        """
        seconds_per_day = 86400
        rainfall_cm_per_day = [
            (precip * seconds_per_day) / time_interval_seconds * 100
            for precip in total_precipitation_m
        ]
        return rainfall_cm_per_day


    @staticmethod
    def compute_barotropic_component(vel, z, axis):
        """
        Helper function to compute barotropic component for a single velocity component.
        """
        
        # Compute layer thicknesses using simple differences
        hz = np.abs(np.diff(z, axis=axis))
        
        # For the boundary layer, duplicate the edge thickness
        # Add one layer at the end with the same thickness as the last computed layer
        hz_padded = np.concatenate([
            hz, 
            np.take(hz, [-1], axis=axis)
        ], axis=axis)
        
        # Compute depth-weighted velocities
        vel_weighted = vel * hz_padded
        
        # Sum along the vertical axis
        vel_integrated = np.sum(vel_weighted, axis=axis)
        
        # Total depth
        total_depth = np.sum(hz_padded, axis=axis)
        
        # Avoid division by zero
        mask = total_depth > 0
        
        # Initialize output array
        vel_bar = np.zeros_like(total_depth)
        
        # Compute depth-averaged velocity
        vel_bar[mask] = vel_integrated[mask] / total_depth[mask]
        
        return vel_bar


    @staticmethod
    def compute_uvbar(u, v, z, axis=0):
        """
        Compute depth-averaged (barotropic) velocity components from 3D velocity profiles
        on ROMS staggered grids.
        
        Parameters:
        -----------
        u : numpy.ndarray
            3D array of u-velocity component on u-grid (nz, ny, nx-1)
        v : numpy.ndarray  
            3D array of v-velocity component on v-grid (nz, ny-1, nx)
        z : numpy.ndarray
            3D array of depth values on rho-grid (nz, ny, nx)
            Assumed to be positive downward, ordered from surface to bottom
        axis : int, optional
            Vertical axis for integration (default: 0 for ROMS convention)
            
        Returns:
        --------
        ubar : numpy.ndarray
            2D depth-averaged u-velocity on u-grid (ny, nx-1)
        vbar : numpy.ndarray
            2D depth-averaged v-velocity on v-grid (ny-1, nx)
        """
        
        # Validate input shapes for ROMS grid convention
        nz, ny, nx = z.shape
        
        if u.shape != (nz, ny, nx-1):
            raise ValueError(f"u must have shape ({nz}, {ny}, {nx-1}), got {u.shape}")
        if v.shape != (nz, ny-1, nx):
            raise ValueError(f"v must have shape ({nz}, {ny-1}, {nx}), got {v.shape}")
        
        # Convert z from rho-grid to u-grid and v-grid
        # u-grid: average z in x-direction (rho points to u points)
        z_u = (z[:, :, :-1] + z[:, :, 1:]) / 2.0
        
        # v-grid: average z in y-direction (rho points to v points)  
        z_v = (z[:, :-1, :] + z[:, 1:, :]) / 2.0
        
        # Compute ubar using z_u
        ubar = convert_tools.compute_barotropic_component(u, z_u, axis)

        # Compute vbar using z_v
        vbar = convert_tools.compute_barotropic_component(v, z_v, axis)

        return ubar, vbar

    @staticmethod
    def compute_w(u, v, pm, pn, z_rho, axis=0):
        """
        Compute vertical velocity w from 3D u, v, pm, pn, and z_rho arrays using the continuity equation:
            du/dx + dv/dy = -dw/dz
        Parameters:
            u: 3D array (nz, ny, nx-1) - u-velocity on u-grid
            v: 3D array (nz, ny-1, nx) - v-velocity on v-grid
            pm: 2D array (ny, nx) - grid metric in x (1/dx)
            pn: 2D array (ny, nx) - grid metric in y (1/dy)
            z_rho: 3D array (nz, ny, nx) - vertical positions on rho grid
            axis: int - vertical axis (default 0)
        Returns:
            w: 3D array (nz+1, ny, nx) - vertical velocity at w-points (staggered in z)
        """
        nz, ny, nx = z_rho.shape
        # Compute z_w (vertical positions at w-points, i.e., interfaces)
        z_w = np.zeros((nz+1, ny, nx))
        z_w[1:-1] = 0.5 * (z_rho[:-1] + z_rho[1:])
        z_w[0] = z_rho[0]
        z_w[-1] = z_rho[-1]

        dz = np.diff(z_w, axis=axis)  # (nz, ny, nx)

        # Compute u and v on rho points by averaging to cell centers
        u_rho = 0.5 * (u[:, :, :-1] + u[:, :, 1:])  # (nz, ny, nx-2)
        v_rho = 0.5 * (v[:, :-1, :] + v[:, 1:, :])  # (nz, ny-2, nx)

        # Pad u_rho and v_rho to match (nz, ny, nx)
        u_rho_full = np.zeros((nz, ny, nx))
        v_rho_full = np.zeros((nz, ny, nx))
        u_rho_full[:, :, 1:-1] = u_rho
        u_rho_full[:, :, 0] = u[:, :, 0]
        u_rho_full[:, :, -1] = u[:, :, -1]
        v_rho_full[:, 1:-1, :] = v_rho
        v_rho_full[:, 0, :] = v[:, 0, :]
        v_rho_full[:, -1, :] = v[:, -1, :]

        # Compute divergence of horizontal velocity
        # Use np.gradient for central differences, then multiply by pm/pn
        dudx = np.gradient(u_rho_full, axis=2) * pm[np.newaxis, :, :]
        dvdy = np.gradient(v_rho_full, axis=1) * pn[np.newaxis, :, :]

        # Compute divergence
        div = dudx + dvdy  # (nz, ny, nx)

        # Integrate -div from the bottom up to get w at w-points
        w = np.zeros((nz+1, ny, nx))
        for k in range(1, nz+1):
            w[k] = w[k-1] - div[k-1] * dz[k-1]

        return w[1:,:,:]
