import numpy as np
import funpy.model_utils as mod_utils 

class grid_tools:
    def __init__(self):
        pass

    @staticmethod
    def compute_rx0(bathymetry):
        """
        Compute rx0 and optionally return locations of violations.
        """
        h = np.asarray(bathymetry, dtype=float)
        h = np.where(h <= 0, np.nan, h)
        
        # X-direction differences
        h_i = h[:, :-1]
        h_i_plus_1 = h[:, 1:]
        rx0_x = np.abs(h_i_plus_1 - h_i) / (h_i_plus_1 + h_i)
        
        # Y-direction differences
        h_j = h[:-1, :]
        h_j_plus_1 = h[1:, :]
        rx0_y = np.abs(h_j_plus_1 - h_j) / (h_j_plus_1 + h_j)
        
        return rx0_x, rx0_y  

    @staticmethod
    def compute_rx1(bathymetry):
        """
        Compute rx1 (Haney) slope parameter.
        rx1 = max(|h(i+1,j) - h(i-1,j)| / [h(i+1,j) + h(i-1,j)])
        """
        h = np.asarray(bathymetry, dtype=float)
        
        # X-direction: skip one cell
        rx1_x = np.abs(h[:, 2:] - h[:, :-2]) / (h[:, 2:] + h[:, :-2])
        
        # Y-direction: skip one cell
        rx1_y = np.abs(h[2:, :] - h[:-2, :]) / (h[2:, :] + h[:-2, :])
        
        # Return maximum value
        return rx1_x, rx1_y

    @staticmethod
    def compute_sigma(N, type='r'):
        if type=='r':
            k = np.arange(1, N + 1)
            sigma = (k - N - 0.5) / N
        elif type=='w':
            k = np.arange(N + 1)
            sigma = (k - N) / N
        else:
            raise ValueError("type must be 'r' for rho or 'w' for w coordinates")
        return sigma 

    @staticmethod
    def compute_cs(sigma, theta_s, theta_b):
        C = (1 - np.cosh(theta_s * sigma)) / (np.cosh(theta_s) - 1)
        C = (np.exp(theta_b * C) - 1) / (1 - np.exp(-theta_b))
        return C

    @staticmethod
    def compute_z(sigma, hc, C, h, zeta):
        # sigma: (nz,), C: (nz,), h: (ny, nx), zeta: (ny, nx, nt)
        nz = sigma.shape[0]
        ny, nx = h.shape
        nt = zeta.shape[2]

        # Expand sigma and C to (nz, 1, 1, 1)
        sigma = sigma[:, None, None, None]  # (nz, 1, 1, 1)
        C = C[:, None, None, None]          # (nz, 1, 1, 1)

        # Expand h to (1, ny, nx, 1)
        h = h[None, :, :, None]             # (1, ny, nx, 1)

        # Expand zeta to (1, ny, nx, nt)
        zeta = zeta[None, :, :, :]          # (1, ny, nx, nt)

        # Compute z0: (nz, ny, nx, nt)
        z0 = (hc * sigma + C * h) / (hc + h)
        z = zeta + (zeta + h) * z0
        return z  # shape: (nz, ny, nx, nt)
    
    @staticmethod
    def get_grid_dims(grid):
        eta_rho, xi_rho = grid.lat_rho.shape
        eta_v, _ = grid.lat_v.shape
        _, xi_u = grid.lat_u.shape
        s_rho = len(grid.s_rho)
        s_w = len(grid.s_w)
        return eta_rho, xi_rho, eta_v, xi_u, s_rho, s_w
    
    @staticmethod
    def create_staggered_grids(lon_rho_grid, lat_rho_grid):
        """
        Create u, v, and psi grids from rho grid.
        
        Args:
            lon_rho_grid: 2D array of longitude at rho points
            lat_rho_grid: 2D array of latitude at rho points
            
        Returns:
            Dictionary with lon_u, lat_u, lon_v, lat_v, lon_psi, lat_psi
        """
        # u-grid: points are halfway between rho points in the x-direction (longitude)
        lon_u = 0.5 * (lon_rho_grid[:, :-1] + lon_rho_grid[:, 1:])
        lat_u = 0.5 * (lat_rho_grid[:, :-1] + lat_rho_grid[:, 1:])

        # v-grid: points are halfway between rho points in the y-direction (latitude)
        lon_v = 0.5 * (lon_rho_grid[:-1, :] + lon_rho_grid[1:, :])
        lat_v = 0.5 * (lat_rho_grid[:-1, :] + lat_rho_grid[1:, :])

        # psi grid 
        lon_psi = 0.25 * (lon_rho_grid[:-1, :-1] + lon_rho_grid[1:, :-1] + 
                          lon_rho_grid[:-1, 1:] + lon_rho_grid[1:, 1:])
        lat_psi = 0.25 * (lat_rho_grid[:-1, :-1] + lat_rho_grid[1:, :-1] + 
                          lat_rho_grid[:-1, 1:] + lat_rho_grid[1:, 1:])
        
        return {
            'lon_u': lon_u, 'lat_u': lat_u,
            'lon_v': lon_v, 'lat_v': lat_v,
            'lon_psi': lon_psi, 'lat_psi': lat_psi
        }
    
    @staticmethod
    def compute_grid_metrics(lon_rho_grid, lat_rho_grid):
        """
        Compute grid metrics: pm, pn, dx, dy, f, xl, el.
        
        Args:
            lon_rho_grid: 2D array of longitude at rho points
            lat_rho_grid: 2D array of latitude at rho points
            
        Returns:
            Dictionary with pm, pn, dx_m, dy_m, f, xl, el
        """
        # Earth's radius in meters
        R = 6371000

        # Convert degrees to radians for calculations
        lat_rho_rad = np.deg2rad(lat_rho_grid)
        lon_rho_rad = np.deg2rad(lon_rho_grid)

        # Compute dx (meters) - distance between adjacent points in x (longitude) direction
        dx_m = np.zeros_like(lon_rho_grid)
        dx_m[:, :-1] = R * np.cos(lat_rho_rad[:, :-1]) * (lon_rho_rad[:, 1:] - lon_rho_rad[:, :-1])
        dx_m[:, -1] = dx_m[:, -2]  # pad last column
        pm = 1/dx_m

        # Compute dy (meters) - distance between adjacent points in y (latitude) direction
        dy_m = np.zeros_like(lat_rho_grid)
        dy_m[:-1, :] = R * (lat_rho_rad[1:, :] - lat_rho_rad[:-1, :])
        dy_m[-1, :] = dy_m[-2, :]  # pad last row
        pn = 1/dy_m

        # compute xl and el 
        xl = np.sum(dx_m[0, :])
        el = np.sum(dy_m[:, 0])

        # compute f 
        f = 2 * 7.2921159e-5 * np.sin(lat_rho_rad)  
        
        return {
            'pm': pm, 'pn': pn,
            'dx_m': dx_m, 'dy_m': dy_m,
            'f': f, 'xl': xl, 'el': el
        }
    
    @staticmethod
    def interpolate_bathymetry(bathy, lon, lat, lon_rho_grid, lat_rho_grid, 
                              smooth_sigma=None, use_log_smoothing=True):
        """
        Interpolate bathymetry to new grid with optional smoothing.
        
        Args:
            bathy: Source bathymetry array (negative depths)
            lon: Source longitude 1D array
            lat: Source latitude 1D array
            lon_rho_grid: Target longitude 2D grid
            lat_rho_grid: Target latitude 2D grid
            smooth_sigma: Smoothing strength (None for no smoothing)
            use_log_smoothing: If True, smooth log of bathymetry
            
        Returns:
            Interpolated bathymetry on new grid
        """
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter
        
        bathy_interp = bathy.copy()
        
        if smooth_sigma is not None:
            # Smooth the bathymetry
            bathy_masked = np.ma.masked_where(bathy > 0, bathy)
            
            if use_log_smoothing:
                bathy_smooth = gaussian_filter(np.log(np.abs(bathy_masked)), sigma=smooth_sigma)
                bathy_smooth = -np.exp(bathy_smooth)  # Convert back to depth
            else:
                bathy_smooth = gaussian_filter(bathy_masked, sigma=smooth_sigma)
            
            bathy_interp = np.where(bathy_masked.mask, bathy, bathy_smooth)
        
        # Interpolate bathymetry to new grid 
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        values = bathy_interp.flatten()
        h = griddata(
            (lon_grid.flatten(), lat_grid.flatten()), values,
            (lon_rho_grid.flatten(), lat_rho_grid.flatten()), method='linear'
        )
        h = h.reshape(lon_rho_grid.shape)
        
        return h
    
    @staticmethod
    def create_masks(h, hmin, mask_rho=None):
        """
        Create masks for rho, u, v, and psi grids.
        
        Args:
            h: Bathymetry array (negative depths)
            hmin: Minimum depth threshold
            mask_rho: Optional pre-defined rho mask (for custom modifications)
            
        Returns:
            Dictionary with mask_rho, mask_u, mask_v, mask_psi
        """
        if mask_rho is None:
            # Create mask (1 for ocean, 0 for land)
            mask_rho = np.where(h < -hmin, 1, 0)
        
        # Create mask for u and v grids
        mask_u = np.minimum(mask_rho[:, :-1], mask_rho[:, 1:])
        mask_v = np.minimum(mask_rho[:-1, :], mask_rho[1:, :])

        # Psi points are at cell corners - a psi point is wet only if 
        # all 4 surrounding rho points are wet
        mask_psi = (mask_rho[:-1, :-1] *    # SW rho point
                    mask_rho[1:, :-1] *     # NW rho point  
                    mask_rho[:-1, 1:] *     # SE rho point
                    mask_rho[1:, 1:])       # NE rho point
        
        return {
            'mask_rho': mask_rho,
            'mask_u': mask_u,
            'mask_v': mask_v,
            'mask_psi': mask_psi
        }
    
    @staticmethod
    def iterative_smoothing(h, rx0_thresh=0.2, max_iter=10, sigma=6, buffer_size=5):
        """
        Perform iterative localized smoothing for steep regions.
        
        Args:
            h: Bathymetry array (negative depths)
            rx0_thresh: rx0 threshold for smoothing
            max_iter: Maximum number of iterations
            sigma: Gaussian smoothing strength
            buffer_size: Number of grid cells for buffer around steep regions
            
        Returns:
            Smoothed bathymetry array
        """
        from scipy.ndimage import gaussian_filter, binary_dilation
        
        h_iter = h.copy()
        for i in range(max_iter):
            rx0_x, rx0_y = grid_tools.compute_rx0(np.abs(h_iter))
            # Pad rx0_x with an extra row in the y-direction
            rx0_x = np.pad(rx0_x, ((0, 0), (0, 1)), mode='edge')
            # Pad rx0_y with an extra column in the x-direction
            rx0_y = np.pad(rx0_y, ((0, 1), (0, 0)), mode='edge')
            max_rx0 = max(np.max(rx0_x), np.max(rx0_y))
            print(f"Iteration {i+1}: max(rx0) = {max_rx0:.4f}")
            
            if max_rx0 < rx0_thresh:
                print(f"Smoothing converged: max(rx0) < {rx0_thresh} after {i+1} iterations.")
                break
            
            steep_mask = (rx0_x > rx0_thresh) | (rx0_y > rx0_thresh)
            steep_mask_buffered = binary_dilation(steep_mask, iterations=buffer_size)
            
            # Smooth the log of the bathymetry for better handling of steep gradients
            h_log = np.log(np.abs(h_iter))
            h_log_gauss = gaussian_filter(h_log, sigma=sigma)
            h_gauss = -np.exp(h_log_gauss)  # Convert back to negative depths
            
            # Create a smooth blending mask using a Gaussian filter on the binary mask
            blend_mask = gaussian_filter(steep_mask_buffered.astype(float), sigma=2)
            blend_mask = np.clip(blend_mask, 0, 1)  # Ensure values are in [0, 1]
            
            # Blend the original and smoothed bathymetry
            h_iter = h_iter * (1 - blend_mask) + h_gauss * blend_mask
            print(f"Iteration {i+1}: {np.sum(steep_mask_buffered)} grid points blended.")
        
        return h_iter
    
    @staticmethod
    def create_roms_grid_dataset(lon_rho_grid, lat_rho_grid, h, masks, staggered_grids,
                                metrics, sigma_params, global_attrs):
        """
        Create a complete ROMS grid xarray Dataset.
        
        Args:
            lon_rho_grid: 2D longitude at rho points
            lat_rho_grid: 2D latitude at rho points
            h: Bathymetry array (positive depths for ROMS)
            masks: Dictionary with mask_rho, mask_u, mask_v, mask_psi
            staggered_grids: Dictionary with lon_u, lat_u, lon_v, lat_v, lon_psi, lat_psi
            metrics: Dictionary with pm, pn, f, xl, el
            sigma_params: Dictionary with sigma_r, sigma_w, Cs_r, Cs_w, N
            global_attrs: Dictionary of global attributes
            
        Returns:
            xarray Dataset ready to save as ROMS grid file
        """
        import xarray as xr
        
        ny, nx = lon_rho_grid.shape
        N = sigma_params['N']
        
        ds_out = xr.Dataset(
            {
                "lat_rho": (("eta_rho", "xi_rho"), lat_rho_grid, {
                    "_FillValue": np.nan,
                    "long_name": "latitude of rho-points",
                    "units": "degrees North"
                }),
                "lon_rho": (("eta_rho", "xi_rho"), lon_rho_grid, {
                    "_FillValue": np.nan,
                    "long_name": "longitude of rho-points",
                    "units": "degrees East"
                }),
                "lat_u": (("eta_rho", "xi_u"), staggered_grids['lat_u'], {
                    "_FillValue": np.nan,
                    "long_name": "latitude of u-points",
                    "units": "degrees North"
                }),
                "lon_u": (("eta_rho", "xi_u"), staggered_grids['lon_u'], {
                    "_FillValue": np.nan,
                    "long_name": "longitude of u-points",
                    "units": "degrees East"
                }),
                "lat_v": (("eta_v", "xi_rho"), staggered_grids['lat_v'], {
                    "_FillValue": np.nan,
                    "long_name": "latitude of v-points",
                    "units": "degrees North"
                }),
                "lon_v": (("eta_v", "xi_rho"), staggered_grids['lon_v'], {
                    "_FillValue": np.nan,
                    "long_name": "longitude of v-points",
                    "units": "degrees East"
                }),
                "y_rho": (("eta_rho", "xi_rho"), lat_rho_grid, {
                    "_FillValue": np.nan,
                    "long_name": "latitude of rho-points",
                    "units": "degrees North"
                }),
                "lat_psi": (("eta_psi", "xi_psi"), staggered_grids['lat_psi'], {
                    "_FillValue": np.nan,
                    "long_name": "latitude of psi-points",
                    "units": "degrees North"
                }),
                "lon_psi": (("eta_psi", "xi_psi"), staggered_grids['lon_psi'], {
                    "_FillValue": np.nan,
                    "long_name": "longitude of psi-points",
                    "units": "degrees East"
                }),        
                "x_rho": (("eta_rho", "xi_rho"), lon_rho_grid, {
                    "_FillValue": np.nan,
                    "long_name": "longitude of rho-points",
                    "units": "degrees East"
                }),        
                "y_u": (("eta_rho", "xi_u"), staggered_grids['lat_u'], {
                    "_FillValue": np.nan,
                    "long_name": "latitude of u-points",
                    "units": "degrees North"
                }),
                "x_u": (("eta_rho", "xi_u"), staggered_grids['lon_u'], {
                    "_FillValue": np.nan,
                    "long_name": "longitude of u-points",
                    "units": "degrees East"
                }),
                "y_v": (("eta_v", "xi_rho"), staggered_grids['lat_v'], {
                    "_FillValue": np.nan,
                    "long_name": "latitude of v-points",
                    "units": "degrees North"
                }),
                "x_v": (("eta_v", "xi_rho"), staggered_grids['lon_v'], {
                    "_FillValue": np.nan,
                    "long_name": "longitude of v-points",
                    "units": "degrees East"
                }),  
                "y_psi": (("eta_psi", "xi_psi"), staggered_grids['lat_psi'], {
                    "_FillValue": np.nan,
                    "long_name": "latitude of psi-points",
                    "units": "degrees North"
                }),
                "x_psi": (("eta_psi", "xi_psi"), staggered_grids['lon_psi'], {
                    "_FillValue": np.nan,
                    "long_name": "longitude of psi-points",
                    "units": "degrees East"
                }),                      
                "angle": (("eta_rho", "xi_rho"), np.zeros_like(lat_rho_grid), {
                    "_FillValue": np.nan,
                    "long_name": "Angle between xi axis and east",
                    "units": "radians",
                    "coordinates": "lat_rho lon_rho"
                }),
                "f": (("eta_rho", "xi_rho"), metrics['f'], {
                    "_FillValue": np.nan,
                    "long_name": "Coriolis parameter at rho-points",
                    "units": "second-1",
                    "coordinates": "lat_rho lon_rho"
                }),
                "pm": (("eta_rho", "xi_rho"), metrics['pm'], {
                    "_FillValue": np.nan,
                    "long_name": "Curvilinear coordinate metric in xi-direction",
                    "units": "meter-1",
                    "coordinates": "lat_rho lon_rho"
                }),
                "pn": (("eta_rho", "xi_rho"), metrics['pn'], {
                    "_FillValue": np.nan,
                    "long_name": "Curvilinear coordinate metric in eta-direction",
                    "units": "meter-1",
                    "coordinates": "lat_rho lon_rho"
                }),
                "spherical": ((), "T", {
                    "Long_name": "Grid type logical switch",
                    "option_T": "spherical"
                }),
                "mask_rho": (("eta_rho", "xi_rho"), masks['mask_rho'], {
                    "long_name": "Mask at rho-points",
                    "units": "land/water (0/1)",
                    "coordinates": "lat_rho lon_rho"
                }),
                "mask_u": (("eta_rho", "xi_u"), masks['mask_u'], {
                    "long_name": "Mask at u-points",
                    "units": "land/water (0/1)",
                    "coordinates": "lat_u lon_u"
                }),
                "mask_v": (("eta_v", "xi_rho"), masks['mask_v'], {
                    "long_name": "Mask at v-points",
                    "units": "land/water (0/1)",
                    "coordinates": "lat_v lon_v"
                }),
                "mask_psi": (("eta_psi", "xi_psi"), masks['mask_psi'], {
                    "long_name": "Mask at psi-points",
                    "units": "land/water (0/1)",
                    "coordinates": "lat_psi lon_psi"
                }),        
                "h": (("eta_rho", "xi_rho"), h, {
                    "_FillValue": np.nan,
                    "long_name": "Bathymetry at rho-points",
                    "units": "meter",
                    "coordinates": "lat_rho lon_rho"
                }),
                "sigma_r": (("s_rho",), sigma_params['sigma_r'], {
                    "_FillValue": np.nan,
                    "long_name": "Fractional vertical stretching coordinate at rho-points",
                    "units": "nondimensional"
                }),
                "Cs_r": (("s_rho",), sigma_params['Cs_r'], {
                    "_FillValue": np.nan,
                    "long_name": "Vertical stretching function at rho-points",
                    "units": "nondimensional"
                }),
                "sigma_w": (("s_w",), sigma_params['sigma_w'], {
                    "_FillValue": np.nan,
                    "long_name": "Fractional vertical stretching coordinate at w-points",
                    "units": "nondimensional"
                }),
                "Cs_w": (("s_w",), sigma_params['Cs_w'], {
                    "_FillValue": np.nan,
                    "long_name": "Vertical stretching function at w-points",
                    "units": "nondimensional"
                }),
                "xl": ((), metrics['xl'], {
                    "long_name": "domain length in the XI-direction",
                    "units": "meter"
                }),
                "el": ((), metrics['el'], {
                    "long_name": "domain length in the ETA-direction",
                    "units": "meter"
                }),
            },
            coords={
                "eta_rho": np.arange(ny),
                "xi_rho": np.arange(nx),
                "xi_u": np.arange(nx - 1),
                "eta_v": np.arange(ny - 1),
                "eta_psi": np.arange(ny-1),
                "xi_psi": np.arange(nx -1),
                "s_rho": np.arange(N),
                "s_w": np.arange(N + 1)
            }
        )

        # Assign global attributes
        ds_out.attrs = global_attrs
        
        return ds_out