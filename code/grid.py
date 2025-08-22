import numpy as np

class grid_tools:
    @staticmethod
    def compute_slope_factor(h, dx=1.0, dy=1.0, epsilon=1e-10):
        """
        Compute the slope factor r = |grad(h)|/h for a 2D array h.
        
        Parameters:
        -----------
        h : numpy.ndarray
            2D array representing the field (e.g., height, depth, etc.)
        dx : float, optional
            Grid spacing in x-direction (default: 1.0)
        dy : float, optional
            Grid spacing in y-direction (default: 1.0)
        epsilon : float, optional
            Small value to avoid division by zero (default: 1e-10)
        
        Returns:
        --------
        r : numpy.ndarray
            2D array of slope factors with same shape as input h
        """
        # Compute gradients using central differences
        grad_y, grad_x = np.gradient(h)
        grad_y /= dy  # Scale by grid spacing in y-direction
        grad_x /= dx  # Scale by grid spacing in x-direction
        
        # Compute magnitude of gradient
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Compute slope factor r = |grad(h)|/h
        r = grad_magnitude / (np.abs(h) + epsilon)

        return grad_y, grad_x, grad_magnitude, r

    @staticmethod
    def compute_slope_parameter(bathymetry, return_components=False, mask=None):
        """
        Vectorized version of slope parameter computation for better performance.
        """
        bathymetry = np.asarray(bathymetry, dtype=float)
        ni, nj = bathymetry.shape
        
        # Apply mask if provided
        if mask is not None:
            bathymetry = bathymetry.copy()
            bathymetry[~mask] = np.nan
        
        # Set non-positive depths to NaN
        bathymetry[bathymetry <= 0] = np.nan
        
        # Compute slope parameter in x-direction
        hA_x = bathymetry[:, :-1]  # depths at i,j
        hB_x = bathymetry[:, 1:]   # depths at i,j+1
        
        denominator_x = hA_x + hB_x
        numerator_x = np.abs(hB_x - hA_x)
        
        sp_x = np.full_like(denominator_x, np.nan)
        valid_x = (denominator_x > 0) & ~np.isnan(denominator_x)
        sp_x[valid_x] = numerator_x[valid_x] / denominator_x[valid_x]
        
        # Compute slope parameter in y-direction
        hA_y = bathymetry[:-1, :]  # depths at i,j
        hB_y = bathymetry[1:, :]   # depths at i+1,j
        
        denominator_y = hA_y + hB_y
        numerator_y = np.abs(hB_y - hA_y)
        
        sp_y = np.full_like(denominator_y, np.nan)
        valid_y = (denominator_y > 0) & ~np.isnan(denominator_y)
        sp_y[valid_y] = numerator_y[valid_y] / denominator_y[valid_y]
        
        if return_components:
            # Compute maximum at overlapping points
            sp_x_pad = np.full((ni, nj), np.nan)
            sp_x_pad[:, :-1] = sp_x
            
            sp_y_pad = np.full((ni, nj), np.nan)
            sp_y_pad[:-1, :] = sp_y
            
            # Maximum over 2x2 neighborhoods
            sp_max = np.nanmax(np.stack([
                sp_x_pad[:-1, :-1],  # sp_x at i,j
                sp_x_pad[1:, :-1],   # sp_x at i+1,j
                sp_y_pad[:-1, :-1],  # sp_y at i,j
                sp_y_pad[:-1, 1:]    # sp_y at i,j+1
            ]), axis=0)
            
            return sp_max, sp_x, sp_y
        else:
            # Just return maximum
            sp_x_pad = np.full((ni, nj), np.nan)
            sp_x_pad[:, :-1] = sp_x
            
            sp_y_pad = np.full((ni, nj), np.nan)
            sp_y_pad[:-1, :] = sp_y
            
            sp_max = np.nanmax(np.stack([
                sp_x_pad[:-1, :-1],
                sp_x_pad[1:, :-1],
                sp_y_pad[:-1, :-1],
                sp_y_pad[:-1, 1:]
            ]), axis=0)
            
            return sp_max

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