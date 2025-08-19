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
    def compute_rx1(h):
        """
        Compute rx1 (Haney parameter) for ROMS grid stiffness.
        """
        # rx1 in x-direction (between i and i+1)
        rx1_x = np.abs(np.diff(h, axis=1)) / (h[:, :-1] + h[:, 1:])
        
        # rx1 in y-direction (between j and j+1) 
        rx1_y = np.abs(np.diff(h, axis=0)) / (h[:-1, :] + h[1:, :])
        
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