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