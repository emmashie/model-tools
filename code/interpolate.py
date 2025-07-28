import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

class interp_tools:
    """
    Interpolation tools for 2D and 3D fields using scipy's griddata.
    Now supports masking: only interpolate at valid (ocean) points.
    """
    def __init__(self):
        pass

    @staticmethod
    def interp2d(src_field, src_lon, src_lat, tgt_lon, tgt_lat, method='nearest'):
        """
        Interpolate a 2D field from source grid to target grid using scipy's griddata.
        Fills any NaNs (outside convex hull) with nearest-neighbor interpolation.

        Parameters
        ----------
        src_field : 2D array (ny_src, nx_src)
            Source field to interpolate
        src_lon : 1D or 2D array
            Source longitudes (same shape as src_field or 1D)
        src_lat : 1D or 2D array
            Source latitudes (same shape as src_field or 1D)
        tgt_lon : 2D array (ny_tgt, nx_tgt)
            Target grid longitudes
        tgt_lat : 2D array (ny_tgt, nx_tgt)
            Target grid latitudes
        method : str
            Interpolation method ('linear', 'nearest', 'cubic')

        Returns
        -------
        field_interp : 2D array (ny_tgt, nx_tgt)
            Interpolated field on target grid
        """
        points = np.column_stack((src_lon.ravel(), src_lat.ravel()))
        values = src_field.ravel()
        # Main interpolation
        field_interp = griddata(points, values, (tgt_lon, tgt_lat), method=method)
        # Fallback for NaNs: nearest neighbor
        if method != 'nearest' and np.any(np.isnan(field_interp)):
            field_nearest = griddata(points, values, (tgt_lon, tgt_lat), method='nearest')
            field_interp = np.where(np.isnan(field_interp), field_nearest, field_interp)
        return field_interp

    @staticmethod
    def interp3d(src_field, src_lon, src_lat, src_z, tgt_lon, tgt_lat, tgt_z, method='nearest'):
        """
        Interpolate a 3D field from source grid to target grid.
        Interpolates horizontally at each vertical level, then vertically at each (y,x) point.

        Parameters
        ----------
        src_field : 3D array (nz_src, ny_src, nx_src)
            Source field to interpolate
        src_lon : 1D or 2D array
            Source longitudes (same shape as src_field[0] or 1D)
        src_lat : 1D or 2D array
            Source latitudes (same shape as src_field[0] or 1D)
        src_z : 1D array (nz_src,)
            Source vertical levels
        tgt_lon : 2D array (ny_tgt, nx_tgt)
            Target grid longitudes
        tgt_lat : 2D array (ny_tgt, nx_tgt)
            Target grid latitudes
        tgt_z : 1D array (nz_tgt,) or 3D array (ny_tgt, nx_tgt, nz_tgt)
            Target vertical levels (can be different at each horizontal point)
        method : str
            Interpolation method ('linear', 'nearest', 'cubic')

        Returns
        -------
        field_interp : 3D array (nz_tgt, ny_tgt, nx_tgt)
            Interpolated field on target grid
        """
        ny_tgt, nx_tgt = tgt_lon.shape
        nz_src = src_field.shape[0]
        # Horizontally interpolate each source vertical level
        field_horiz = np.empty((nz_src, ny_tgt, nx_tgt))
        for k in range(nz_src):
            field_horiz[k] = interp_tools.interp2d(
                src_field[k], src_lon, src_lat, tgt_lon, tgt_lat, method=method
            )
        # Vertically interpolate at each (y, x) using interp1d with extrapolation
        if tgt_z.ndim == 1:
            nz_tgt = len(tgt_z)
            field_interp = np.empty((nz_tgt, ny_tgt, nx_tgt))
            for j in range(ny_tgt):
                for i in range(nx_tgt):
                    tmp_ind = np.isfinite(field_horiz[:, j, i])
                    if tmp_ind.any():
                        f = interp1d(
                            src_z[tmp_ind], 
                            field_horiz[:, j, i][tmp_ind], 
                            kind='linear', 
                            bounds_error=False, 
                            fill_value='extrapolate', 
                            assume_sorted=True
                        )
                        field_interp[:, j, i] = f(tgt_z)
                    else:
                        field_interp[:, j, i] = np.nan
        elif tgt_z.ndim == 3:
            nz_tgt = tgt_z.shape[2]
            field_interp = np.empty((nz_tgt, ny_tgt, nx_tgt))
            for j in range(ny_tgt):
                for i in range(nx_tgt):
                    tmp_ind = np.isfinite(field_horiz[:, j, i])
                    if tmp_ind.any():
                        f = interp1d(
                            src_z[tmp_ind],
                            field_horiz[:, j, i][tmp_ind],
                            kind='linear',
                            bounds_error=False,
                            fill_value='extrapolate',
                            assume_sorted=True
                        )
                        field_interp[:, j, i] = f(tgt_z[j, i, :])
                    else:
                        field_interp[:, j, i] = np.nan
        else:
            raise ValueError("tgt_z must be 1D or 3D array")
        return field_interp
