## netcdf_handler.py

import netCDF4 as nc
import os

class NetCDFHandler:
    """Class responsible for handling netCDF file creation."""

    def __init__(self, default_filename: str = "output.nc"):
        """Initializes the NetCDFHandler with a default filename.

        Args:
            default_filename (str): The default filename for the netCDF file. Defaults to 'output.nc'.
        """
        self.default_filename = default_filename

    def create_netcdf_file(self, data_path: str, filename: str = None) -> None:
        """Creates a netCDF file from the downloaded data.

        Args:
            data_path (str): Path to the downloaded data file.
            filename (str, optional): Filename for the netCDF file. Defaults to None, which uses the default filename.
        """
        if filename is None:
            filename = self.default_filename

        try:
            # Open the downloaded netCDF file
            with nc.Dataset(data_path, 'r') as src:
                # Create a new netCDF file
                with nc.Dataset(filename, 'w', format='NETCDF4') as dst:
                    # Copy dimensions from source to destination
                    for name, dimension in src.dimensions.items():
                        dst.createDimension(
                            name, (len(dimension) if not dimension.isunlimited() else None))

                    # Copy variables from source to destination
                    for name, variable in src.variables.items():
                        dst_var = dst.createVariable(
                            name, variable.datatype, variable.dimensions)
                        dst_var.setncatts({k: variable.getncattr(k)
                                           for k in variable.ncattrs()})
                        dst_var[:] = variable[:]

            print(f"NetCDF file successfully created: {filename}")

        except OSError as e:
            print(f"An error occurred while creating netCDF file: {e}")
