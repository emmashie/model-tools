## main.py

import sys
from download import Downloader
from netcdf_handler import NetCDFHandler

class Main:
    """Main class to orchestrate the downloading and processing of SRTM15+ bathymetry data."""

    def __init__(self):
        """Initializes the Main class with default latitude and longitude ranges."""
        self.lat_range = (-90.0, 90.0)  # Default latitude range
        self.lon_range = (-180.0, 180.0)  # Default longitude range
        self.output_filename = "output.nc"  # Default output filename

    def main(self) -> None:
        """Main function to execute the data download and netCDF file creation."""
        # Parse command-line arguments for latitude and longitude ranges
        self.parse_arguments()

        # Initialize the Downloader and NetCDFHandler
        downloader = Downloader()
        netcdf_handler = NetCDFHandler()

        # Download the data
        data_path = downloader.download_data(self.lat_range, self.lon_range)

        if data_path:
            # Create the netCDF file
            netcdf_handler.create_netcdf_file(data_path, self.output_filename)
        else:
            print("Failed to download data. Exiting.")

    def parse_arguments(self) -> None:
        """Parses command-line arguments to set latitude and longitude ranges."""
        if len(sys.argv) >= 5:
            try:
                self.lat_range = (float(sys.argv[1]), float(sys.argv[2]))
                self.lon_range = (float(sys.argv[3]), float(sys.argv[4]))
                if len(sys.argv) == 6:
                    self.output_filename = sys.argv[5]
            except ValueError:
                print("Invalid arguments. Using default latitude and longitude ranges.")
        else:
            print("Insufficient arguments. Using default latitude and longitude ranges.")

if __name__ == "__main__":
    main_instance = Main()
    main_instance.main()
