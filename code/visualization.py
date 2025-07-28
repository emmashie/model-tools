## visualization.py

import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    def __init__(self, figsize: tuple = (10, 6)):
        """
        Initializes the Visualization class with a default figure size.

        Args:
            figsize (tuple): Size of the figure for plotting. Default is (10, 6).
        """
        self.figsize = figsize

    def plot_data(self, original_data: np.ndarray, smoothed_data: np.ndarray) -> None:
        """
        Plots the original and smoothed bathymetry data.

        Args:
            original_data (np.ndarray): The original bathymetry data.
            smoothed_data (np.ndarray): The smoothed bathymetry data.
        """
        if original_data is None or smoothed_data is None:
            raise ValueError("Original and smoothed data must be provided for plotting.")

        try:
            plt.figure(figsize=self.figsize)
            plt.subplot(2, 1, 1)
            plt.title("Original Bathymetry Data")
            plt.imshow(original_data, cmap='viridis', aspect='auto')
            plt.colorbar(label='Depth')

            plt.subplot(2, 1, 2)
            plt.title("Smoothed Bathymetry Data")
            plt.imshow(smoothed_data, cmap='viridis', aspect='auto')
            plt.colorbar(label='Depth')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            raise RuntimeError(f"Failed to plot data: {e}")