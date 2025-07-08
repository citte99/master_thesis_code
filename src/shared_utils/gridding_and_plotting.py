import torch
from matplotlib import pyplot as plt
import numpy as np

from .units import _arcsec_to_rad

def _grid_lens(grid_size_arcsec, image_res, device, dtype=torch.float32):
    grid_width_arcsec = _arcsec_to_rad(grid_size_arcsec)
    half_width_image = grid_width_arcsec / 2.0
    xx = torch.linspace(-half_width_image, half_width_image, steps=image_res, dtype=dtype, device=device)
    yy = torch.linspace(-half_width_image, half_width_image, steps=image_res, dtype=dtype, device=device)
    # Using indexing='xy' to maintain consistency with your original code.
    grid = torch.stack(torch.meshgrid(xx, yy, indexing='xy'), dim=2)
    return grid


def _plot2D_on_grid(plotted_data, grid, ax=None, show_colorbar=True):
    """
    Plots a 2D image on the given grid.
    If ax is provided, the image is plotted on that axis.
    """
    if ax is None:
        ax = plt.gca()
    
    # Flip the y-axis to match xy indexing
    plotted_data = torch.flipud(plotted_data)
    plotted_data = plotted_data.cpu().numpy()
    grid = grid.cpu().numpy()
    
    im = ax.imshow(plotted_data,
                   extent=(grid[0, 0, 0],
                           grid[0, -1, 0],
                           grid[0, 0, 1],
                           grid[-1, 0, 1]))
    if show_colorbar:
        # Use the figure associated with the axis to add a colorbar
        ax.figure.colorbar(im, ax=ax)