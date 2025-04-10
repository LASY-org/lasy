import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c, epsilon_0

from .laser_utils import get_duration, get_w0


def show_laser(grid, dim, show_intensity, **kw):
    """
    Show a 2D image of the laser represented on the grid.

    Parameters
    ----------
    grid : Grid
        The Grid object to be plotted

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    show_intensity : bool
        if False the laser amplitude is plotted
        if True then the intensity of the laser is plotted along with lineouts
        and a measure of the pulse duration and spot size

    **kw : additional arguments to be passed to matplotlib's imshow command
    """
    if show_intensity:
        F = epsilon_0 * c / 2 * np.abs(grid.get_temporal_field()) ** 2 / 1e4
        cbar_label = r"I (W/cm$^2$)"
    else:
        F = np.abs(grid.get_temporal_field())
        cbar_label = r"$|E_{envelope}|$ (V/m)"

    # Calculate spatial scales for the axes
    if grid.hi[0] > 1:
        # scale is meters
        spatial_scale = 1
        spatial_unit = r"(m)"
    elif grid.hi[0] > 1e-3:
        # scale is millimeters
        spatial_scale = 1e-3
        spatial_unit = r"(mm)"
    else:
        # scale is microns
        spatial_scale = 1e-6
        spatial_unit = r"($\mu m$)"

    # Calculate temporal scales for the axes
    if grid.hi[-1] - grid.lo[-1] > 1e-9:
        # scale is nanoseconds
        temporal_scale = 1e-9
        temporal_unit = r"(ns)"
    elif grid.hi[-1] - grid.lo[-1] > 1e-12:
        # scale is picoseconds
        temporal_scale = 1e-12
        temporal_unit = r"(ps)"
    else:
        # scale is femtoseconds
        temporal_scale = 1e-15
        temporal_unit = r"(fs)"

    if dim == "rt":
        # Show field in the plane y=0, above and below axis, with proper sign for each mode
        F_plot = [
            np.concatenate(((-1.0) ** m * F[m, ::-1], F[m]))
            for m in grid.azimuthal_modes
        ]
        F_plot = sum(F_plot)  # Sum all the modes
        extent = [
            grid.lo[-1] / temporal_scale,
            grid.hi[-1] / temporal_scale,
            -grid.hi[0] / spatial_scale,
            grid.hi[0] / spatial_scale,
        ]

    else:
        # In 3D show an image in the xt plane
        i_slice = int(F.shape[1] // 2)
        F_plot = F[:, i_slice, :]
        extent = [
            grid.lo[-1] / temporal_scale,
            grid.hi[-1] / temporal_scale,
            grid.lo[0] / spatial_scale,
            grid.hi[0] / spatial_scale,
        ]

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(
        F_plot, extent=extent, cmap="Reds", aspect="auto", origin="lower", **kw
    )
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label)
    ax.set_xlabel(r"t " + temporal_unit)
    ax.set_ylabel(r"x " + spatial_unit)

    if show_intensity:
        # Create projected lineouts along time and space
        temporal_lineout = np.sum(F_plot, axis=0) / np.sum(F_plot, axis=0).max()
        ax.plot(
            grid.axes[-1] / temporal_scale,
            0.15 * temporal_lineout * (extent[3] - extent[2]) + extent[2],
            c=(0.3, 0.3, 0.3),
        )

        spatial_lineout = np.sum(F_plot, axis=1) / np.sum(F_plot, axis=1).max()
        ax.plot(
            0.15 * spatial_lineout * (extent[1] - extent[0]) + extent[0],
            np.linspace(extent[2], extent[3], F_plot.shape[0]),
            c=(0.3, 0.3, 0.3),
        )

        # Get the pulse duration
        tau = 2 * get_duration(grid, dim) / temporal_scale
        ax.text(
            0.55,
            0.95,
            r"Pulse Duration   = %.1f " % (tau) + temporal_unit[1:-1],
            transform=ax.transAxes,
        )

        # Get the spot size
        w0 = get_w0(grid, dim) / spatial_scale
        ax.text(
            0.55,
            0.9,
            r"Spot Size           = %.1f " % (w0) + spatial_unit[1:-1],
            transform=ax.transAxes,
        )
