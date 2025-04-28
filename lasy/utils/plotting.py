import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c, epsilon_0
from copy import deepcopy

from .laser_utils import get_duration, get_w0

# default time and space units (value and label)
units_def = {'t': {'value': 1e-15, 'label': 'fs'},
             'x': {'value': 1e-6, 'label': r'\mu m'}}             


def show_laser(grid, dim, show_intensity=False, t_shift = 0, udict={}, **kw):
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

    show_intensity : bool, default: False
        if False the laser amplitude is plotted
        if True then the intensity of the laser is plotted along with lineouts
        and a measure of the pulse duration and spot size

    t_shift : float, default: 0
        Shift the temporal axis by `t_shift` seconds.
        It also can be a string with `"left"`, `"right"` or `"center"`,
        to shift the temporal axis to the left, right or center of the time axis.

    udict : dict, default: {}
        Dictionary with the information of the unit scales of the axes,
        e.g. ``{'t': {'value': 1e-15, 'label': 'fs'}, 'x': {'value': 1e-6, 'label': r'\mu m'}}``
        Allows the user to override the default unit scales.

    **kw : additional arguments to be passed to matplotlib's imshow command
    """
    if "cmap" in kw.keys():
        pass
    else:
        kw["cmap"] = "Reds"  # Set default colormap

    if show_intensity:
        F = epsilon_0 * c / 2 * np.abs(grid.get_temporal_field()) ** 2 / 1e4
        cbar_label = r"I (W/cm$^2$)"
    else:
        F = np.abs(grid.get_temporal_field())
        cbar_label = r"$|E_{envelope}|$ (V/m)"

    # Set default unit scales for the axes
    units = deepcopy(units_def)

    # Calculate spatial scales for the axes
    if grid.hi[0] > 1:
        # scale is meters
        units['x']['value'] = 1
        units['x']['label'] = 'm'
    elif grid.hi[0] > 1e-3:
        # scale is millimeters
        units['x']['value'] = 1e-3
        units['x']['label'] = 'mm'
    else:
        # scale is microns (default)
        pass

    # Calculate temporal scales for the axes
    if grid.hi[-1] - grid.lo[-1] > 1e-9:
        # scale is nanoseconds
        units['t']['value'] = 1e-9
        units['t']['label'] = 'ns'
    elif grid.hi[-1] - grid.lo[-1] > 1e-12:
        # scale is picoseconds
        units['t']['value'] = 1e-12
        units['t']['label'] = 'ps'
    else:
        # scale is femtoseconds (default)
        pass

    # Allows the user to override default units
    for k in udict.keys():
        units[k] = udict[k]

    # Allow the user to shift the temporal axis
    if t_shift == 'left':
        t_shift = grid.lo[-1]
    elif t_shift == 'right':
        t_shift = grid.hi[-1]
    elif t_shift == 'center':
        t_shift = 0.5 * (grid.hi[-1] + grid.lo[-1])
    elif not isinstance(t_shift, float):
        raise ValueError(
            f"Invalid value for t_shift.\n"
            f"It should be one of 'left', 'right', 'center', or a float.\n"
        )

    if dim == "rt":
        # Show field in the plane y=0, above and below axis, with proper sign for each mode
        F_plot = [
            np.concatenate(((-1.0) ** m * F[m, ::-1], F[m]))
            for m in grid.azimuthal_modes
        ]
        F_plot = sum(F_plot)  # Sum all the modes
        extent = [
            (grid.lo[-1] - t_shift) / units['t']['value'],
            (grid.hi[-1] - t_shift) / units['t']['value'],
            -grid.hi[0] / units['x']['value'],
            grid.hi[0] / units['x']['value'],
        ]

    else:
        # In 3D show an image in the xt plane
        i_slice = int(F.shape[1] // 2)
        F_plot = F[:, i_slice, :]
        extent = [
            (grid.lo[-1] - t_shift) / units['t']['value'],
            (grid.hi[-1] - t_shift) / units['t']['value'],
            grid.lo[0] / units['x']['value'],
            grid.hi[0] / units['x']['value'],
        ]

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(F_plot, extent=extent, aspect="auto", origin="lower", **kw)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label)
    ax.set_xlabel(r"t " + r"($%s$)" % units['t']['label'])
    ax.set_ylabel(r"x " + r"($%s$)" % units['x']['label'])

    if t_shift != 0:
        ax.text(
            0.025,
            1.05,
            r"Time shift = %.2e s" % t_shift,
            transform=ax.transAxes,
            fontsize='x-small',
            ha='left',
            va='top',
        )

    if show_intensity:
        # Create projected lineouts along time and space
        temporal_lineout = np.sum(F_plot, axis=0) / np.sum(F_plot, axis=0).max()
        ax.plot(
            (grid.axes[-1] - t_shift) / units['t']['value'],
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
        tau = 2 * get_duration(grid, dim) / units['t']['value']
        ax.text(
            0.95,
            0.95,
            r"Pulse duration = %.2f " % (tau) + r"$%s$" % units['t']['label'],
            transform=ax.transAxes,
            fontsize='small',
            ha='right',
            va='top',
        )

        # Get the spot size
        w0 = get_w0(grid, dim) / units['x']['value']
        ax.text(
            0.95,
            0.9,
            r"Spot size = %.2f " % (w0) + r"$%s$" % units['x']['label'],
            transform=ax.transAxes,
            fontsize='small',
            ha='right',
            va='top',
        )
