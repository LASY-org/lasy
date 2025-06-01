import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c, epsilon_0

from .laser_utils import field_to_vector_potential, get_duration, get_w0


def show_laser(
    grid,
    dim,
    envelope_type="field",
    t_shift=0,
    show_lineout=True,
    show_max=False,
    omega0=None,
    udict={},
    **kw,
):
    r"""
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

    envelope_type : string, default: "field"
        Options are:
        - ``'field'``: Show the envelope of the laser field.
        - ``'intensity'``: Show the intensity of the laser field.
        - ``'vector_potential'``: Show the vector potential of the laser field.

    t_shift : float, default: 0
        Shift the temporal axis by `t_shift` seconds.
        It also can be a string with `"left"`, `"right"` or `"center"`,
        to shift the temporal axis such that the t=0 lies at the left,
        right or center of the x-axis.

    show_lineout : bool, default: True
        Show the lineout of the laser field.

    show_max : bool, default: False
        Print the maximum intensity of the laser field.

    omega0 : scalar
        Angular frequency at which the envelope is defined.
        Needed if `envelope_type == "vector_potential"`.

    udict : dict, default: {}
        Dictionary with the information of the unit scales of the axes,
        e.g. ``{'t': {'value': 1e-15, 'label': 'fs'}, 'x': {'value': 1e-6, 'label': r'\mu m'}}``
        Override the default unit scales.

    **kw : additional arguments to be passed to matplotlib's imshow command
    """
    if "cmap" in kw.keys():
        pass
    else:
        kw["cmap"] = "Reds"  # Set default colormap

    if envelope_type == "intensity":
        F = epsilon_0 * c / 2 * np.abs(grid.get_temporal_field()) ** 2 / 1e4
        cbar_label = r"I (W/cm$^2$)"
    elif envelope_type == "vector_potential":
        assert omega0 is not None, (
            "omega0 must be provided if envelope_type == 'vector_potential'"
        )
        F = np.abs(field_to_vector_potential(grid=grid, omega0=omega0))
        cbar_label = r"$|a|$"
    elif envelope_type == "field":
        F = np.abs(grid.get_temporal_field())
        cbar_label = r"$|E|$ (V/m)"
    else:
        raise ValueError(
            "Invalid value for envelope_type.\n"
            "It should be one of 'field', 'intensity' or 'vector_potential'.\n"
        )

    # Set default unit scales for the axes
    units = {
        "t": {"value": 1e-15, "label": "fs"},
        "x": {"value": 1e-6, "label": r"\mu m"},
    }

    # Calculate spatial scales for the axes
    if grid.hi[0] > 1:
        # scale is meters
        units["x"]["value"] = 1
        units["x"]["label"] = "m"
    elif grid.hi[0] > 1e-3:
        # scale is millimeters
        units["x"]["value"] = 1e-3
        units["x"]["label"] = "mm"

    # Calculate temporal scales for the axes
    if grid.hi[-1] - grid.lo[-1] > 1e-9:
        # scale is nanoseconds
        units["t"]["value"] = 1e-9
        units["t"]["label"] = "ns"
    elif grid.hi[-1] - grid.lo[-1] > 1e-12:
        # scale is picoseconds
        units["t"]["value"] = 1e-12
        units["t"]["label"] = "ps"

    # Override default units
    for k in udict.keys():
        units[k] = udict[k]

    # Shift the temporal axis
    if t_shift == "left":
        t_shift = grid.lo[-1]
    elif t_shift == "right":
        t_shift = grid.hi[-1]
    elif t_shift == "center":
        t_shift = 0.5 * (grid.hi[-1] + grid.lo[-1])
    elif not isinstance(t_shift, (float, int)):
        raise ValueError(
            "Invalid value for t_shift.\n"
            "It should be one of 'left', 'right', 'center', or a number.\n"
        )

    if dim == "rt":
        # Show field in the plane y=0, above and below axis, with proper sign for each mode
        F_plot = [
            np.concatenate(((-1.0) ** m * F[m, ::-1], F[m]))
            for m in grid.azimuthal_modes
        ]
        F_plot = sum(F_plot)  # Sum all the modes
        extent = [
            (grid.lo[-1] - t_shift) / units["t"]["value"],
            (grid.hi[-1] - t_shift) / units["t"]["value"],
            -grid.hi[0] / units["x"]["value"],
            grid.hi[0] / units["x"]["value"],
        ]

    else:
        # In 3D show an image in the xt plane
        i_slice = int(F.shape[1] // 2)
        F_plot = F[:, i_slice, :]
        extent = [
            (grid.lo[-1] - t_shift) / units["t"]["value"],
            (grid.hi[-1] - t_shift) / units["t"]["value"],
            grid.lo[0] / units["x"]["value"],
            grid.hi[0] / units["x"]["value"],
        ]

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.075)
    im = ax.imshow(F_plot, extent=extent, aspect="auto", origin="lower", **kw)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label)
    ax.set_xlabel(r"t " + r"($%s$)" % units["t"]["label"])
    ax.set_ylabel(r"x " + r"($%s$)" % units["x"]["label"])

    position = grid.position
    vpos = 0.98
    ax.text(
        0.025,
        vpos,
        r"Position = %.3f mm" % (position / 1e-3),
        transform=ax.transAxes,
        fontsize="x-small",
        ha="left",
        va="top",
    )

    if t_shift != 0:
        vpos = vpos - 0.05
        ax.text(
            0.025,
            vpos,
            r"Time shift = %.1f fs" % (-t_shift / 1e-15),
            transform=ax.transAxes,
            fontsize="x-small",
            ha="left",
            va="top",
        )

    if show_lineout:
        # Create projected lineouts along time and space
        temporal_lineout = np.sum(F_plot, axis=0) / np.sum(F_plot, axis=0).max()
        ax.plot(
            (grid.axes[-1] - t_shift) / units["t"]["value"],
            0.15 * temporal_lineout * (extent[3] - extent[2]) + extent[2],
            c=(0.3, 0.3, 0.3),
        )

        spatial_lineout = np.sum(F_plot, axis=1) / np.sum(F_plot, axis=1).max()
        ax.plot(
            0.15 * spatial_lineout * (extent[1] - extent[0]) + extent[0],
            np.linspace(extent[2], extent[3], F_plot.shape[0]),
            c=(0.3, 0.3, 0.3),
        )

    field_max = np.max(F_plot)
    vpos = 0.98
    if envelope_type == "intensity":
        field_max_label = r"$I_{max}$ = %.2e $W/cm^2$" % (field_max)

        # Get the pulse duration
        tau = 2 * get_duration(grid, dim) / units["t"]["value"]
        ax.text(
            0.975,
            vpos,
            r"Pulse duration = %.2f " % (tau) + r"$%s$" % units["t"]["label"],
            transform=ax.transAxes,
            fontsize="x-small",
            ha="right",
            va="top",
        )
        vpos = vpos - 0.05

        # Get the spot size
        w0 = get_w0(grid, dim) / units["x"]["value"]
        ax.text(
            0.975,
            vpos,
            r"Spot size = %.2f " % (w0) + r"$%s$" % units["x"]["label"],
            transform=ax.transAxes,
            fontsize="x-small",
            ha="right",
            va="top",
        )
        vpos = vpos - 0.05
    elif envelope_type == "vector_potential":
        field_max_label = r"$|a_{max}|$ = %.3f" % (field_max)
    elif envelope_type == "field":
        field_max_label = r"$|E_{max}|$ = %.2e V/m" % (field_max)
    else:
        field_max_label = r"Max = %.2e" % (field_max)

    if show_max:
        ax.text(
            0.975,
            vpos,
            field_max_label,
            transform=ax.transAxes,
            fontsize="x-small",
            ha="right",
            va="top",
        )
