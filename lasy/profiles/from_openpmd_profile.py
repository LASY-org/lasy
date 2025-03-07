import numpy as np
import openpmd_api as io
from scipy.constants import c

from lasy.utils.laser_utils import (
    create_grid,
    vector_potential_to_field,
)

from .from_array_profile import FromArrayProfile


class FromOpenPMDProfile(FromArrayProfile):
    r"""
    Profile defined from an openPMD file.

    Parameters
    ----------
    path : string
        Path to the openPMD file containing the laser field or envelope.

    iteration : int
        Iteration at which the argument is read.

    field : string
        Name of the field containing the laser pulse
    """

    def __init__(
        self,
        path,
        iteration,
        field,
    ):
        # Read the data
        series = io.Series(path, io.Access.read_only)
        i = series.iterations[iteration]
        m = i.meshes[field]
        array = m[io.Mesh_Record_Component.SCALAR].load_chunk()
        series.flush()

        # Extract the required parameters
        omg0 = m.get_attribute("angularFrequency")
        wavelength = 2 * np.pi * c / omg0
        grid_offset = m.get_attribute("gridGlobalOffset")
        grid_spacing = m.get_attribute("gridSpacing")

        try:
            pol = m.polarization
        except AttributeError:
            print('Polarization not found. Defaulting to (1, 0)')
            pol = (1, 0)

        # Define parameters to create a profile
        if len(m.axis_labels) == 2:  # 'rt'
            if m.axis_labels[0] == "r":
                ir = 0
                it = 1
            else:
                it = 0
                ir = 1

            n_t = array.shape[it + 1]
            n_r = array.shape[ir + 1]
            t = np.linspace(
                grid_offset[it], grid_offset[it] + (n_t - 1) * grid_spacing[it], n_t
            )
            r = np.linspace(
                grid_offset[ir], grid_offset[ir] + (n_r - 1) * grid_spacing[ir], n_r
            )
    
            dim = "rt"
            axes = {"r": r, "t": t}
            axes_order = ["r", "t"]
            if m.axis_labels[1] == "r":
                array = np.swapaxes(array, 1, 2)

        elif len(m.axis_labels) == 3:  # 'xyt'
            if m.axis_labels[0] == "x":
                ix = 0
                iy = 1
                it = 2
            else:
                ix = 2
                iy = 1
                it = 0

            n_x = array.shape[ix]
            n_y = array.shape[iy]
            n_t = array.shape[it]
            x = np.linspace(
                grid_offset[ix], grid_offset[ix] + (n_x - 1) * grid_spacing[ix], n_x
            )
            y = np.linspace(
                grid_offset[iy], grid_offset[iy] + (n_y - 1) * grid_spacing[iy], n_y
            )
            t = np.linspace(
                grid_offset[it], grid_offset[it] + (n_t - 1) * grid_spacing[it], n_t
            )
            dim = "xyt"
            axes = {"x": x, "y": y, "t": t}
            axes_order = ["x", "y", "t"]
            if m.axis_labels[2] == "x":
                array = np.swapaxes(array, 0, 2)
        else:
            print(
                "Error: The dimension of the field is not supported. The valid dimensions are 'rt' and 'xyt'."
            )
            return None

        # If longitudinal dimension was `z`, change it to `t`
        if "z" in m.axis_labels:
            axes["t"] = (axes["t"] - axes["t"][0]) / c
            array = np.flip(array, axis=-1)

        # If the field is stored as vector potential,
        # convert it to electric field
        vector_to_field = False
        try:
            if m.envelopeField == "normalized_vector_potential":
                vector_to_field = True
        except AttributeError:
            if field == "a":
                vector_to_field = True
        if vector_to_field:
            grid = create_grid(array, axes, dim)
            array = vector_potential_to_field(grid, omg0)

        super().__init__(
            wavelength=wavelength,
            pol=pol,
            array=array,
            dim=dim,
            axes=axes,
            axes_order=axes_order,
        )
