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
        pol = m.get_attribute("polarization")

        # Define parameters to create a profile
        if len(m.axis_labels) == 2:  # 'rt'
            n_t = array.shape[1]
            n_r = array.shape[2]
            grid_offset = m.get_attribute("gridGlobalOffset")
            t = np.linspace(grid_offset[0], grid_offset[0] + (n_t - 1) * m.grid_spacing[0], n_t)
            r = np.linspace(grid_offset[1], grid_offset[1] + (n_r - 1) * m.grid_spacing[1], n_r)
            axes = {"r": r, "t": t}
            dim = "rt"
            axes_order = m.axis_labels[::-1]
            array = np.transpose(array, (0, 2, 1))
        elif len(m.axis_labels) == 3:  # 'xyt'
            n_x = array.shape[2]
            n_y = array.shape[1]
            n_t = array.shape[0]
            grid_offset = m.get_attribute("gridGlobalOffset")
            x = np.linspace(grid_offset[2], grid_offset[2] + (n_x - 1) * m.grid_spacing[2], n_x)
            y = np.linspace(grid_offset[1], grid_offset[1] + (n_y - 1) * m.grid_spacing[1], n_y)
            t = np.linspace(grid_offset[0], grid_offset[0] + (n_t - 1) * m.grid_spacing[0], n_t)
            axes = {"x": x, "y": y, "t": t}
            dim = "xyt"
            axes_order = m.axis_labels[::-1]
            array = np.transpose(array, (2, 1, 0))
        else:
            print(
                "Error: The dimension of the field is not supported. The valid dimensions are 'rt' and 'xyt'."
            )
            return None

        # If the field is stored as vector potential, convert it to field
        if m.get_attribute("envelopeField") == "normalized_vector_potential":
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
