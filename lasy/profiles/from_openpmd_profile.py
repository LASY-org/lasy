import numpy as np
import openpmd_api as io
from scipy.constants import c

from lasy.utils.laser_utils import (
    create_grid,
    field_to_envelope,
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
        Name of the field containing the laser pulse.

    coordinate : string
        Name of the component of the field to be read.

    omega0 : float
        Angular frequency at which laser envelope is defined.

    is_envelope : bool
        Whether the field provided uses the (complex) envelope representation, as
        used internally in lasy. If False, field is assumed to represent the
        the full (real) electric field (with fast oscillations).
    """

    def __init__(
        self,
        path,
        iteration,
        field,
        coordinate=None,
        omega0=None,
        is_envelope=True,
    ):
        # Read the data
        series = io.Series(path, io.Access.read_only)
        i = series.iterations[iteration]
        m = i.meshes[field]

        if coordinate is None:
            array = m[io.Mesh_Record_Component.SCALAR].load_chunk()
        else:
            array = m[coordinate].load_chunk()
        series.flush()
        # This is rqeuired for creating the grid
        if is_envelope:
            array = array.astype(np.complex128)
        else:
            array = array.astype(np.float64)

        # Extract the required parameters to set the grid
        grid_offset = m.get_attribute("gridGlobalOffset")
        grid_spacing = m.get_attribute("gridSpacing")
        try:
            grid_position = m.get_attribute(
                "position"
            )  # node (0.0) or cell (0.5) centered info for each axis
        except io.ErrorNoSuchAttribute:
            grid_position = m[coordinate].get_attribute("position")
        axis_labels = m.get_attribute("axisLabels")

        # Read/set polarization.
        try:
            pol = m.get_attribute("polarization")
        except io.ErrorNoSuchAttribute:
            print("Polarization not found. Defaulting to (1, 0)")
            pol = (1, 0)

        if len(axis_labels) == 2:
            idx_offset = 1
            dim = "rt"
        elif len(axis_labels) == 3:
            idx_offset = 0
            dim = "xyt"
        else:
            print(
                "Error: The dimension of the field is not supported. The valid dimensions are 'rt' and 'xyt'."
            )
            raise ValueError

        # Define parameters to create a profile
        axes = {}
        axes_order = []
        for idx, label in enumerate(axis_labels):
            # Define the axis array
            N = array.shape[idx + idx_offset]
            axis = np.linspace(
                grid_offset[idx] + grid_position[idx] * grid_spacing[idx],
                grid_offset[idx] + (N - 1 + grid_position[idx]) * grid_spacing[idx],
                N,
            )
            # If label is `z`, change it to `t`
            if label == "z":
                axis = (axis - axis[0]) / c
                array = np.flip(array, axis=idx + idx_offset)
                label = "t"

            # Add axis to the dictionary and label to the list
            axes[label] = axis
            axes_order.append(label)

        # Set the LASY order here.
        # (If not, `create_grid` will fail below when converting
        # from vector potential to electric field.)
        if axes_order[0] == "t":
            axes_order = axes_order[::-1]
            array = np.swapaxes(array, idx_offset, 2)

        # Read angular frequency
        if is_envelope:
            if omega0 is not None:
                omg0 = omega0
            else:
                try:
                    omg0 = m.get_attribute("angularFrequency")
                except io.ErrorNoSuchAttribute:
                    raise ValueError(
                        "Angular frequency not found. Please provide the value.\
                            If you are using Wake-T, please store the field as a"
                    )
        else:  # If electric field is provided, convert it to envelope
            assert omega0 is None
            temp_grid = create_grid(array, axes, dim, is_envelope=False)
            grid, omg0 = field_to_envelope(temp_grid, dim)

        wavelength = 2 * np.pi * c / omg0

        # If the field is stored as vector potential,
        # convert it to electric field
        vector_to_field = False
        try:
            if m.get_attribute("envelopeField") == "normalized_vector_potential":
                vector_to_field = True
        except io.ErrorNoSuchAttribute:
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
