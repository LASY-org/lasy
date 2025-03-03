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
        Name of the field containing the laser pulse

    is_envelope : boolean
        Whether the field to read represents a laser envelope.
        If not, the envelope is obtained from the electric field
        using a Hilbert transform. If not specified, lasy will try to guess
        whether the field is an envelope by checking whether it is a complex
        array.

    phase_unwrap_nd : boolean (optional)
        If True, the phase unwrapping is n-dimensional (2- or 3-D depending on dim).
        If False, the phase unwrapping is done in t, treating each transverse cell
        separately. This should be less accurate but faster.
        If set to True, scikit-image must be installed.
    """

    def __init__(
        self,
        path,
        iteration,
        field,
        is_envelope=False,
        phase_unwrap_nd=False,
        polarization=None,
    ):
        series = io.Series(path, io.Access.read_only)
        i = series.iterations[iteration]
        m = i.meshes[field]
        print(m.axis_labels)
        print(m.shape)
        arr = m[io.Mesh_Record_Component.SCALAR].load_chunk()
        series.flush()
        omg0 = m.get_attribute("angularFrequency")
        wavelength = 2 * np.pi * c / omg0
        if polarization is None:
            pol = m.get_attribute("polarization")
        else:
            pol = polarization

        if len(m.axis_labels) == 2:  # 'rt'
            n_r = int(arr.shape[2])
            n_t = int(arr.shape[1] / 2)

            r = np.linspace(0, n_r * m.grid_spacing[1], n_r)
            t = np.linspace(-n_t * m.grid_spacing[0], n_t * m.grid_spacing[0], 2 * n_t)

            axes = {"r": r, "t": t}
            dim = "rt"
            axes_order = m.axis_labels[::-1]

        elif len(m.axis_labels) == 3:  # 'xyt'
            n_x = int(arr.shape[2] / 2)
            n_y = int(arr.shape[1] / 2)
            n_t = int(arr.shape[0] / 2)

            x = np.linspace(-n_x * m.grid_spacing[2], n_x * m.grid_spacing[2], 2 * n_x)
            y = np.linspace(-n_y * m.grid_spacing[1], n_y * m.grid_spacing[1], 2 * n_y)
            t = np.linspace(-n_t * m.grid_spacing[0], n_t * m.grid_spacing[0], 2 * n_t)

            axes = {"x": x, "y": y, "t": t}
            dim = "xyt"
            axes_order = m.axis_labels[::-1]

        else:
            print(
                "Error: The dimension of the field is not supported. The valid dimensions are 'rt' and 'xyt'."
            )
            return None

        # If array does not contain the envelope but the electric field,
        # extract the envelope with a Hilbert transform
        if is_envelope == True:
            grid = create_grid(arr, axes, dim, is_envelope=is_envelope)
            grid, omg0 = field_to_envelope(grid, dim, phase_unwrap_nd)
            data = grid.get_temporal_field()[0]
        else:
            pass

        # If the filed is stored in form of a vector potential
        if m.get_attribute("envelopeField") == "normalized_vector_potential":
            if dim == "rt":
                grid = create_grid(np.transpose(arr, (0, 2, 1)), axes, dim)
                data_rt = vector_potential_to_field(grid, omg0)
                data = data_rt[0, :, :]

            else:
                grid = create_grid(np.transpose(arr, (2, 1, 0)), axes, dim)
                data = vector_potential_to_field(grid, omg0)
        else:
            if dim == "rt":
                data = np.transpose(arr[0, :, :], (1, 0))
            else:
                data = np.transpose(arr, (2, 1, 0))

        data = data / np.max(np.abs(data))  # Normalization

        super().__init__(
            wavelength=wavelength,
            pol=pol,
            array=data,
            dim=dim,
            axes=axes,
            axes_order=axes_order,
        )
