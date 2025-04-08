import numpy as np
import openpmd_api as io
from scipy.constants import c

from lasy.utils.laser_utils import (
    create_grid,
    field_to_envelope,
    vector_potential_to_field,
)

from .from_array_profile import FromArrayProfile


def _extract_array(m, series, component=None):
    """
    Extract and reorder openPMD-formatted array to LASY ordering.

    Namely, ensure that:
     - The longitudinal dimension is t, not z
     - The last axis is t

    Parameters
    ----------
    m : openPMD-api mesh record object
        The array and metadata are read from this.

    series : openPMD Series
        The series containing data m. Only passed for the flush.

    Returns
    -------
    axes_order : List of strings
        Name and ordering of the axes array.
        Input argument for FromArrayProfile, see there for more details.

    axes : Python dictionary containing the axes vectors
        e.g. keys: 'x', 'y', 't' and values: the 1D arrays of each axis.
        Input argument for FromArrayProfile, see there for more details.

    array : 3D array of complex numbers
        Reordered array, with axes in the right order and t last.

    """
    if component is not None:
        array = m[component].load_chunk()
        position = m[component].get_attribute("position")
    else:
        array = m[io.Mesh_Record_Component.SCALAR].load_chunk()
        position = m.get_attribute("position")
    series.flush()
    # node (0.0) or cell (0.5) centered info for each axis
    axis_labels = m.get_attribute("axisLabels")
    grid_offset = m.get_attribute("gridGlobalOffset")
    grid_spacing = m.get_attribute("gridSpacing")
    assert len(axis_labels) in [2, 3]
    if len(axis_labels) == 2:
        idx_offset = 1
        assert axis_labels in [["r", "z"], ["z", "r"], ["r", "t"], ["t", "r"]]
    else:  # len(axis_labels) == 3
        idx_offset = 0
        assert axis_labels in [
            ["x", "y", "z"],
            ["z", "y", "x"],
            ["x", "y", "t"],
            ["t", "y", "x"],
        ]

    # Define parameters to create a profile
    axes = {}
    axes_order = []
    for idx, label in enumerate(axis_labels):
        # Define the axis array
        N = array.shape[idx + idx_offset]
        axis = np.linspace(
            grid_offset[idx] + position[idx] * grid_spacing[idx],
            grid_offset[idx] + (N - 1 + position[idx]) * grid_spacing[idx],
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

    # Set to LASY order here, time is last axis.
    if axes_order[0] == "t":
        axes_order = axes_order[::-1]
        array = np.swapaxes(array, idx_offset, 2)

    return axes_order, axes, array


def _convert_modes(arr_list, dim_in, is_env, verbose=False):
    """
    Convert from openPMD mode decomposition to LASY mode decomposition.

    Convert from openPMD mode decomposition in cos(m*theta) and sin(m*theta), stored, m in [0, Nmodes] to LASY mode decomposition exp(i*m*theta) m in [-Nmodes+1,Nmodes-1] (array of complex numbers, see https://github.com/LASY-org/lasy/blob/development/README.md):
     - Electromagnetic + cylindrical: we assume Er and Etheta
        https://github.com/openPMD/openPMD-standard/blob/latest/STANDARD.md#required-attributes-for-each-mesh-record. Complex modes, the real and imag part are stored in 2 real arrays.
     - Envelope + cylindrical: we assume the array is Ex (in principle, we should measure the polarization). Complex modes, stored as arrays of complex numbers. See openPMD link above aas well as https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_LaserEnvelope.md.
     - Cartesian: do not do anything.

    Parameters
    ----------
    arr_list : list of Numpy arrays
        List of 3D arrays to be converted. They are processed independently.

    dim_in : string
        "cartesian" or "cylindrical". Dimensionality of input data.

    is_env : bool
        Whether the input data represents a laser envelope.
        Otherwise electric field is assumed, specifically x-polarized at the moment.

    verbose : bool (optional)
        If true, print some more intermediate steps.

    Returns
    -------
    array_out : 3D array
        The array converted to LASY mode decomposition.
        This is still the full field, not yet the envelope.
    """
    if dim_in == "cartesian":
        assert len(arr_list) == 1
        return arr_list[0]
    nmodes_in = (arr_list[0].shape[0] + 1) // 2
    if verbose:
        print("nmodes_in:", nmodes_in)
    if is_env:
        assert len(arr_list) == 1
        assert np.iscomplexobj(arr_list[0])
        array_in = arr_list[0]
        array_out = np.zeros_like(arr_list[0], dtype="complex128")
        array_out[0, :, :] = array_in[0, :, :]
        # The data is already Ex, we simply to convert from
        # cos(m*theta) and sin(m*theta) to exp(i*m*theta).
        for imode in range(1, nmodes_in):
            array_out[imode, :, :] = 0.5 * (
                array_in[2 * imode - 1] + 1j * array_in[2 * imode]
            )
            array_out[-imode, :, :] = 0.5 * (
                array_in[2 * imode - 1] - 1j * array_in[2 * imode]
            )
    else:
        # arr_list contains 2 elements, Er and Etheta, that need to be
        # combined into Ex. At this point, still operating on full field.
        assert len(arr_list) == 2
        assert np.isrealobj(arr_list[0]) and np.isrealobj(arr_list[1])
        Er_in = arr_list[0]
        Et_in = arr_list[1]

        nmodes_out = nmodes_in - 1
        if verbose:
            print("nmodes_out:", nmodes_in)
        array_out = np.zeros(shape=(2 * nmodes_out - 1, *Er_in.shape[1:]))
        # The _in arrays have real and imag parts separated, so we add them
        # together by hand
        for imode in range(nmodes_out):
            # input 2*m - 1 and 2*m are real and imag part of mode m, respectively
            # The +1 is conversion from Er & Etheta representation to Ex
            # output exp(i*m*theta) modes: for some reasons, all data goes in
            # m >= 0 modes
            array_out[imode, :, :] = 0.5 * (
                Er_in[2 * (imode + 1) - 1, :, :] - Et_in[2 * (imode + 1), :, :]
            )
            # Here we assume that mode 0 is only plasma, so we never use it
            if imode >= 2:
                array_out[imode, :, :] += 0.5 * (
                    Er_in[2 * (imode - 1) - 1, :, :] + Et_in[2 * (imode - 1), :, :]
                )

    return array_out


class FromOpenPMDProfile(FromArrayProfile):
    r"""
    Profile defined from an openPMD file.

    Upon initialization, read from an openPMD profile, builds interpolation objects on the array data and use them to create function evaluate.

    Parameters
    ----------
    filename : string
        Name of openPMD file to read the envelope from, including path.

    dimension : string
        "cartesian" or "cylindrical".
        Dimensionality of the data from the openPMD file.

    is_envelope : bool
        Whether the openPMD file represents a laser envelope.
        Otherwise, electric field is assumed, and its envelope is extracted.

    field_name : string (optional)
        Required if is_envelope is True.
        The name of the envelope field (this is not prescribed by the openPMD standard for the envelope).

    verbose : bool (optional)
        If true, print some more intermediate steps.
    """

    def __init__(
        self, filename, dimension, is_envelope, field_name=None, verbose=False
    ):
        assert dimension in ["cartesian", "cylindrical"]
        dim = "rt" if dimension == "cylindrical" else "xyt"
        series = io.Series(filename, io.Access.read_only)
        iterations = np.array(series.iterations)
        i = series.iterations[iterations[-1]]
        if is_envelope:
            if verbose:
                print("Read envelope")
            assert field_name is not None, (
                "field_name must be specified for an envelope"
            )
            m = i.meshes[field_name]
            omg0 = m.get_attribute("angularFrequency")
            try:
                envelopeField = m.get_attribute("envelopeField")
                pol = m.get_attribute("polarization")
            except Exception:
                envelopeField = "normalized_vector_potential"
                pol = (1, 0)
                print(
                    "WARNING: 'envelopeField' and/or 'polarization' attributes must be specified according to the standard but are currently missing for mesh record "
                    + field_name
                    + ", see https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_LaserEnvelope.md. Assumed 'normalized_vector_potential' and (1,0), respectively."
                )
            axes_order, axes, array = _extract_array(m, series)
            assert (
                dimension == "cylindrical"
                and axes_order == ["r", "t"]
                or dimension == "cartesian"
                and axes_order == ["x", "y", "t"]
            ), (
                "'dimension' not consistent with properties of array read from openPMD file"
            )
            array = _convert_modes([array], dimension, is_envelope, verbose)
            if envelopeField == "normalized_vector_potential":
                if verbose:
                    print("Convert from vector potential to electric field")
                grid = create_grid(array, axes, dim)
                array = vector_potential_to_field(grid, omg0)
        else:
            if dimension == "cartesian":
                field_list = ["E"]
                coord_list = ["x"]
            else:
                field_list = ["E", "E"]
                coord_list = ["r", "t"]
            array_list = []
            for count, field in enumerate(field_list):
                # Read the data
                m = i.meshes[field]
                component = coord_list[count]
                axes_order, axes, array = _extract_array(m, series, component)
                array_list.append(array)
            array = _convert_modes(array_list, dimension, is_envelope, verbose)
            grid = create_grid(array, axes, dim, is_envelope=False)
            omg0 = field_to_envelope(grid, dim)
            array = grid.get_temporal_field()
            pol = (1, 0)
        wavelength = 2 * np.pi * c / omg0

        super().__init__(
            wavelength=wavelength,
            pol=pol,
            array=array,
            dim=dim,
            axes=axes,
            axes_order=axes_order,
        )
