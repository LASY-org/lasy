import numpy as np
import openpmd_api as io
from scipy.constants import c

from lasy.utils.laser_utils import (
    create_grid,
    field_to_envelope,
    vector_potential_to_field,
)

from .from_array_profile import FromArrayProfile


def _reorder_array(array, m, position, verbose=False):
    axis_labels = m.get_attribute("axisLabels")
    grid_offset = m.get_attribute("gridGlobalOffset")
    grid_spacing = m.get_attribute("gridSpacing")
    if len(axis_labels) == 2:
        idx_offset = 1
        assert axis_labels in [["r", "z"], ["z", "r"], ["r", "t"], ["t", "r"]]
    elif len(axis_labels) == 3:
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
    if dim_in == "cartesian":
        assert len(arr_list) == 1
        return arr_list[0]
    # Convert to LASY mode decomposition exp(i*m*theta) m in [-N+1,N-1]:
    # - Electromagnetic + cylindrical: we assume Er and Etheta
    #   https://github.com/openPMD/openPMD-standard/blob/latest/STANDARD.md#required-attributes-for-each-mesh-record
    # - Envelope + cylindrical: we assume Ex (actually Epol)
    #   Modes in file are still assumes cos(theta) and sin(theta)
    nmodes_in = (arr_list[0].shape[0] + 1) // 2
    if verbose:
        print("nmodes_in:", nmodes_in)
    if is_env:
        assert len(arr_list) == 1
        assert np.iscomplexobj(arr_list[0])
        array_in = arr_list[0]
        array_out = np.zeros_like(arr_list[0], dtype="complex128")
        array_out[0, :, :] = array_in[0, :, :]
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
            # 2*m - 1 and 2*m are real and imag part of mode m, respectively
            # The +1 is conversion from Er & Etheta representation to Ex
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

    Parameters
    ----------
    path : string
        Path to the openPMD file containing the laser field or envelope.

    iteration : int
        Iteration at which the argument is read.

    field : string
        Name of the field containing the laser pulse.

    component : string
        Name of the component of the field to be read.
    """

    def __init__(
        self,
        filename,
        dimension,
        is_envelope,
        field_name=None,
        verbose=False
    ):
        assert dimension in ["cartesian", "cylindrical"]
        dim = "rt" if dimension == "cylindrical" else "xyt"
        series = io.Series(filename, io.Access.read_only)
        iterations = np.array(series.iterations)
        i = series.iterations[iterations[-1]]
        if is_envelope:
            if verbose:
                print("Read envelope")
            assert field_name is not None, "field_name must be specified for an envelope"
            m = i.meshes[field_name]
            omg0 = m.get_attribute("angularFrequency")
            try:
                envelopeField = m.get_attribute("envelopeField")
                pol = m.get_attribute("polarization")
            except:
                envelopeField = "normalized_vector_potential"
                pol = (1, 0)
                print("WARNING: 'envelopeField' and/or 'polarization' attributes must be specified according to the standard but are currently missing for mesh record " + field_name + ", see https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_LaserEnvelope.md. Assumed 'normalized_vector_potential' and (1,0), respectively.")
            array = m[io.Mesh_Record_Component.SCALAR].load_chunk()
            # node (0.0) or cell (0.5) centered info for each axis
            position = m.get_attribute("position")
            series.flush()
            axes_order, axes, array = _reorder_array(array, m, position)
            assert dimension == 'cylindrical' and axes_order == ['r', 't'] or \
                   dimension == 'cartesian' and axes_order == ['x', 'y', 't'], "'dimension' not consistent with properties of array read from openPMD file"
            array = _convert_modes([array], dimension, is_envelope)
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
                # Get data `array` and `position`.
                array = m[component].load_chunk()
                position = m[component].get_attribute("position")
                series.flush()
                axes_order, axes, array = _reorder_array(array, m, position)
                array_list.append(array)
            array = _convert_modes(array_list, dimension, is_envelope)
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
