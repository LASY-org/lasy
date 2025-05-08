import os

import numpy as np
import openpmd_api as io
from scipy.constants import c

from lasy import __version__ as lasy_version

from .laser_utils import field_to_vector_potential


def write_to_openpmd_file(
    dim,
    write_dir,
    file_prefix,
    file_format,
    iteration,
    grid,
    wavelength,
    pol,
    save_as_vector_potential=False,
):
    """
    Write the laser field into an openPMD file.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - 'xyt': The laser pulse is represented on a 3D grid:
                 Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - 'rt' : The laser pulse is represented on a 2D grid:
                 Cylindrical (r) transversely, and temporal (t) longitudinally.

    file_prefix : string
        The file name will start with this prefix.

    write_dir : string
        The directory where the file will be written.

    file_format : string
        Format to be used for the output file. Options are "h5" and "bp".

    iteration : int
        The iteration number for the file to be written.

    grid : Grid
        A grid object containing the 3-dimensional array
        with complex envelope of the electric field and metadata

    wavelength : scalar
        Central wavelength for which the laser pulse envelope is defined.

    pol : list of 2 complex numbers
        Polarization vector that multiplies array to get the Ex and Ey arrays.

    save_as_vector_potential : bool (optional)
        Whether the envelope is converted to normalized vector potential
        before writing to file.
    """
    array = grid.get_temporal_field()

    # Create file
    full_filepath = os.path.join(
        write_dir, "{}_%05T.{}".format(file_prefix, file_format)
    )
    os.makedirs(write_dir, exist_ok=True)
    series = io.Series(full_filepath, io.Access.create)
    series.set_software("lasy", lasy_version)

    i = series.iterations[iteration]

    # Define the mesh
    m = i.meshes["laserEnvelope"]
    m.grid_spacing = [
        (hi - lo) / (npoints - 1)
        for hi, lo, npoints in zip(grid.hi, grid.lo, grid.npoints)
    ][::-1]
    m.grid_global_offset = grid.lo[::-1]
    m.grid_global_offset[0] += grid.position / c
    if dim == "xyt":
        m.geometry = io.Geometry.cartesian
        m.axis_labels = ["t", "y", "x"]
    elif dim == "rt":
        m.geometry = io.Geometry.thetaMode
        m.axis_labels = ["t", "r"]

    # Store metadata needed to reconstruct the field
    m.set_attribute("angularFrequency", 2 * np.pi * c / wavelength)
    m.set_attribute("polarization", pol)
    if save_as_vector_potential:
        m.set_attribute("envelopeField", "normalized_vector_potential")
        m.unit_dimension = {}
    else:
        m.set_attribute("envelopeField", "electric_field")
        m.unit_dimension = {
            io.Unit_Dimension.M: 1,
            io.Unit_Dimension.L: 1,
            io.Unit_Dimension.I: -1,
            io.Unit_Dimension.T: -3,
        }

    if save_as_vector_potential:
        array = field_to_vector_potential(grid, 2 * np.pi * c / wavelength)

    # Pick the correct field
    assert dim in ["xyt", "rt"]
    if dim == "xyt":
        # Switch from x,y,t (internal to lasy) to t,y,x (in openPMD file)
        # This is because many PIC codes expect x to be the fastest index
        data = np.transpose(array).copy()
    else:  # dim == "rt"
        # The representation of modes in openPMD
        # (see https://github.com/openPMD/openPMD-standard/blob/latest/STANDARD.md#required-attributes-for-each-mesh-record)
        # is different than the representation of modes internal to lasy.
        # Thus, there is a non-trivial conversion here
        ncomp = 2 * grid.n_azimuthal_modes - 1
        data = np.zeros((ncomp, grid.npoints[0], grid.npoints[1]), dtype=array.dtype)
        data[0, :, :] = array[0, :, :]
        for mode in range(1, grid.n_azimuthal_modes):
            # cos(m*theta) part of the mode
            data[2 * mode - 1, :, :] = array[mode, :, :] + array[-mode, :, :]
            # sin(m*theta) part of the mode
            data[2 * mode, :, :] = -1.0j * array[mode, :, :] + 1.0j * array[-mode, :, :]
        # Switch from m,r,t (internal to lasy) to m,t,r (in openPMD file)
        # This is because many PIC codes expect r to be the fastest index
        data = np.transpose(data, axes=[0, 2, 1]).copy()

    # Define the dataset
    dataset = io.Dataset(data.dtype, data.shape)
    env = m[io.Mesh_Record_Component.SCALAR]
    env.position = np.zeros(len(dim), dtype=np.float64)
    env.reset_dataset(dataset)
    env.store_chunk(data)

    series.flush()
    series.close()


def extract_array(m, series, component=None):
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


def convert_modes(arr_list, geometry, is_env, verbose=False):
    """
    Convert from openPMD mode decomposition to LASY mode decomposition.

    Convert from openPMD mode decomposition of the electric field in cos(m*theta) and sin(m*theta), stored, m in [0, Nmodes] to LASY mode decomposition exp(i*m*theta) m in [-Nmodes+1,Nmodes-1] (array of complex numbers, see https://github.com/LASY-org/lasy/blob/development/README.md):
     - Electromagnetic + cylindrical: we assume Er and Etheta, and construct Ex and Ey.
        https://github.com/openPMD/openPMD-standard/blob/latest/STANDARD.md#required-attributes-for-each-mesh-record. Complex modes, the real and imag part are stored in 2 real arrays.
     - Envelope + cylindrical: we assume the array is E + polarization. Complex modes, stored as arrays of complex numbers. See openPMD link above aas well as https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_LaserEnvelope.md.
     - Cartesian: do not do anything.

    Parameters
    ----------
    arr_list : list of Numpy arrays
        List of 3D arrays to be converted.

    geometry : string
        Geometry of input data from openPMD standard, "cartesian" or "thetaMode" supported.

    is_env : bool
        Whether the input data represents a laser envelope.
        Otherwise electric field is assumed, specifically x-polarized at the moment.

    verbose : bool (optional)
        If true, print some more intermediate steps.

    Returns
    -------
    arrays_out : list of 3D array
        The arrays converted to LASY mode decomposition.
        This is still the full field, not yet the envelope.
    """
    if geometry == "cartesian":
        return arr_list

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
        return [array_out]
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
        Ex_out = np.zeros(shape=(2 * nmodes_out - 1, *Er_in.shape[1:]))
        Ey_out = np.zeros(shape=(2 * nmodes_out - 1, *Er_in.shape[1:]))
        # The _in arrays have real and imag parts separated, so we add them
        # together by hand
        for imode in range(nmodes_out):
            # input 2*m - 1 and 2*m are real and imag part of mode m, respectively
            # The +1 is conversion from Er & Etheta representation to Ex
            # output exp(i*m*theta) modes: for some reasons, all data goes in
            # m >= 0 modes
            Ex_out[imode, :, :] = 0.5 * (
                Er_in[2 * (imode + 1) - 1, :, :] - Et_in[2 * (imode + 1), :, :]
            )
            Ey_out[imode, :, :] = 0.5 * (
                Er_in[2 * (imode + 1), :, :] + Et_in[2 * (imode + 1) - 1, :, :]
            )
            # Here we assume that mode 0 is only plasma, so we never use it
            if imode >= 2:
                Ex_out[imode, :, :] += 0.5 * (
                    Er_in[2 * (imode - 1) - 1, :, :] + Et_in[2 * (imode - 1), :, :]
                )
                Ey_out[imode, :, :] += 0.5 * (
                    -Er_in[2 * (imode - 1), :, :] + Et_in[2 * (imode - 1) - 1, :, :]
                )
        return [Ex_out, Ey_out]


def isolate_polarization(arrays, dim):
    """
    Extract single array and polarization vector from Ex and Ey.

    Parameters
    ----------
    arrays : list of 2 3D numpy arrays
        Envelopes of Ex and Ey.

    dim : string
        Dimensionality of the array. Options are:

        - 'xyt': The laser pulse is represented on a 3D grid:
                 Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - 'rt' : The laser pulse is represented on a 2D grid:
                 Cylindrical (r) transversely, and temporal (t) longitudinally.

    Returns
    -------
    array : 3D numpy array
        Laser envelope at LASY convention

    pol : tuple of 2 elements
        Polarization vector (px, py), both px and py are complex numbers.
    """
    Ex, Ey = arrays  # Ex and Ey envelopes
    if dim == "rt":
        print(
            "Cylindrical input with full field: polarization is extract from mode 0 only"
        )
        Ex = Ex[0]
        Ey = Ey[0]
    rho2 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    # Amplitude of polarization vectors
    rho_x = np.sqrt(np.sum(np.abs(Ex) ** 2) / np.sum(rho2))
    rho_y = np.sqrt(np.sum(np.abs(Ey) ** 2) / np.sum(rho2))
    # Phase in x is assumed 0. This is a convention.
    phi_x = 0
    phi_y = np.average(np.angle(Ey) - np.angle(Ex), weights=rho2)
    px = rho_x * np.exp(1j * phi_x)
    py = rho_y * np.exp(1j * phi_y)
    pol = (px, py)
    print("polarization state detected: (px, py) =", pol)
    array = arrays[0] / px if rho_x >= rho_y else arrays[1] / py

    return array, pol
