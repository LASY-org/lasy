import numpy as np
import openpmd_api as io
from scipy.constants import c


def write_to_openpmd_file(dim, file_prefix, file_format, field, wavelength, pol):
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

    file_format : string
        Format to be used for the output file. Options are "h5" and "bp".

    grid : Grid
        A grid object containing the 3-dimensional array
        with complex envelope of the electric field and metadata

    wavelength : scalar
        Central wavelength for which the laser pulse envelope is defined.

    pol : list of 2 complex numbers
        Polarization vector that multiplies array to get the Ex and Ey arrays.
    """
    array = field.array

    # Create file
    series = io.Series("{}_%05T.{}".format(file_prefix, file_format), io.Access.create)
    i = series.iterations[0]

    # Define the mesh
    m = i.meshes["laserEnvelope"]
    m.grid_spacing = [
        (hi - lo) / (npoints - 1)
        for hi, lo, npoints in zip(field.hi, field.lo, field.npoints)
    ][::-1]
    m.grid_global_offset = field.lo[::-1]
    m.unit_dimension = {
        io.Unit_Dimension.M: 1,
        io.Unit_Dimension.L: 1,
        io.Unit_Dimension.I: -1,
        io.Unit_Dimension.T: -3,
    }
    if dim == "xyt":
        m.geometry = io.Geometry.cartesian
        m.axis_labels = ["t", "y", "x"]
    elif dim == "rt":
        m.geometry = io.Geometry.thetaMode
        m.axis_labels = ["t", "r"]

    # Store metadata needed to reconstruct the field
    m.set_attribute("angularFrequency", 2 * np.pi * c / wavelength)
    m.set_attribute("polarization", pol)
    m.set_attribute("isLaserEnvelope", True)

    # Pick the correct field
    if dim == "xyt":
        # Switch from x,y,t (internal to lasy) to t,y,x (in openPMD file)
        # This is because many PIC codes expect x to be the fastest index
        data = np.transpose(array).copy()
    elif dim == "rt":
        # The representation of modes in openPMD
        # (see https://github.com/openPMD/openPMD-standard/blob/latest/STANDARD.md#required-attributes-for-each-mesh-record)
        # is different than the representation of modes internal to lasy.
        # Thus, there is a non-trivial conversion here
        ncomp = 2 * field.n_azimuthal_modes - 1
        data = np.zeros((ncomp, field.npoints[0], field.npoints[1]), dtype=array.dtype)
        data[0, :, :] = array[0, :, :]
        for mode in range(1, field.n_azimuthal_modes):
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
    env.position = [0] * len(dim)
    env.reset_dataset(dataset)
    env.store_chunk(data)

    series.flush()
