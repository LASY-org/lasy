import openpmd_api as io

def write_to_openpmd_file(file_prefix, file_format, box, dim, array_in, pol):
    """
    Write the laser field into an openPMD file

    Parameters
    ----------
    file_prefix: string
        The file name will start with this prefix.

    file_format: string
        Format to be used for the output file. Options are "h5" and "bp".

    box: an object of type lasy.utils.box.Box
        Defines the grid over which the laser is dumped

    dim: string
        Dimension of the array, 'rz' or 'xyz'

    array_in: numpy complex array
        n-dimensional (n=2 for dim='rz', n=3 for dim='xyz') array with laser field
        The array should contain the complex envelope of the electric field.

    pol: list of 2 complex numbers
        Polarization vector that multiplies array_in to get the Ex and Ey fields.
    """
    # Create file
    series = io.Series(
        "{}_%05T.{}".format(file_prefix, file_format),
        io.Access.create)
    i = series.iterations[0]

    # Define the field metadata
    E = i.meshes["E"]
    E.grid_spacing = [ (hi-lo)/npoints for hi, lo, npoints in \
                           zip( box.hi, box.lo, box.npoints ) ]
    E.grid_global_offset = box.lo
    E.axis_labels = ['x', 'y', 't']
    E.unit_dimension = {
        io.Unit_Dimension.M:  1,
        io.Unit_Dimension.L:  1,
        io.Unit_Dimension.I: -1,
        io.Unit_Dimension.T: -3
    }
    if dim == 'xyz':
        E.geometry = io.Geometry.cartesian
    elif dim == 'rz':
        E.geometry = io.Geometry.thetaMode

    # Define the data sets
    dataset = io.Dataset(field.dtype, field.shape)

    Ex = E["x"]
    Ex.position = [0]*len(dim)
    Ex.reset_dataset(dataset)
    Ex.store_chunk( field*pol[0] )

    Ey = E["y"]
    Ey.position = [0]*len(dim)
    Ey.reset_dataset(dataset)
    Ey.store_chunk( field*pol[1] )

    series.flush()
