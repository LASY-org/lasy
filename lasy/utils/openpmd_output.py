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
        Polarization vector that multiplies array_in to get the Ex and Ey array_ins.
    """
    # Create file
    series = io.Series(
        "{}_%05T.{}".format(file_prefix, file_format),
        io.Access.create)
    i = series.iterations[0]

    # Define Ex_real, Ex_imag, Ey_real, Ey_imag as scalar records
    for comp_name in ['Ex_real', 'Ex_imag', 'Ey_real', 'Ey_imag']:

        # Define the mesh
        m = i.meshes[comp_name]
        m.grid_spacing = [ (hi-lo)/npoints for hi, lo, npoints in \
                               zip( box.hi, box.lo, box.npoints ) ]
        m.grid_global_offset = box.lo
        m.axis_labels = ['x', 'y', 't']
        m.unit_dimension = {
            io.Unit_Dimension.M:  1,
            io.Unit_Dimension.L:  1,
            io.Unit_Dimension.I: -1,
            io.Unit_Dimension.T: -3
        }
        if dim == 'xyz':
            m.geometry = io.Geometry.cartesian
        elif dim == 'rz':
            m.geometry = io.Geometry.thetaMode

        # Define the dataset
        dataset = io.Dataset(array_in.real.dtype, array_in.real.shape)
        E = m[io.Mesh_Record_Component.SCALAR]
        E.position = [0]*len(dim)
        E.reset_dataset(dataset)

        # Pick the correct field
        if comp_name == 'Ex_real':
            data = (array_in*pol[0]).real.copy()
        elif comp_name == 'Ex_imag':
            data = (array_in*pol[0]).imag.copy()
        elif comp_name == 'Ey_real':
            data = (array_in*pol[1]).real.copy()
        elif comp_name == 'Ey_imag':
            data = (array_in*pol[1]).imag.copy()
        E.store_chunk( data )

    series.flush()
