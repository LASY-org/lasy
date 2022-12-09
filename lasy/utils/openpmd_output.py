import numpy as np
import openpmd_api as io

def write_to_openpmd_file(file_prefix, file_format, grid,
                          wavelength, pol):
    """
    Write the laser field into an openPMD file

    Parameters
    ----------
    file_prefix: string
        The file name will start with this prefix.

    file_format: string
        Format to be used for the output file. Options are "h5" and "bp".

    grid: Grid
        A grid object containing the 3-dimensional array
        with complex envelope of the electric field and metadata

    wavelength: scalar
        Central wavelength for which the laser pulse envelope is defined.

    pol: list of 2 complex numbers
        Polarization vector that multiplies array to get the Ex and Ey arrays.
    """
    array = grid.field
    box = grid.box
    dim = box.dim

    # Create file
    series = io.Series(
        "{}_%05T.{}".format(file_prefix, file_format),
        io.Access.create)
    i = series.iterations[0]

    # Store metadata needed to reconstruct the field
    i.set_attribute("wavelength", wavelength)
    i.set_attribute("pol", pol)

    # Define E_real, E_imag as scalar records
    for comp_name in ['E_real', 'E_imag']:

        # Define the mesh
        m = i.meshes[comp_name]
        m.grid_spacing = [ (hi-lo)/npoints for hi, lo, npoints in \
                               zip( box.hi, box.lo, box.npoints ) ]
        m.grid_global_offset = box.lo
        m.unit_dimension = {
            io.Unit_Dimension.M:  1,
            io.Unit_Dimension.L:  1,
            io.Unit_Dimension.I: -1,
            io.Unit_Dimension.T: -3
        }
        if dim == 'xyt':
            m.geometry = io.Geometry.cartesian
            m.axis_labels = ['x', 'y', 't']
        elif dim == 'rt':
            m.geometry = io.Geometry.thetaMode
            m.axis_labels = ['r', 't']

        # Define the dataset
        dataset = io.Dataset(array.real.dtype, array.real.shape)
        E = m[io.Mesh_Record_Component.SCALAR]
        E.position = [0]*len(dim)
        E.reset_dataset(dataset)

        # Pick the correct field
        if dim == 'xyt':
            if comp_name == 'E_real':
                data = array.real.copy()
            elif comp_name == 'E_imag':
                data = array.imag.copy()

        elif dim == 'rt':
            # The representation of modes in openPMD
            # (see https://github.com/openPMD/openPMD-standard/blob/latest/STANDARD.md#required-attributes-for-each-mesh-record)
            # is different than the representation of modes internal to lasy.
            # Thus, there is a non-trivial conversion here
            ncomp = 2*box.n_azimuthal_modes - 1
            data = np.zeros( (ncomp, box.npoints[0], box.npoints[1]) )
            if comp_name == 'E_real':
                data[0,:,:] = array[0,:,:].real
                for mode in range(1,box.n_azimuthal_modes):
                    # Real part of the mode
                    data[2*m-1,:,:] = array[m,:,:].real + array[-m,:,:].real
                    # Imaginary part of the mode
                    data[2*m,:,:] = array[m,:,:].imag - array[-m,:,:].imag
            elif comp_name == 'E_imag':
                data[0,:,:] = array[0,:,:].imag
                for m in range(1,ncomp):
                    # Real part of the mode
                    data[2*m-1,:,:] = array[m,:,:].imag + array[-m,:,:].imag
                    # Imaginary part of the mode
                    data[2*m,:,:] = -array[m,:,:].real + array[-m,:,:].real

        E.store_chunk( data )

    series.flush()
