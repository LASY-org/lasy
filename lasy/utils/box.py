import numpy as np


class Box:
    """
    Contain metadata on index and physical space for an array,
    as well as handy methods.
    """

    def __init__(self, dim, lo, hi, npoints, n_azimuthal_modes):
        """
        Initialize a Box object

        Parameters
        ----------
        dim: string
            Dimensionality of the array. Options are:
            - 'xyt': The laser pulse is represented on a 3D grid:
                     Cartesian (x,y) transversely, and temporal (t) longitudinally.
            - 'rt' : The laser pulse is represented on a 2D grid:
                     Cylindrical (r) transversely, and temporal (t) longitudinally.

        lo, hi : list of scalars
            Lower and higher end of the physical domain of the box.
            One element per direction (2 for dim='rt', 3 for dim='xyt')

        npoints : tuple of int
            Number of points in each direction.
            One element per direction (2 for dim='rt', 3 for dim='xyt')
            For the moment, the lower end is assumed to be (0,0) in rt and (0,0,0) in xyt

        n_azimuthal_modes: int (optional)
            Only used if `dim` is 'rt'. The number of azimuthal modes
            used in order to represent the laser field.
        """
        ndims = 2 if dim == 'rt' else 3
        assert(dim in ['rt', 'xyt'])
        assert(len(lo) == ndims)
        assert(len(hi) == ndims)

        self.lo = []
        self.hi = []
        self.npoints = []
        self.axes = []
        self.dx = []
        # Loop through coordinates, starting with time (index -1, i.e. last
        # element in `lo`, `hi`, etc.) and then continuing with x and y
        # in 3D Cartesian (index 0 and 1) or with r in cylindrical (index 0)
        # This is in order to make time the slowest-varying variable
        # throughout the code (i.e. first variable, in C-order arrays)
        if dim == 'xyt':
            coords = [-1, 0, 1]
        else:
            coords = [-1, 0]
        for i in coords:
            axis = np.linspace(lo[i], hi[i], npoints[i])
            self.axes.append(axis)
            self.dx.append(axis[1] - axis[0])
            self.npoints.append( npoints[i] )
            self.lo.append( lo[i] )
            self.hi.append( hi[i] )

        if dim == 'rt':
            self.n_azimuthal_modes = n_azimuthal_modes
            self.azimuthal_modes = np.r_[
                np.arange(n_azimuthal_modes),
                np.arange(-n_azimuthal_modes+1, 0, 1) ]
