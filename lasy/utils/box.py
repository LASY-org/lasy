import numpy as np


class Box:
    """
    Contain metadata on index and physical space for an array,
    as well as handy methods.

    Parameters
    ----------
    dim : string
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

    n_azimuthal_modes : int (optional)
        Only used if `dim` is 'rt'. The number of azimuthal modes
        used in order to represent the laser field.
    """

    def __init__(self, dim, lo, hi, npoints, n_azimuthal_modes):
        ndims = 2 if dim == "rt" else 3
        assert dim in ["rt", "xyt"]
        assert len(lo) == ndims
        assert len(hi) == ndims

        self.lo = list(lo)
        self.hi = list(hi)
        self.npoints = npoints
        self.axes = []
        self.dx = []
        for i in range(ndims):
            self.axes.append(np.linspace(lo[i], hi[i], npoints[i]))
            self.dx.append(self.axes[i][1] - self.axes[i][0])

        if dim == "rt":
            self.n_azimuthal_modes = n_azimuthal_modes
            self.azimuthal_modes = np.r_[
                np.arange(n_azimuthal_modes), np.arange(-n_azimuthal_modes + 1, 0, 1)
            ]
