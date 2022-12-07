import scipy.constants as scc

from lasy.utils.box import Box
from lasy.utils.grid import Grid
from lasy.utils.openpmd_output import write_to_openpmd_file

class Laser:
    """
    Top-level class that can evaluate a laser profile on a grid,
    propagate it, and write it to a file.
    """

    def __init__(self, dim, npoints, profile, tlim, xlim=None, ylim=None, rlim=None):
        """
        Construct a laser object

        Parameters
        ----------
        dim: string
            Dimensionality of the array. Options are:
            - 'xyt': The laser pulse is represented on a 3D grid:
                     Cartesian (x,y) transversely, and temporal (t) longitudinally.
            - 'rt' : The laser pulse is represented on a 2D grid:
                     Cylindrical (r) transversely, and temporal (t) longitudinally.

        npoints : tuple of int
            Number of points in each direction.
            One element per direction (2 for dim='rt', 3 for dim='xyt')
            For the moment, the lower end is assumed to be (0,0) in rt and (0,0,0) in xyt

        profile: an object of type lasy.laser_profiles.laser_profile.LaserProfile
            Defines how to evaluate the envelope field

        tlim : list of 2 scalars
            Lower and higher end of the physical box in the temporal direction.

        xlim : list of 2 scalars
            Lower and higher end of the physical box in the x direction.
            Required for dim='xyt'.

        ylim : list of 2 scalars
            Lower and higher end of the physical box in the x direction.
            Required for dim='xyt'.

        rlim : list of 2 scalars
            Lower and higher end of the physical box in the radial.
            Required for dim='rt'.
        """

        assert dim in ['rt', 'xyt']

        if dim == 'rt':
            assert(rlim is not None)
            lo = (rlim[0], tlim[0])
            hi = (rlim[1], tlim[1])
        else dim == 'xyt':
            assert(xlim is not None and ylim is not None)            
            lo = (xlim[0], ylim[0], tlim[0])
            hi = (xlim[1], ylim[1], tlim[1])
            
        self.box = Box(dim, lo, hi, npoints)
        self.field = Grid(self.box)
        self.dim = self.box.dim
        self.profile = profile

        # Evaluate the laser profile on the grid
        profile.evaluate( self.field.field, self.box )

    def propagate(self, distance):
        """
        Propagate the laser pulse by the distance specified

        Parameters
        ----------
        distance: scalar
            Distance by which the laser pulse should be propagated
        """

        self.field.box.lo[-1] += distance/scc.c
        self.field.box.hi[-1] += distance/scc.c
        # This mimics a laser pulse propagating rigidly.
        # TODO: actual propagation.

    def write_to_file(self, file_prefix="laser", file_format='h5'):
        """
        Write the laser profile + metadata to file.

        Parameters
        ----------
        file_prefix: string
            The file name will start with this prefix.

        file_format: string
            Format to be used for the output file. Options are "h5" and "bp".
        """
        write_to_openpmd_file( file_prefix, file_format,
                               self.field.box, self.dim, self.field.field,
                               self.profile.lambda0, self.profile.pol )
