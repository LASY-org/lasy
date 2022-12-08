import scipy.constants as scc

from lasy.utils.box import Box
from lasy.utils.grid import Grid
from lasy.utils.openpmd_output import write_to_openpmd_file

class Laser:
    """
    Top-level class that can evaluate a laser profile on a grid,
    propagate it, and write it to a file.
    """

    def __init__(self, dim, lo, hi, npoints, profile,
                 n_azimuthal_modes=1 ):
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

        lo, hi : list of scalars
            Lower and higher end of the physical domain of the box.
            One element per direction (2 for dim='rt', 3 for dim='xyt')

        npoints : tuple of int
            Number of points in each direction.
            One element per direction (2 for dim='rt', 3 for dim='xyt')
            For the moment, the lower end is assumed to be (0,0) in rt and (0,0,0) in xyt

        profile: an object of type lasy.laser_profiles.laser_profile.LaserProfile
            Defines how to evaluate the envelope field

        n_azimuthal_modes: int (optional)
            Only used if `dim` is 'rt'. The number of azimuthal modes
            used in order to represent the laser field.
        """
        self.box = Box(dim, lo, hi, npoints, n_azimuthal_modes)
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
