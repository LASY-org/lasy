import numpy as np
import scipy.constants as scc

from lasy.utils.grid import Grid
from lasy.utils.openpmd_output import write_to_openpmd_file

class Laser:
    """
    Base class for laser profiles.
    The laser pulse is assumed to propagate in the z direction.
    """

    def __init__(self, box, wavelength, pol):
        """
        Construct Laser class from specified arrays

        Parameters
        ----------
        dim: string
            Dimensionality of the array. Options are:
            - 'xyt': The laser pulse is represented on a 3D grid:
                     Cartesian (x,y) transversely, and temporal (t) longitudinally.
            - 'rt' : The laser pulse is represented on a 2D grid:
                     Cylindrical (r) transversely, and temporal (t) longitudinally.

        lo: list of scalars
            Lower end of the physical domain where the laser array is defined

        hi: list of scalars
            Higher end of the physical domain where the laser array is defined

        wavelength: scalar
            Central wavelength for which the laser pulse envelope is defined.

        pol: list of 2 complex numbers
            Polarization vector that multiplies array_in to get the Ex and Ey fields.
            The envelope of each component of the electric field is given by:
            - Ex_env = array_in*pol(0)
            - Ey_env = array_in*pol(1)
            Standard polarizations can be obtained from:
            - Linear polarization in x: pol = (1,0)
            - Linear polarization in y: pol = (0,1)
            - Circular polarization: pol = (1,j)/sqrt(2) (j is the imaginary number)
            The polarization vector is normalized to have a unitary magnitude.
        """
        assert(len(pol) == 2)

        self.dim = box.dim
        norm_pol = np.sqrt(np.abs(pol[0])**2 + np.abs(pol[1])**2)
        self.pol = np.array([pol[0]/norm_pol, pol[1]/norm_pol])
        self.lambda0 = wavelength
        self.omega0 = 2*scc.pi*scc.c/self.lambda0
        self.field = Grid(box)

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
                               self.lambda0, self.pol )
