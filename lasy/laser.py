import numpy as np
import scipy.constants as scc

from lasy.utils.box import Box
from lasy.utils.grid import Grid


class Laser:
    """
    Base class for laser profiles.
    The laser pulse is assumed to propagate in the z direction.
    """

    def __init__(self, dim, lo, hi, array_in, wavelength, pol):
        """
        Construct Laser class from specified arrays

        Parameters
        ----------
        dim: string
            Dimension of the array, 'rz' or 'xyz'

        lo: list of scalars
            Lower end of the physical domain where the laser array is defined

        hi: list of scalars
            Higher end of the physical domain where the laser array is defined

        array_in: numpy complex array
            n-dimensional (n=2 for dim='rz', n=3 for dim='xyz') array with laser field
            The array should contain the complex envelope of the electric field.
            The magnetic field is assumed orthogonal to the electric field, with the same profile
            and a magnitude c times lower.

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

        self.ndims = 2 if dim == 'rz' else 3

        assert(dim in ['rz', 'xyz'])
        assert(len(lo) == self.ndims)
        assert(len(hi) == self.ndims)
        assert(array_in.ndim == self.ndims)
        assert(len(pol) == 2)

        self.dim = dim
        norm_pol = np.sqrt(np.abs(pol[0])**2 + np.abs(pol[1])**2)
        self.pol = np.array([pol[0]/norm_pol, pol[1]/norm_pol])
        self.lambda0 = wavelength
        self.omega0 = 2*scc.pi*scc.c/self.lambda0
        box = Box(dim, lo, hi, array_in.shape)
        self.field = Grid(box, array_in=array_in)

    def propagate(self, distance):
        """
        Propagate the laser pulse by the distance specified

        Parameters
        ----------
        distance: scalar
            Distance by which the laser pulse should be propagated
        """

        self.field.box.lo[-1] += distance
        self.field.box.hi[-1] += distance
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

        # As Angel suggested, I would recommend putting the openPMD implementation
        # in a separate file, and not importing openPMD-api here.
        # That should clean up the import and allow for more flexibility
        # TODO: actual dumping to file.
        print("lo = ", self.field.box.lo)
        import openpmd_api as io
        series = io.Series(
            "{}_%05T.{}".format(file_prefix, file_format),
            io.Access.create)
        i = series.iterations[0]
        series.flush()
        print(self.field.field)
        print("The laser was dumped. Check the recycle bin.")
