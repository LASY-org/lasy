import numpy as np
import scipy.constants as scc

from lasy.utils.box import Box
from lasy.utils.grid import Grid
from lasy.utils.openpmd_output import write_to_openpmd_file
from lasy.utils.laser_utils import normalize_energy, normalize_peak_field_amplitude, normalize_peak_intensity

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

        profile: an object of type lasy.profiles.profile.Profile
            Defines how to evaluate the envelope field

        n_azimuthal_modes: int (optional)
            Only used if `dim` is 'rt'. The number of azimuthal modes
            used in order to represent the laser field.
        """
        box = Box(dim, lo, hi, npoints, n_azimuthal_modes)
        self.box = box
        self.field = Grid(dim, self.box)
        self.dim = dim
        self.profile = profile

        # Create the grid on which to evaluate the laser, evaluate it
        if self.dim == 'xyt':
            t, x, y = np.meshgrid( *box.axes, indexing='ij')
            self.field.field[...] = profile.evaluate( x, y, t )
        elif self.dim == 'rt':
            # Generate 2*n_azimuthal_modes - 1 evenly-spaced values of
            # theta, to evaluate the laser
            n_theta = 2*box.n_azimuthal_modes - 1
            theta1d = 2*np.pi/n_theta * np.arange(n_theta)
            theta, t, r = np.meshgrid( theta1d, *box.axes, indexing='ij')
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            # Evaluate the profile on the generated grid
            envelope = profile.evaluate( x, y, t )
            # Perform the azimuthal decomposition
            self.field.field[...] = np.fft.ifft(envelope, axis=0)

    def normalize(self, value, kind=None):
        """
        Normalize the pulse either to the energy, peak field amplitude or peak intensity

        Parameters
        ----------
        value: scalar
            Value to which to normalize the field property that is defined in 'kind'
        kind: string
            Distance by which the laser pulse should be propagated
            Options: 'energy', 'field', 'intensity'
        """
        
        if kind == 'energy':
            normalize_energy(value, self.field)
        elif kind == 'field':
            normalize_peak_field_amplitude(value, self.field)
        elif kind == 'intensity':
            normalize_peak_intensity(value, self.field)
        else:
            raise ValueError(f'kind "{kind}" not recognized')


    def propagate(self, distance):
        """
        Propagate the laser pulse by the distance specified

        Parameters
        ----------
        distance: scalar
            Distance by which the laser pulse should be propagated
        """

        self.field.box.lo[0] += distance/scc.c
        self.field.box.hi[0] += distance/scc.c
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
        write_to_openpmd_file(self.dim, file_prefix, file_format, self.field,
                               self.profile.lambda0, self.profile.pol )
