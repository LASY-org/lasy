import numpy as np
import scipy.constants as scc

from lasy.utils.box import Box
from lasy.utils.grid import Grid
from lasy.utils.openpmd_output import write_to_openpmd_file
from lasy.utils.laser_energy import normalize_energy

from axiprop.lib import PropagatorSymmetric, PropagatorFFT2, PropagatorResampling
from axiprop.utils import get_temporal_radial, get_temporal_slice2d

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
            x, y, t = np.meshgrid( *box.axes, indexing='ij')
            self.field.field[...] = profile.evaluate( x, y, t )
        elif self.dim == 'rt':
            # Generate 2*n_azimuthal_modes - 1 evenly-spaced values of
            # theta, to evaluate the laser
            n_theta = 2*box.n_azimuthal_modes - 1
            theta1d = 2*np.pi/n_theta * np.arange(n_theta)
            theta, r, t = np.meshgrid( theta1d, *box.axes, indexing='ij')
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            # Evaluate the profile on the generated grid
            envelope = profile.evaluate( x, y, t )
            # Perform the azimuthal decomposition
            self.field.field[...] = np.fft.ifft(envelope, axis=0)

        normalize_energy(self.dim, profile.laser_energy, self.field)

    def propagate(self, distance, nr_boundary=16):
        """
        Propagate the laser pulse by the distance specified

        Parameters
        ----------
        distance: scalar
            Distance by which the laser pulse should be propagated
        """

        dt = self.box.dx[-1]
        omega0 = self.profile.omega0

        if self.dim == 'rt':
            m_azimuthal_mode = 0
            A_local = self.field.field[m_azimuthal_mode].T
            Nt, Nr = A_local.shape
            omega_shape = ( Nt, 1 )
            Rmax = self.box.hi[0]
            Propagator = PropagatorResampling
            spatial_axes = ( self.box.axes[0] + 0.75 * self.box.dx[0], )
            # Propagator = PropagatorSymmetric
            # spatial_axes = ( ( Rmax, Nr ), )
            A_local[:,-nr_boundary:] *= np.cos(np.r_[0:np.pi/2:nr_boundary*1j])**0.5
        elif self.dim == 'xyt':
            A_local = self.field.field.T
            Nt, Nx, Ny = A_local.shape
            omega_shape = ( Nt, 1, 1 )

            Lx = self.box.hi[0] - self.box.lo[0]
            Ly = self.box.hi[1] - self.box.lo[1]
            Propagator = PropagatorFFT2
            spatial_axes = ( (Lx, Nx), (Ly, Ny),)

        A_local = np.fft.fft( A_local, axis=0 )
        omega_axis = 2 * np.pi * np.fft.fftfreq( Nt, dt )  + omega0

        try:
            self.prop;
        except:
            self.prop = Propagator( *spatial_axes, omega_axis/scc.c )

        A_local = self.prop.step( A_local, distance, overwrite=True )
        A_local *= np.exp(-1j * omega_axis.reshape(omega_shape) \
                            *  distance / scc.c)

        A_local = np.fft.ifft( A_local, axis=0 )

        if self.dim == 'rt':
            self.field.field[m_azimuthal_mode] = A_local.T
        elif self.dim == 'xyt':
            self.field.field[:] = A_local.T

    def get_full_field(self, T_range, dt_new=None, dNr=1):
        dt = self.box.dx[-1]
        omega0 = self.profile.omega0
        Tmin = self.field.box.lo[-1]

        if dt_new is None:
            dt_new = 2*np.pi / omega0 / 24

        if self.dim == 'rt':
            m_azimuthal_mode = 0
            A_local = self.field.field[m_azimuthal_mode].T
            Nt = A_local.shape[0]
            Rmax = self.box.hi[0]
            Rmin = 0.0
            omega_shape = ( Nt, 1 )
        elif self.dim == 'xyt':
            A_local = self.field.field.T
            Nt = A_local.shape[0]
            omega_shape = ( Nt, 1, 1 )
            Rmax = self.box.hi[0]
            Rmin = self.box.lo[0]

        A_local = np.fft.fft( A_local, axis=0, norm="forward" )
        A_local = np.fft.fftshift(A_local, axes=0)


        omega_axis = 2 * np.pi * np.fft.fftfreq( Nt, dt )  + omega0
        omega_axis = np.fft.fftshift(omega_axis)

        A_local *= np.exp(-1j * omega_axis.reshape(omega_shape) * Tmin)

        tt = np.arange(*T_range, dt_new)
        Et = np.zeros((tt.size, A_local.shape[1]))

        if self.dim == 'rt':
            Et = get_temporal_radial(A_local[:,::dNr], Et, tt, omega_axis/scc.c)
        elif self.dim == 'xyt':
            Et = get_temporal_slice2d(A_local[:,::dNr], Et, tt, omega_axis/scc.c)

        extent = np.r_[ T_range, [Rmin, Rmax] ]

        return Et, extent

    def propagate_mimic(self, distance):
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
        write_to_openpmd_file(self.dim, file_prefix, file_format, self.field,
                               self.profile.lambda0, self.profile.pol )
