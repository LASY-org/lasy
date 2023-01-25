import numpy as np
import scipy.constants as scc

from lasy.utils.box import Box
from lasy.utils.grid import Grid
from lasy.utils.openpmd_output import write_to_openpmd_file
from lasy.utils.laser_utils import normalize_energy, normalize_peak_field_amplitude, normalize_peak_intensity

from axiprop.lib import PropagatorResampling, PropagatorFFT2
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

    def time_to_frequency(self):
        """
        Transform field from the temporal to the frequency domain via FFT,
        and create the frequency axis if necessary
        """
        times_axis = {'rt': 1, 'xyt': 0}[self.dim]
        self.field.field_fft = np.fft.fft(self.field.field,
                                          axis=times_axis,
                                          norm="forward")
        try:
            self.field.omega;
        except:
            dt = self.box.dx[0]
            omega0 = self.profile.omega0
            Nt = self.field.field.shape[times_axis]
            self.field.omega = 2 * np.pi * np.fft.fftfreq(Nt, dt) + omega0

    def frequency_to_time(self):
        """
        Transform field from the frequency to the temporal domain via iFFT
        """
        times_axis = {'rt': 1, 'xyt': 0}[self.dim]
        self.field.field = np.fft.ifft(self.field.field_fft,
                                       axis=times_axis,
                                       norm="forward")

    def translate_spectral(self, translate_time):
        """
        Add the phase corresponding to the time-axis translation by

        Parameters
        ----------
        translate_time: float (m)
            Time interval by which the time axis of the field should be
            translated. Note, that after `time_to_frequency()` call the
            time region for `get_full_field()` is set to `[0, Tmax-Tmin]`.
        """

        if self.dim == 'rt':
            Nt = self.field.field.shape[1]
            omega_shape = (1, Nt, 1)
        elif self.dim == 'xyt':
            Nt = self.field.field.shape[0]
            omega_shape = (Nt, 1, 1)

        self.field.field_fft *= np.exp(-1j * translate_time
                                * self.field.omega.reshape(omega_shape))

    def propagate(self, distance, nr_boundary=16):
        """
        Propagate the laser pulse by the distance specified

        Parameters
        ----------
        distance: scalar
            Distance by which the laser pulse should be propagated

        nr_boundary: integer (optional for 'rt')
            Number of cells at the end of radial axis, where the field
            will be attenuated (to assert proper Hankel transform)
        """
        if self.dim == 'rt':
            Propagator = PropagatorResampling
            spatial_axes = (self.box.axes[1],)
            # apply the boundary "absorbtion"
            absorb_layer_axis = np.r_[0 : np.pi/2 : nr_boundary*1j]
            absorb_layer_shape = np.cos(absorb_layer_axis)**0.5
            absorb_layer_shape[-1] = 0.0
            self.field.field[..., -nr_boundary:] *= absorb_layer_shape
        elif self.dim == 'xyt':
            Nt, Nx, Ny = self.field.field.shape
            Lx = self.box.hi[1] - self.box.lo[1]
            Ly = self.box.hi[2] - self.box.lo[2]
            Propagator = PropagatorFFT2
            spatial_axes = ((Lx, Nx), (Ly, Ny))

        self.time_to_frequency()

        try:
            self.prop;
        except:
            if self.dim == 'rt':
                azimuthal_modes = np.r_[
                    np.arange(self.box.n_azimuthal_modes),
                    np.arange(-self.box.n_azimuthal_modes+1, 0, 1) ]

                self.prop = [Propagator(*spatial_axes, self.field.omega/scc.c,
                                         mode=m) for m in azimuthal_modes]
            elif self.dim == 'xyt':
                self.prop = Propagator(*spatial_axes, self.field.omega/scc.c)

        if self.dim == 'rt':
            for m in range(self.field.field_fft.shape[0]):
                self.field.field_fft[m] = self.prop[m].step(
                        self.field.field_fft[m], distance, overwrite=True)
        elif self.dim == 'xyt':
            self.field.field_fft = self.prop.step(self.field.field_fft,
                                                  distance, overwrite=True)

        self.translate_spectral(distance / scc.c)
        self.frequency_to_time()

    def get_full_field(self, T_range, dt_new=None):
        """
        Reconstruct the laser pulse with carrier frequency using DFT

        Parameters
        ----------
        T_range: list or tuple (Tmin, Tmax) with Tmin, Tmax floats (s)
            Time interval in which the field should be reconstructed

        dt_new: float (s) (optional)
            Size of the step that is used to resolve T_range. Default `None`
            corresponds to the step of 1/24 of the optical cycle of the
            carrier frequency `omega0`

        Returns:
        --------
            Et: ndarray (V/m)
                The reconstructed field of the shape (Nt_new, Nr) (for `rt`)
                of (Nt_new, Nx) (for `xyt`), with `Nt_new=(Tmax-Tmin)/dt_new`
            extent: ndarray (Tmin, Tmax, Xmin, Xmax)
                Physical extent of the reconstructed field
        """
        try:
            self.field.field_fft;
        except:
            self.time_to_frequency()

        Tmin_box = self.field.box.lo[0]
        self.translate_spectral(Tmin_box)
        extent = np.r_[T_range, [self.box.lo[1], self.box.hi[1]]]

        if dt_new is None:
            dt_new = 2*np.pi / self.profile.omega0 / 24

        tt = np.arange(*T_range, dt_new)
        Et = np.zeros((tt.size, self.field.field.shape[-1]))

        if self.dim == 'rt':
            for m in range(self.field.field.shape[0]):
                Et = get_temporal_radial(self.field.field_fft[m],
                                         Et, tt, self.field.omega/scc.c)
        elif self.dim == 'xyt':
            Et = get_temporal_slice2d(self.field.field_fft,
                                      Et, tt, self.field.omega/scc.c)
        # restore initial field_fft
        self.translate_spectral(-Tmin_box)
        return Et, extent

    def propagate_mimic(self, distance):
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
