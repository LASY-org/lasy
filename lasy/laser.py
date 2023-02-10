import numpy as np
import scipy.constants as scc
from axiprop.lib import PropagatorFFT2, PropagatorResampling

from lasy.utils.box import Box
from lasy.utils.grid import Grid
from lasy.utils.laser_utils import (
    normalize_energy,
    normalize_peak_field_amplitude,
    normalize_peak_intensity,
)
from lasy.utils.openpmd_output import write_to_openpmd_file


class Laser:
    """
    Top-level class that can evaluate a laser profile on a grid,
    propagate it, and write it to a file.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    lo, hi : list of scalars
        Lower and higher end of the physical domain of the box.
        One element per direction (2 for ``dim='rt'``, 3 for ``dim='xyt'``)

    npoints : tuple of int
        Number of points in each direction.
        One element per direction (2 for ``dim='rt'``, 3 for ``dim='xyt'``)
        For the moment, the lower end is assumed to be (0,0) in rt and (0,0,0) in xyt

    profile : an object of type lasy.profiles.profile.Profile
        Defines how to evaluate the envelope field

    n_azimuthal_modes : int (optional)
        Only used if ``dim`` is ``'rt'``. The number of azimuthal modes
        used in order to represent the laser field.

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> from lasy.laser import Laser
    >>> from lasy.profiles.gaussian_profile import GaussianProfile
    >>> # Create profile.
    >>> profile = GaussianProfile(
    ...     wavelength=0.6e-6,  # m
    ...     pol=(1, 0),
    ...     laser_energy=1.,  # J
    ...     w0=5e-6,  # m
    ...     tau=30e-15,  # s
    ...     t_peak=0.  # s
    ... )
    >>> # Create laser with given profile in `rt` geometry.
    >>> laser = Laser(
    ...     dim="rt",
    ...     lo=(0e-6, -60e-15),
    ...     hi=(10e-6, +60e-15),
    ...     npoints=(50, 400),
    ...     profile=profile
    ... )
    >>> # Propagate and visualize.
    >>> n_steps = 3
    >>> propagate_step = 1e-3
    >>> fig, axes = plt.subplots(1, n_steps, sharey=True)
    >>> for step in range(n_steps):
    >>>     laser.propagate(propagate_step)
    >>>     E_rt, imshow_extent = laser.get_full_field()
    >>>     imshow_extent[:2] *= 1e12
    >>>     imshow_extent[2:] *= 1e6
    >>>     vmax = np.abs(E_rt).max()
    >>>     axes[step].imshow(
    ...         E_rt.T,
    ...         origin="lower",
    ...         aspect="auto",
    ...         vmax=vmax,
    ...         vmin=-vmax,
    ...         extent=imshow_extent,
    ...         cmap='bwr',
    ...     )
    >>>     axes[step].set(xlabel='t (ps)')
    >>>     if step == 0:
    >>>         axes[step].set(ylabel='r (µm)')
    """

    def __init__(self, dim, lo, hi, npoints, profile, n_azimuthal_modes=1):
        box = Box(dim, lo, hi, npoints, n_azimuthal_modes)
        self.box = box
        self.field = Grid(dim, self.box)
        self.dim = dim
        self.profile = profile

        # Create the grid on which to evaluate the laser, evaluate it
        if self.dim == "xyt":
            x, y, t = np.meshgrid(*box.axes, indexing="ij")
            self.field.field[...] = profile.evaluate(x, y, t)
        elif self.dim == "rt":
            # Generate 2*n_azimuthal_modes - 1 evenly-spaced values of
            # theta, to evaluate the laser
            n_theta = 2 * box.n_azimuthal_modes - 1
            theta1d = 2 * np.pi / n_theta * np.arange(n_theta)
            theta, r, t = np.meshgrid(theta1d, *box.axes, indexing="ij")
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # Evaluate the profile on the generated grid
            envelope = profile.evaluate(x, y, t)
            # Perform the azimuthal decomposition
            self.field.field[...] = np.fft.ifft(envelope, axis=0)

    def normalize(self, value, kind="energy"):
        """
        Normalize the pulse either to the energy, peak field amplitude or peak intensity

        Parameters
        ----------
        value: scalar
            Value to which to normalize the field property that is defined in ``kind``
        kind: string (optional)
            Distance by which the laser pulse should be propagated
            Options: ``'energy``', ``'field'``, ``'intensity'`` (default is ``'energy'``)
        """

        if kind == "energy":
            normalize_energy(self.dim, value, self.field)
        elif kind == "field":
            normalize_peak_field_amplitude(value, self.field)
        elif kind == "intensity":
            normalize_peak_intensity(value, self.field)
        else:
            raise ValueError(f'kind "{kind}" not recognized')

    def time_to_frequency(self):
        """
        Transform field from the temporal to the frequency domain via FFT,
        and create the frequency axis if necessary.
        """
        times_axis = {"rt": 1, "xyt": 0}[self.dim]
        self.field.field_fft = np.fft.fft(
            self.field.field, axis=times_axis, norm="forward"
        )

        if not hasattr(self.field, "omega"):
            dt = self.box.dx[0]
            omega0 = self.profile.omega0
            Nt = self.field.field.shape[times_axis]
            self.field.omega = 2 * np.pi * np.fft.fftfreq(Nt, dt) + omega0

    def frequency_to_time(self):
        """
        Transform field from the frequency to the temporal domain via iFFT.
        """
        times_axis = {"rt": 1, "xyt": 0}[self.dim]
        self.field.field = np.fft.ifft(
            self.field.field_fft, axis=times_axis, norm="forward"
        )

    def move_time_window(self, translate_time):
        """
        Translate the ``box`` and phase of ``field_fft`` in time by a given amount.

        Parameters
        ----------
        translate_time: float (s)
            Time interval by which the time temporal definitions of the
            laser should be translated.
        """
        self.box.lo[0] += translate_time
        self.box.hi[0] += translate_time
        self.box.axes[0] += translate_time

        if self.dim == "rt":
            Nt = self.field.field.shape[1]
            omega_shape = (1, Nt, 1)
        elif self.dim == "xyt":
            Nt = self.field.field.shape[0]
            omega_shape = (Nt, 1, 1)

        self.field.field_fft *= np.exp(
            -1j * translate_time * self.field.omega.reshape(omega_shape)
        )

    def propagate(self, distance, nr_boundary=None, backend="NP"):
        """
        Propagate the laser pulse by the distance specified

        Parameters
        ----------
        distance: scalar
            Distance by which the laser pulse should be propagated

        nr_boundary: integer (optional)
            Number of cells at the end of radial axis, where the field
            will be attenuated (to assert proper Hankel transform).
            Only used for ``'rt'``.
        """
        time_axis_indx = {"rt": 1, "xyt": 0}[self.dim]

        # apply boundary "absorption" if required
        if nr_boundary is not None:
            assert type(nr_boundary) is int and nr_boundary > 0
            absorb_layer_axis = np.r_[0 : np.pi / 2 : nr_boundary * 1j]
            absorb_layer_shape = np.cos(absorb_layer_axis) ** 0.5
            absorb_layer_shape[-1] = 0.0
            if self.dim == "rt":
                self.field.field[..., -nr_boundary:] *= absorb_layer_shape
            else:
                self.field.field[:, -nr_boundary:, :] *= absorb_layer_shape[
                    None, :, None
                ]
                self.field.field[:, :, -nr_boundary:] *= absorb_layer_shape[
                    None, None, :
                ]
                self.field.field[:, :nr_boundary, :] *= absorb_layer_shape[::-1][
                    None, :, None
                ]
                self.field.field[:, :, :nr_boundary] *= absorb_layer_shape[::-1][
                    None, None, :
                ]

        # Transform the field from temporal to frequency domain
        self.field.field_fft = np.fft.fft(
            self.field.field, axis=time_axis_indx, norm="forward"
        )

        # Create the frequency axis if necessary
        if not hasattr(self.field, "omega"):
            dt = self.box.dx[0]
            omega0 = self.profile.omega0
            Nt = self.field.field.shape[time_axis_indx]
            self.field.omega = 2 * np.pi * np.fft.fftfreq(Nt, dt) + omega0

        if self.dim == "rt":
            # make 3D shape for the frequency axis
            omega_shape = (1, self.field.field.shape[1], 1)
            # Construct the propagator (check if exists)
            if not hasattr(self, "prop"):
                spatial_axes = (self.box.axes[1],)
                self.prop = []
                for m in self.box.azimuthal_modes:
                    self.prop.append(
                        PropagatorResampling(
                            *spatial_axes,
                            self.field.omega / scc.c,
                            mode=m,
                            backend=backend,
                            verbose=False,
                        )
                    )
            # Propagate the spectral image
            for i_m in range(self.box.azimuthal_modes.size):
                self.field.field_fft[i_m] = self.prop[i_m].step(
                    self.field.field_fft[i_m], distance, overwrite=True
                )
        else:
            # make 3D shape for the frequency axis
            omega_shape = (self.field.field.shape[0], 1, 1)
            # Construct the propagator (check if exists)
            if not hasattr(self, "prop"):
                Nt, Nx, Ny = self.field.field.shape
                Lx = self.box.hi[1] - self.box.lo[1]
                Ly = self.box.hi[2] - self.box.lo[2]
                spatial_axes = ((Lx, Nx), (Ly, Ny))
                self.prop = PropagatorFFT2(
                    *spatial_axes,
                    self.field.omega / scc.c,
                    backend=backend,
                    verbose=False,
                )
            # Propagate the spectral image
            self.field.field_fft = self.prop.step(
                self.field.field_fft, distance, overwrite=True
            )

        # Choose the time translation assuming propagation at v=c
        translate_time = distance / scc.c
        # Translate the box
        self.box.lo[0] += translate_time
        self.box.hi[0] += translate_time
        self.box.axes[0] += translate_time

        # Translate the phase of spectral image
        self.field.field_fft *= np.exp(
            -1j * translate_time * self.field.omega.reshape(omega_shape)
        )

        # Transform field from frequency to temporal domain
        self.field.field = np.fft.ifft(
            self.field.field_fft, axis=time_axis_indx, norm="forward"
        )

        # Translate phase of the retrieved envelope by the distance
        self.field.field *= np.exp(1j * self.profile.omega0 * distance / scc.c)

    def write_to_file(self, file_prefix="laser", file_format="h5"):
        """
        Write the laser profile + metadata to file.

        Parameters
        ----------
        file_prefix: string
            The file name will start with this prefix.

        file_format: string
            Format to be used for the output file. Options are ``"h5"`` and ``"bp"``.
        """
        write_to_openpmd_file(
            self.dim,
            file_prefix,
            file_format,
            self.field,
            self.profile.lambda0,
            self.profile.pol,
        )

    def get_full_field(self, theta=0, slice=0, slice_axis="x"):
        """
        Reconstruct the laser pulse with carrier frequency on the default grid

        Parameters
        ----------
        theta: float (rad) (optional)
            Azimuthal angle

        slice: float (optional)
            Normalised position of the slice from -0.5 to 0.5

        Returns
        -------
            Et: ndarray (V/m)
                The reconstructed field of the shape (Nt_new, Nr) (for ``'rt'``)
                or (Nt_new, Nx) (for ``'xyt'``)

            extent: ndarray (Tmin, Tmax, Xmin, Xmax)
                Physical extent of the reconstructed field
        """
        omega0 = self.profile.omega0
        field = self.field.field.copy()
        time_axis = self.box.axes[0][:, None]

        if self.dim == "rt":
            azimuthal_phase = np.exp(-1j * self.box.azimuthal_modes * theta)
            field *= azimuthal_phase[:, None, None]
            field = field.sum(0)
        elif slice_axis == "x":
            Nx_middle = field.shape[-2] // 2 - 1
            Nx_slice = int((1 + slice) * Nx_middle)
            field = field[:, Nx_slice, :]
        elif slice_axis == "y":
            Ny_middle = field.shape[-1] // 2 - 1
            Ny_slice = int((1 + slice) * Ny_middle)
            field = field[:, :, Ny_slice]
        else:
            return None

        field *= np.exp(-1j * omega0 * time_axis)
        field = np.real(field)
        ext = np.r_[self.box.lo[0], self.box.hi[0], self.box.lo[1], self.box.hi[1]]

        return field, ext
