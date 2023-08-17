import numpy as np
from scipy.constants import c

from axiprop.lib import PropagatorFFT2, PropagatorResampling

from lasy.utils.grid import Grid
from lasy.utils.laser_utils import (
    normalize_energy,
    normalize_peak_field_amplitude,
    normalize_peak_intensity,
)
from lasy.utils.openpmd_output import write_to_openpmd_file


class Laser:
    """
    Evaluate a laser profile on a grid, propagate it, and write it to a file.

    This is a top-level class.

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
    >>> from lasy.utils.laser_utils import get_full_field
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
    >>>     E_rt, extent = get_full_field(laser)
    >>>     extent[2:] *= 1e6
    >>>     extent[:2] *= 1e12
    >>>     tmin, tmax, rmin, rmax = extent
    >>>     vmax = np.abs(E_rt).max()
    >>>     axes[step].imshow(
    ...         E_rt,
    ...         origin="lower",
    ...         aspect="auto",
    ...         vmax=vmax,
    ...         vmin=-vmax,
    ...         extent=[tmin, tmax, rmin, rmax],
    ...         cmap='bwr',
    ...     )
    >>>     axes[step].set(xlabel='t (ps)')
    >>>     if step == 0:
    >>>         axes[step].set(ylabel='r (µm)')
    """

    def __init__(self, dim, lo, hi, npoints, profile, n_azimuthal_modes=1):
        self.grid = Grid(dim, lo, hi, npoints, n_azimuthal_modes)
        self.dim = dim
        self.profile = profile

        # Create the grid on which to evaluate the laser, evaluate it
        if self.dim == "xyt":
            x, y, t = np.meshgrid(*self.grid.axes, indexing="ij")
            self.grid.field[...] = profile.evaluate(x, y, t)
        elif self.dim == "rt":
            # Generate 2*n_azimuthal_modes - 1 evenly-spaced values of
            # theta, to evaluate the laser
            n_theta = 2 * self.grid.n_azimuthal_modes - 1
            theta1d = 2 * np.pi / n_theta * np.arange(n_theta)
            theta, r, t = np.meshgrid(theta1d, *self.grid.axes, indexing="ij")
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # Evaluate the profile on the generated grid
            envelope = profile.evaluate(x, y, t)
            # Perform the azimuthal decomposition
            self.grid.field[...] = np.fft.ifft(envelope, axis=0)

        # For profiles that define the energy, normalize the amplitude
        if hasattr(profile, "laser_energy"):
            self.normalize(profile.laser_energy, kind="energy")

    def normalize(self, value, kind="energy"):
        """
        Normalize the pulse either to the energy, peak field amplitude or peak intensity.

        Parameters
        ----------
        value: scalar
            Value to which to normalize the field property that is defined in ``kind``
        kind: string (optional)
            Distance by which the laser pulse should be propagated
            Options: ``'energy``', ``'field'``, ``'intensity'`` (default is ``'energy'``)
        """
        if kind == "energy":
            normalize_energy(self.dim, value, self.grid)
        elif kind == "field":
            normalize_peak_field_amplitude(value, self.grid)
        elif kind == "intensity":
            normalize_peak_intensity(value, self.grid)
        else:
            raise ValueError(f'kind "{kind}" not recognized')

    def propagate(self, distance, nr_boundary=None, backend="NP"):
        """
        Propagate the laser pulse by the distance specified.

        Parameters
        ----------
        distance : scalar
            Distance by which the laser pulse should be propagated

        nr_boundary : integer (optional)
            Number of cells at the end of radial axis, where the field
            will be attenuated (to assert proper Hankel transform).
            Only used for ``'rt'``.
        backend : string (optional)
            Backend used by axiprop (see axiprop documentation).
        """
        time_axis_indx = -1

        # apply boundary "absorption" if required
        if nr_boundary is not None:
            assert type(nr_boundary) is int and nr_boundary > 0
            absorb_layer_axis = np.linspace(0, np.pi / 2, nr_boundary)
            absorb_layer_shape = np.cos(absorb_layer_axis) ** 0.5
            absorb_layer_shape[-1] = 0.0
            if self.dim == "rt":
                self.grid.field[:, -nr_boundary:, :] *= absorb_layer_shape[
                    None, :, None
                ]
            else:
                self.grid.field[-nr_boundary:, :, :] *= absorb_layer_shape[
                    :, None, None
                ]
                self.grid.field[:nr_boundary, :, :] *= absorb_layer_shape[::-1][
                    :, None, None
                ]
                self.grid.field[:, -nr_boundary:, :] *= absorb_layer_shape[
                    None, :, None
                ]
                self.grid.field[:, :nr_boundary, :] *= absorb_layer_shape[::-1][
                    None, :, None
                ]

        # Transform the field from temporal to frequency domain
        field_fft = np.fft.ifft(self.grid.field, axis=time_axis_indx, norm="backward")

        # Create the frequency axis
        dt = self.grid.dx[time_axis_indx]
        omega0 = self.profile.omega0
        Nt = self.grid.field.shape[time_axis_indx]
        omega = 2 * np.pi * np.fft.fftfreq(Nt, dt) + omega0
        # make 3D shape for the frequency axis
        omega_shape = (1, 1, self.grid.field.shape[time_axis_indx])

        if self.dim == "rt":
            # Construct the propagator (check if exists)
            if not hasattr(self, "prop"):
                spatial_axes = (self.grid.axes[0],)
                self.prop = []
                for m in self.grid.azimuthal_modes:
                    self.prop.append(
                        PropagatorResampling(
                            *spatial_axes,
                            omega / c,
                            mode=m,
                            backend=backend,
                            verbose=False,
                        )
                    )
            # Propagate the spectral image
            for i_m in range(self.grid.azimuthal_modes.size):
                transform_data = np.transpose(field_fft[i_m]).copy()
                self.prop[i_m].step(transform_data, distance, overwrite=True)
                field_fft[i_m, :, :] = np.transpose(transform_data).copy()
        else:
            # Construct the propagator (check if exists)
            if not hasattr(self, "prop"):
                Nx, Ny, Nt = self.grid.field.shape
                Lx = self.grid.hi[0] - self.grid.lo[0]
                Ly = self.grid.hi[1] - self.grid.lo[1]
                spatial_axes = ((Lx, Nx), (Ly, Ny))
                self.prop = PropagatorFFT2(
                    *spatial_axes,
                    omega / c,
                    backend=backend,
                    verbose=False,
                )
            # Propagate the spectral image
            transform_data = np.transpose(field_fft).copy()
            self.prop.step(transform_data, distance, overwrite=True)
            field_fft[:, :, :] = np.transpose(transform_data).copy()

        # Choose the time translation assuming propagation at v=c
        translate_time = distance / c

        # Translate the domain
        self.grid.lo[time_axis_indx] += translate_time
        self.grid.hi[time_axis_indx] += translate_time
        self.grid.axes[time_axis_indx] += translate_time

        # Translate the phase of spectral image
        field_fft *= np.exp(-1j * translate_time * omega.reshape(omega_shape))

        # Transform field from frequency to temporal domain
        self.grid.field[:, :, :] = np.fft.fft(
            field_fft, axis=time_axis_indx, norm="backward"
        )

        # Translate phase of the retrieved envelope by the distance
        self.grid.field *= np.exp(1j * omega0 * distance / c)

    def write_to_file(
        self, file_prefix="laser", file_format="h5", save_as_vector_potential=False
    ):
        """
        Write the laser profile + metadata to file.

        Parameters
        ----------
        file_prefix : string
            The file name will start with this prefix.

        file_format : string
            Format to be used for the output file. Options are ``"h5"`` and ``"bp"``.

        save_as_vector_potential : bool (optional)
            Whether the envelope is converted to normalized vector potential
            before writing to file.
        """
        write_to_openpmd_file(
            self.dim,
            file_prefix,
            file_format,
            self.grid,
            self.profile.lambda0,
            self.profile.pol,
            save_as_vector_potential,
        )
