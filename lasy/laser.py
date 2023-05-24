import numpy as np
from scipy.constants import c

from axiprop.lib import PropagatorFFT2, PropagatorResampling
from axiprop.containers import ScalarFieldEnvelope

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
    >>>     E_rt, extent = laser.get_full_field()
    >>>     extent[:2] *= 1e6
    >>>     extent[2:] *= 1e12
    >>>     rmin, rmax, tmin, tmax = extent
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
            normalize_energy(self.dim, value, self.field)
        elif kind == "field":
            normalize_peak_field_amplitude(value, self.field)
        elif kind == "intensity":
            normalize_peak_intensity(value, self.field)
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
                self.field.field[:, -nr_boundary:, :] *= absorb_layer_shape[
                    None, :, None
                ]
            else:
                self.field.field[-nr_boundary:, :, :] *= absorb_layer_shape[
                    :, None, None
                ]
                self.field.field[:nr_boundary, :, :] *= absorb_layer_shape[::-1][
                    :, None, None
                ]
                self.field.field[:, -nr_boundary:, :] *= absorb_layer_shape[
                    None, :, None
                ]
                self.field.field[:, :nr_boundary, :] *= absorb_layer_shape[::-1][
                    None, :, None
                ]

        # Transform the field from temporal to frequency domain
        field_fft = np.fft.ifft(self.field.field, axis=time_axis_indx, norm="backward")

        # Create the frequency axis
        dt = self.box.dx[time_axis_indx]
        omega0 = self.profile.omega0
        Nt = self.field.field.shape[time_axis_indx]
        omega = 2 * np.pi * np.fft.fftfreq(Nt, dt) + omega0
        # make 3D shape for the frequency axis
        omega_shape = (1, 1, self.field.field.shape[time_axis_indx])

        if self.dim == "rt":
            # Construct the propagator (check if exists)
            if not hasattr(self, "prop"):
                spatial_axes = (self.box.axes[0],)
                self.prop = []
                for m in self.box.azimuthal_modes:
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
            for i_m in range(self.box.azimuthal_modes.size):
                transform_data = np.transpose(field_fft[i_m]).copy()
                self.prop[i_m].step(transform_data, distance, overwrite=True)
                field_fft[i_m, :, :] = np.transpose(transform_data).copy()
        else:
            # Construct the propagator (check if exists)
            if not hasattr(self, "prop"):
                Nx, Ny, Nt = self.field.field.shape
                Lx = self.box.hi[0] - self.box.lo[0]
                Ly = self.box.hi[1] - self.box.lo[1]
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
        # Translate the box
        self.box.lo[time_axis_indx] += translate_time
        self.box.hi[time_axis_indx] += translate_time
        self.box.axes[time_axis_indx] += translate_time

        # Translate the phase of spectral image
        field_fft *= np.exp(-1j * translate_time * omega.reshape(omega_shape))

        # Transform field from frequency to temporal domain
        self.field.field[:, :, :] = np.fft.fft(
            field_fft, axis=time_axis_indx, norm="backward"
        )

        # Translate phase of the retrieved envelope by the distance
        self.field.field *= np.exp(1j * omega0 * distance / c)

    def export_to_z(self, z_axis=None, z0=0.0, t0=0.0, backend="NP"):
        """
        Convert laser pulse defined in the temporal domain to the spatial domain.

        Parameters
        ----------
        z_axis : 1D ndarray of doubles (optional)
            Spatial `z` axis along which the field should be reconstructed.
            If not provided, `z_axis = c * t_axis` is considered.

        z0 : scalar (optional)
            Position from which the field is produced (emitted)

        t0 : scalar (optional)
            Moment of time at which the field is produced

        backend : string (optional)
            Backend used by axiprop (see axiprop documentation).
        """
        time_axis_indx = -1

        t_axis = self.field.box.axes[time_axis_indx]
        if z_axis is None:
            z_axis = t_axis * c

        omega0 = self.profile.omega0
        FieldAxprp = ScalarFieldEnvelope(omega0 / c, t_axis)

        if self.dim == "rt":
            # Construct the propagator
            prop = []
            for m in self.box.azimuthal_modes:
                prop.append(
                    PropagatorResampling(
                        self.box.axes[0],
                        FieldAxprp.k_freq,
                        mode=m,
                        backend=backend,
                        verbose=False,
                    )
                )

            field_z = np.zeros(
                (self.field.field.shape[0], self.field.field.shape[1], z_axis.size),
                dtype=self.field.field.dtype,
            )

            # Convert the spectral image to the spatial field representation
            for i_m in range(self.box.azimuthal_modes.size):
                FieldAxprp.import_field(np.transpose(self.field.field[i_m]).copy())

                field_z[i_m] = (
                    prop[i_m].t2z(FieldAxprp.Field_ft, z_axis, z0=z0, t0=t0).T
                )

                field_z[i_m] *= np.exp(-1j * (z_axis / c + t0) * omega0)
        else:
            # Construct the propagator
            Nx, Ny, Nt = self.field.field.shape
            Lx = self.box.hi[0] - self.box.lo[0]
            Ly = self.box.hi[1] - self.box.lo[1]
            prop = PropagatorFFT2(
                (Lx, Nx),
                (Ly, Ny),
                FieldAxprp.k_freq,
                backend=backend,
                verbose=False,
            )
            # Convert the spectral image to the spatial field representation
            FieldAxprp.import_field(np.transpose(self.field.field).copy())
            field_z = prop.t2z(FieldAxprp.Field_ft, z_axis, z0=z0, t0=t0).T
            field_z *= np.exp(-1j * (z_axis / c + t0) * omega0)

        return field_z

    def import_from_z(self, field_z, z_axis, z0=0.0, t0=0.0, backend="NP"):
        """
        Export laser pulse from the field map in the spatial domain.

        Parameters
        ----------
        z_axis : 1D ndarray of doubles
            Spatial `z` axis along which the field should be reconstructed.

        z0 : scalar (optional)
            Position at which the field should be recorded

        t0 : scalar (optional)
            Moment of time at which the field should be recorded

        backend : string (optional)
            Backend used by axiprop (see axiprop documentation).
        """
        z_axis_indx = -1
        t_axis = self.field.box.axes[z_axis_indx]
        dz = z_axis[1] - z_axis[0]
        Nz = z_axis.size

        # Transform the field from spatial to wavenumber domain
        field_fft = np.fft.fft(field_z, axis=z_axis_indx, norm="forward")

        # Create the axes for wavenumbers, and for corresponding frequency
        omega0 = self.profile.omega0
        omega = 2 * np.pi * np.fft.fftfreq(Nz, dz / c) + omega0
        k_z = omega / c

        if self.dim == "rt":
            # Construct the propagator
            prop = []
            for m in self.box.azimuthal_modes:
                prop.append(
                    PropagatorResampling(
                        self.box.axes[0],
                        omega / c,
                        mode=m,
                        backend=backend,
                        verbose=False,
                    )
                )

            # Convert the spectral image to the spatial field representation
            for i_m in range(self.box.azimuthal_modes.size):
                transform_data = np.transpose(field_fft[i_m]).copy()
                transform_data *= np.exp(-1j * z_axis[0] * (k_z[:, None] - omega0 / c))
                self.field.field[i_m] = (
                    prop[i_m].z2t(transform_data, t_axis, z0=z0, t0=t0).T
                )
                self.field.field[i_m] *= np.exp(1j * (z0 / c + t_axis) * omega0)
        else:
            # Construct the propagator
            Nx, Ny, Nt = self.field.field.shape
            Lx = self.box.hi[0] - self.box.lo[0]
            Ly = self.box.hi[1] - self.box.lo[1]
            prop = PropagatorFFT2(
                (Lx, Nx),
                (Ly, Ny),
                omega / c,
                backend=backend,
                verbose=False,
            )
            # Convert the spectral image to the spatial field representation
            transform_data = np.transpose(field_fft).copy()
            transform_data *= np.exp(
                -1j * z_axis[0] * (k_z[:, None, None] - omega0 / c)
            )
            self.field.field = prop.z2t(transform_data, t_axis, z0=z0, t0=t0).T
            self.field.field *= np.exp(1j * (z0 / c + t_axis) * omega0)

    def write_to_file(self, file_prefix="laser", file_format="h5"):
        """
        Write the laser profile + metadata to file.

        Parameters
        ----------
        file_prefix : string
            The file name will start with this prefix.

        file_format : string
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
