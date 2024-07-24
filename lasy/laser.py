import numpy as np
from axiprop.lib import PropagatorFFT2, PropagatorResampling
from scipy.constants import c

from lasy.utils.grid import Grid, time_axis_indx
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

    n_theta_evals : int (optional)
        Only used if ``dim`` is ``'rt'``. The number of points in the theta
        (azimuthal) direction at which to evaluate the laser field, before
        decomposing it into ``n_azimuthal_modes`` azimuthal modes. By default,
        this is set to ``2*n_azimuthal_modes - 1``. However, for highly asymmetrical
        profiles, it may be necessary to increase this number.

        For instance, using ``n_theta_evals=20`` and ``n_azimuthal_modes=1``
        will evaluate the laser field at 20 points in the azimuthal direction
        and then average the values to extract the amplitude of the azimuthal mode 0.

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

    def __init__(
        self, dim, lo, hi, npoints, profile, n_azimuthal_modes=1, n_theta_evals=None
    ):
        self.grid = Grid(dim, lo, hi, npoints, n_azimuthal_modes)
        self.dim = dim
        self.profile = profile
        self.output_iteration = 0  # Incremented each time write_to_file is called

        # Create the grid on which to evaluate the laser, evaluate it
        if self.dim == "xyt":
            x, y, t = np.meshgrid(*self.grid.axes, indexing="ij")
            self.grid.set_temporal_field(profile.evaluate(x, y, t))
        elif self.dim == "rt":
            if n_theta_evals is None:
                # Generate 2*n_azimuthal_modes - 1 evenly-spaced values of
                # theta, to evaluate the laser
                n_theta_evals = 2 * self.grid.n_azimuthal_modes - 1
            # Make sure that there are enough points to resolve the azimuthal modes
            assert n_theta_evals >= 2 * self.grid.n_azimuthal_modes - 1
            theta1d = 2 * np.pi / n_theta_evals * np.arange(n_theta_evals)
            theta, r, t = np.meshgrid(theta1d, *self.grid.axes, indexing="ij")
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # Evaluate the profile on the generated grid
            envelope = profile.evaluate(x, y, t)
            # Perform the azimuthal decomposition
            azimuthal_modes = np.fft.ifft(envelope, axis=0)
            field = azimuthal_modes[:n_azimuthal_modes]
            if n_azimuthal_modes > 1:
                field = np.concatenate(
                    (field, azimuthal_modes[-n_azimuthal_modes + 1 :])
                )
            self.grid.set_temporal_field(field)

        # For profiles that define the energy, normalize the amplitude
        if hasattr(profile, "laser_energy"):
            self.normalize(profile.laser_energy, kind="energy")

    def normalize(self, value, kind="energy"):
        """
        Normalize the pulse either to the energy, peak field amplitude or peak intensity.

        Parameters
        ----------
        value : scalar
            Value to which to normalize the field property that is defined in ``kind``
        kind : string (optional)
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

    def apply_optics(self, optical_element):
        """
        Propagate the laser pulse through a thin optical element.

        Parameters
        ----------
        optical_element: an :class:`.OpticalElement` object (optional)
            Represents a thin optical element, through which the laser
            propagates.
        """
        # Create the frequency axis
        dt = self.grid.dx[time_axis_indx]
        omega0 = self.profile.omega0
        Nt = self.grid.shape[time_axis_indx]
        omega_1d = 2 * np.pi * np.fft.fftfreq(Nt, dt) + omega0

        # Apply optical element
        spectral_field = self.grid.get_spectral_field()
        if self.dim == "rt":
            r, omega = np.meshgrid(self.grid.axes[0], omega_1d, indexing="ij")
            # The line below assumes that amplitude_multiplier
            # is cylindrically symmetric, hence we pass
            # `r` as `x` and 0 as `y`
            multiplier = optical_element.amplitude_multiplier(r, 0, omega, omega0)
            # The azimuthal modes are the components of the Fourier transform
            # along theta (FT_theta). Because the multiplier is assumed to be
            # cylindrically symmetric (i.e. theta-independent):
            # FT_theta[ multiplier * field ] = multiplier * FT_theta[ field ]
            # Thus, we can simply multiply each azimuthal mode by the multiplier.
            for i_m in range(self.grid.azimuthal_modes.size):
                spectral_field[i_m, :, :] *= multiplier
        else:
            x, y, omega = np.meshgrid(
                self.grid.axes[0], self.grid.axes[1], omega_1d, indexing="ij"
            )
            spectral_field *= optical_element.amplitude_multiplier(x, y, omega, omega0)
        self.grid.set_spectral_field(spectral_field)

    def propagate(self, distance, nr_boundary=None, backend="NP", show_progress=True):
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
        show_progress : bool (optional)
            Whether to show a progress bar when performing the computation
        """
        # apply boundary "absorption" if required
        if nr_boundary is not None:
            assert type(nr_boundary) is int and nr_boundary > 0
            absorb_layer_axis = np.linspace(0, np.pi / 2, nr_boundary)
            absorb_layer_shape = np.cos(absorb_layer_axis) ** 0.5
            absorb_layer_shape[-1] = 0.0
            field = self.grid.get_temporal_field()
            if self.dim == "rt":
                field[:, -nr_boundary:, :] *= absorb_layer_shape[None, :, None]
            else:
                field[-nr_boundary:, :, :] *= absorb_layer_shape[:, None, None]
                field[:nr_boundary, :, :] *= absorb_layer_shape[::-1][:, None, None]
                field[:, -nr_boundary:, :] *= absorb_layer_shape[None, :, None]
                field[:, :nr_boundary, :] *= absorb_layer_shape[::-1][None, :, None]
            self.grid.set_temporal_field(field)

        # Create the frequency axis
        dt = self.grid.dx[time_axis_indx]
        omega0 = self.profile.omega0
        Nt = self.grid.shape[time_axis_indx]
        omega = 2 * np.pi * np.fft.fftfreq(Nt, dt) + omega0

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
            spectral_field = self.grid.get_spectral_field()
            for i_m in range(self.grid.azimuthal_modes.size):
                transform_data = np.transpose(spectral_field[i_m]).copy()
                self.prop[i_m].step(
                    transform_data,
                    distance,
                    overwrite=True,
                    show_progress=show_progress,
                )
                spectral_field[i_m, :, :] = np.transpose(transform_data).copy()
            self.grid.set_spectral_field(spectral_field)
        else:
            # Construct the propagator (check if exists)
            if not hasattr(self, "prop"):
                Nx, Ny, Nt = self.grid.shape
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
            spectral_field = self.grid.get_spectral_field()
            transform_data = np.moveaxis(spectral_field, -1, 0).copy()
            self.prop.step(
                transform_data, distance, overwrite=True, show_progress=show_progress
            )
            spectral_field = np.moveaxis(transform_data, 0, -1).copy()

        # Choose the time translation assuming propagation at v=c
        translate_time = distance / c

        # This translation (e.g. delay in time, compared to t=0, associated
        # with the propagation) is not automatically handled by the above
        # propagators, so it needs to be added by hand.
        # Note: subtracting by omega0 is only a global phase convention,
        # that derives from the definition of the envelope in lasy.
        spectral_field *= np.exp(-1j * (omega[None, None, :] - omega0) * translate_time)
        self.grid.set_spectral_field(spectral_field)

        # Translate the domain
        self.grid.lo[time_axis_indx] += translate_time
        self.grid.hi[time_axis_indx] += translate_time
        self.grid.axes[time_axis_indx] += translate_time

    def write_to_file(
        self,
        file_prefix="laser",
        file_format="h5",
        write_dir="diags",
        save_as_vector_potential=False,
    ):
        """
        Write the laser profile + metadata to file.

        Parameters
        ----------
        write_dir : string
            The directory where the file will be written.

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
            write_dir,
            file_prefix,
            file_format,
            self.output_iteration,
            self.grid,
            self.profile.lambda0,
            self.profile.pol,
            save_as_vector_potential,
        )
        self.output_iteration += 1

    def show(self, **kw):
        """
        Show a 2D image of the laser amplitude.

        Parameters
        ----------
        **kw : additional arguments to be passed to matplotlib's imshow command
        """
        temporal_field = self.grid.get_temporal_field()
        if self.dim == "rt":
            # Show field in the plane y=0, above and below axis, with proper sign for each mode
            E = [
                np.concatenate(
                    ((-1.0) ** m * temporal_field[m, ::-1], temporal_field[m])
                )
                for m in self.grid.azimuthal_modes
            ]
            E = sum(E)  # Sum all the modes
            extent = [
                self.grid.lo[-1],
                self.grid.hi[-1],
                -self.grid.hi[0],
                self.grid.hi[0],
            ]

        else:
            # In 3D show an image in the xt plane
            i_slice = int(temporal_field.shape[1] // 2)
            E = temporal_field[:, i_slice, :]
            extent = [
                self.grid.lo[-1],
                self.grid.hi[-1],
                self.grid.lo[0],
                self.grid.hi[0],
            ]

        import matplotlib.pyplot as plt

        plt.imshow(abs(E), extent=extent, aspect="auto", origin="lower", **kw)
        cb = plt.colorbar()
        cb.set_label("$|E_{envelope}|$ (V/m)")
        plt.xlabel("t (s)")
        plt.ylabel("x (m)")
