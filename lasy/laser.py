import numpy as np

from lasy.utils.grid import Grid
from lasy.utils.laser_utils import (
    normalize_average_intensity,
    normalize_energy,
    normalize_peak_field_amplitude,
    normalize_peak_fluence,
    normalize_peak_intensity,
    normalize_peak_power,
)
from lasy.utils.openpmd_helper import write_to_openpmd_file
from lasy.utils.plotting import show_laser


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
    ...     laser_energy=1.0,  # J
    ...     w0=5e-6,  # m
    ...     tau=30e-15,  # s
    ...     t_peak=0.0,  # s
    ... )
    >>> # Create laser with given profile in `rt` geometry.
    >>> laser = Laser(
    ...     dim="rt",
    ...     lo=(0e-6, -60e-15),
    ...     hi=(10e-6, +60e-15),
    ...     npoints=(50, 400),
    ...     profile=profile,
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
        self.grid = Grid(
            dim,
            lo,
            hi,
            npoints,
            n_azimuthal_modes,
            is_cw=profile.is_cw,
            is_plane_wave=profile.is_plane_wave,
        )
        self.dim = dim
        self.profile = profile
        self.output_iteration = 0  # Incremented each time write_to_file is called

        # Create the grid on which to evaluate the laser, evaluate it
        if self.dim == "xyt":
            x, y, t = np.meshgrid(*self.grid.axes, indexing="ij")
            self.grid.set_temporal_field(profile.evaluate(x, y, t))
        elif self.dim == "rt":
            profile_rt = profile.dim == "rt" if hasattr(profile, "dim") else False
            if profile_rt:
                r, t = np.meshgrid(*self.grid.axes, indexing="ij")
                field = np.zeros(
                    (2 * self.grid.n_azimuthal_modes - 1, *r.shape), dtype="complex128"
                )
                for mode in range(2 * self.grid.n_azimuthal_modes - 1):
                    field[mode, :, :] = profile.evaluate_mrt(mode, r, t)
            else:
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

        # For profiles that define the energy, peak fluence or peak power, normalize the amplitude
        if hasattr(profile, "laser_energy"):
            self.normalize(profile.laser_energy, kind="energy")
        elif hasattr(profile, "peak_fluence"):
            self.normalize(profile.peak_fluence, kind="peak_fluence")
        elif hasattr(profile, "peak_power"):
            self.normalize(profile.peak_power, kind="peak_power")

    def normalize(self, value, kind="energy"):
        """
        Normalize the pulse either to the energy, peak field amplitude, peak fluence, peak power, peak intensity, or average intensity. The average intensity option operates on the envelope.

        Parameters
        ----------
        value : scalar
            Value to which to normalize the field property that is defined in ``kind``
        kind : string (optional)
            Options: ``'energy``', ``'field'``, ``'intensity'``, ``'average_intensity'``, ``'peak_fluence'``, ``'peak_power'``, (default is ``'energy'``)
        """
        if kind == "energy":
            normalize_energy(self.dim, value, self.grid)
        elif kind == "field":
            normalize_peak_field_amplitude(value, self.grid)
        elif kind == "intensity":
            normalize_peak_intensity(value, self.grid)
        elif kind == "average_intensity":
            normalize_average_intensity(value, self.grid)
        elif kind == "peak_power":
            normalize_peak_power(self.dim, value, self.grid)
        elif kind == "peak_fluence":
            normalize_peak_fluence(value, self.grid)
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
        # Apply optical element
        spectral_field, spectral_axis = self.grid.get_spectral_field()
        if self.dim == "rt":
            r, omega = np.meshgrid(
                self.grid.axes[0], spectral_axis + self.profile.omega0, indexing="ij"
            )
            # The line below assumes that amplitude_multiplier
            # is cylindrically symmetric, hence we pass
            # `r` as `x` and an array of 0s as `y`
            multiplier = optical_element.amplitude_multiplier(
                r, np.zeros_like(r), omega
            )
            # The azimuthal modes are the components of the Fourier transform
            # along theta (FT_theta). Because the multiplier is assumed to be
            # cylindrically symmetric (i.e. theta-independent):
            # FT_theta[ multiplier * field ] = multiplier * FT_theta[ field ]
            # Thus, we can simply multiply each azimuthal mode by the multiplier.
            for i_m in range(self.grid.azimuthal_modes.size):
                spectral_field[i_m, :, :] *= multiplier
        else:
            x, y, omega = np.meshgrid(
                self.grid.axes[0],
                self.grid.axes[1],
                spectral_axis + self.profile.omega0,
                indexing="ij",
            )
            spectral_field *= optical_element.amplitude_multiplier(x, y, omega)
        self.grid.set_spectral_field(spectral_field)

    def add_propagator(self, propagator):
        """
        Apply a propagator object to the laser pulse.

        Parameters
        ----------
        propagator: a :class:`.Propagator` object (optional)
            Represents a propagation method.
        """
        self.propagator = propagator

    # I really would like to avoid these kwargs, such that one can change the
    # propagator without affecting the call to laser.propagate. We'll see if
    # that's reasonable.
    def propagate(self, distance, **kwargs):
        """
        Propagate the laser pulse by the distance specified.

        Parameters
        ----------
        distance : scalar
            Distance by which the laser pulse should be propagated

        grid : Grid object (optional)
            Resample the field onto a new grid of different radial size and/or different number
            of radial grid points. Only works for ``'rt'``.
        """
        if not hasattr(self, "propagator"):
            from lasy.propagators import AxipropPropagator

            propagator = AxipropPropagator()
            self.add_propagator(propagator)

        grid_out = self.propagator.propagate(
            self.grid, self.dim, self.profile.omega0, distance=distance, **kwargs
        )
        self.grid = grid_out

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

    def show(
        self,
        envelope_type="field",
        t_shift=0,
        show_lineout=True,
        show_max=False,
        udict={},
        **kw,
    ):
        r"""
        Show a 2D image of the laser amplitude or intensity.

        Parameters
        ----------
        envelope_type : string, default: "field"
            Options are:
            - ``'field'``: Show the envelope of the laser field.
            - ``'intensity'``: Show the intensity of the laser field.
            - ``'vector_potential'``: Show the vector potential of the laser field.

        t_shift : float, default: 0
            Shift the temporal axis by `t_shift` seconds.
            It also can be a string with `"left"`, `"right"` or `"center"`,
            to shift the temporal axis such that the t=0 lies at the left,
            right or center of the x-axis.

        show_lineout : bool, default: True
            Show the lineout of the laser field.

        show_max : bool, default: False
            Print the maximum intensity of the laser field.

        udict : dict, default: {}
            Dictionary with the information of the unit scales of the axes,
            e.g. ``{'t': {'value': 1e-15, 'label': 'fs'}, 'x': {'value': 1e-6, 'label': r'\mu m'}}``
            Override the default unit scales.

        **kw : additional arguments to be passed to matplotlib's imshow command
        """
        show_laser(
            self.grid,
            self.dim,
            envelope_type=envelope_type,
            t_shift=t_shift,
            show_lineout=show_lineout,
            show_max=show_max,
            udict=udict,
            omega0=self.profile.omega0,
            **kw,
        )
