from copy import deepcopy

import numpy as np
from scipy.constants import c

from lasy.utils.fft_wrapper import fft

from .propagator import Propagator


class AngularSpectrumPropagator(Propagator):
    r"""
    Class that represents a double FFT propagator using the angular spectrum method.

    The propagated field is calculated in the following method:

    .. math::

        E_\mathrm{propagated} (x,y,\omega) =
        \mathcal{F}_{x,y}\left[\mathcal{F}_{x,y}\left[ E_\mathrm{input}(x,y,\omega) \right]
        \times\exp(i\,n\,\Delta z\,\sqrt{k_z^2-k_x^2-k_y^2}) \right]

    where :math:`E_{i} (x,y,\omega)` is the initial/propagated fields complex field envelope
    and :math:`\mathcal{F}_{x,y}` is the 2D Fourier transform in the transverse (x,y) axes.

    Parameters
    ----------
    omega0 : float (in rad/s)
        The center frequency of the laser field.

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    n : int, float, 1d array or callable, optional
        Refractive index of the medium in which to propagate the laser.
        Can be either a single value if dispersive effects are ignored, a 1d array
        describing the refractive index along the frequency/wavelength axis of the
        laser pulse, or a function of the wavelength (in meters).
        Default value is n=1. to describe propagation in vacuum.
    """

    def __init__(self, omega0, dim, n=1.0):
        super().__init__()
        self.update(dim=dim, omega0=omega0, n=n)

    def update(self, dim, omega0, n):
        r"""
        Initialize or update the propagator if needed.

        Parameters
        ----------
        dim : string
            Dimensionality of the array. Options are:
            - ``'xyt'``: Laser pulse represented on a 3D Cartesian grid.
            - ``'rt'`` : Laser pulse represented on a 2D cylindrical grid.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        n : scalar, 1d array or callable
            Refractive index of the medium in which to propagate the laser.
            Can be either a single value if dispersive effects are ignored, a 1d array
            describing the refractive index along the frequency/wavelength axis of the
            laser pulse, or a function of the wavelength (in meters).
            Default value is n=1. to describe propagation in vacuum.
        """
        dim = dim if dim is not None else self.dim
        assert isinstance(n, (int, float, np.ndarray)) or callable(n)
        assert dim in ["rt", "xyt"]

        self.dim = dim
        self.omega0 = omega0 if omega0 is not None else self.omega0
        self.n = n  # refractive index

    def propagate(self, grid_in, dim=None, omega0=None, distance=None, grid_out=None):
        r"""
        Propagates the laser field in z direction by a given distance using the angular spectrum method.

        Parameters
        ----------
        distance : scalar
            Distance by which the laser is propagated.

        grid_in : Grid
            Grid object containing the laser to propagate.

        dim : string (optional)
            Dimensionality of the array. Options are:
            - ``'xyt'``: Laser pulse represented on a 3D Cartesian grid.
            - ``'rt'`` : Laser pulse represented on a 2D cylindrical grid.

        omega0 : float (optional)
            The main frequency :math:`\omega_0` (in rad.s^-1), which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        grid_out : Grid object (optional)
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.

        Returns
        -------
        Grid object with laser data after propagation.
        """
        assert distance is not None, "Distance must be provided for propagation."

        self.update(omega0=omega0, dim=dim, n=self.n)

        if grid_out is None:
            grid_out = deepcopy(grid_in)

        if self.dim == "rt":
            field, dt = self._propagate_mrt(distance, grid_in)

        else:  # self.dim == "xyt"
            field, dt = self._propagate_xyt(distance, grid_in)

        # update the grid
        grid_out.set_spectral_field(field)
        grid_out.position += distance
        grid_out.axes[-1] += dt
        grid_out.lo[-1] += dt
        grid_out.hi[-1] += dt

        return grid_out

    def _propagate_xyt(self, distance, grid_in):
        # Get the spectral field in the spatial domain
        field, omega = grid_in.get_spectral_field()

        omega += self.omega0
        kz = omega / c

        # get field in k-space and spatial frequency axes
        field_kspace, axes_freq = fft(
            arr_in=field,
            which="transverse",
            axes_in=[grid_in.axes[0], grid_in.axes[1]],
            from_domain="frequency",
        )

        kx = 2 * np.pi * axes_freq[0]
        ky = 2 * np.pi * axes_freq[1]

        # Calculate the refractive index if it is a function of wavelength
        n = self.n(2 * np.pi * c / omega) if callable(self.n) else self.n

        # Calculate the phase shift in k-space
        phase = (
            distance
            * n
            * (kz[None, None, :] ** 2 - kx[:, None, None] ** 2 - ky[None, :, None] ** 2)
            ** 0.5
        )

        # compensate group delay to keep pulse centered in grid
        if np.ndim(n) > 0:
            dndom = np.gradient(n, omega)
            dndom = np.interp(self.omega0, omega, dndom)
            n0 = np.interp(self.omega0, omega, n)
        else:
            dndom = 0
            n0 = n

        v_group = c / (n0 + self.omega0 * dndom)
        gd = distance / v_group

        phase = phase - gd * (omega - self.omega0)[None, None, :]

        # Apply the phase shift to the field in k-space
        field_kspace *= np.exp(1j * phase)

        # Transform back to the spatial domain
        field, _ = fft(
            arr_in=field_kspace,
            which="transverse",
            axes_in=(kx / (2 * np.pi), ky / (2 * np.pi)),
            from_domain="real",
        )

        # calculate time difference between propagation in vacuum and in medium
        dt = distance / v_group - distance / c

        return field, dt

    def _propagate_mrt(self, distance, grid_in):
        print(
            "'rt' geometry not yet supported by AngularSpectrumPropagator, skipping propagation"
        )
        field = grid_in.field
        dt = 0
        return field, dt
