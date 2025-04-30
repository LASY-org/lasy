import numpy as np
from scipy.constants import c

from lasy.utils.fft_wrapper import fft

from .propagator import Propagator


class AngularSpectrumDFFTPropagator(Propagator):
    r"""
    Class that represents a dual FFT propagator using the angular spectrum method spectrum dual FFT propagator.

    The propagated field is calculated in the following method:

    .. math::

        E_\mathrm{propagated} (x,y,\omega) =
        \mathcal{F}_{x,y}\left[\mathcal{F}_{x,y}\left[ E_\mathrm{input}(x,y,\omega) \right]
        \times\exp(i\,n\,\Delta z\,\sqrt{k_z^2-k_x^2-k_y^2}) \right]

    where :math:`E_{i} (x,y,\omega)` is the initial/propagated fields complex field envelope
    and :math:`\mathcal{F}_{x,y}` is the 2D fourier transform in the transverse (x,y) axes.

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

    n : float, 1d array of floats or callable, Optional
        Refractive index of the medium in which to propagate the laser.
        Can be either a single value if dispersive effects are ignored, a 1d array
        describing the refractive index along the frequency/wavelength axis of the
        laser pulse, or a function of the wavelength (in meters).
        Default value is n=1. to describe propagation in vacuum.
    """

    def __init__(self, omega0, dim, n=1.0):
        super().__init__()
        self.n = n  # refractive index
        self.omega0 = omega0
        self.dim = dim

    def propagate(self, distance, grid, dim=None):
        dim = self.dim if not dim else dim

        if dim == "rt":
            print("'rt' geometry not yet supported by AngularSpectrumPropagator")

        if dim == "xyt":
            # Get the spectral field in the spatial domain
            field, omega = grid.get_spectral_field()

            omega += self.omega0
            kz = omega / c

            # get field in k-space and spatial frequency axes
            field_kspace, axes_freq = fft(
                arr_in=field,
                which="transverse",
                axes_in=[grid.axes[0], grid.axes[1]],
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
                * (
                    kz[None, None, :] ** 2
                    - kx[:, None, None] ** 2
                    - ky[None, :, None] ** 2
                )
                ** 0.5
            )

            # compensate group delay to keep pulse centered in grid
            Nx, Ny, _ = phase.shape
            phase_onaxis = phase[Nx // 2, Ny // 2, :]

            order = np.argsort(omega)

            phase_onaxis = phase_onaxis[order]
            omega_sorted = omega[order]

            phase_onaxis = np.unwrap(phase_onaxis)

            gd = np.gradient(phase_onaxis, omega_sorted)
            gd = np.interp(self.omega0, omega_sorted, gd)

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

            grid.set_spectral_field(field)
