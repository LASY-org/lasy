import numpy as np
from scipy.constants import c

from lasy.utils.fft_wrapper import fft

from .propagator import Propagator


class AngularSpectrumDFFTPropagator(Propagator):
    """
    Angular spectrum dual FFT propagator.
    """

    def __init__(self, omega0, n=1.0):
        super().__init__()
        self.n = n  # refractive index
        self.omega0 = omega0

    def propagate(self, distance, grid, dim):
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
            n = self.n(2*np.pi*c/omega) if callable(self.n) else self.n

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
