<<<<<<< HEAD
from .propagator import Propagator
from lasy.utils.fft_wrapper import fft
=======
import numpy as np
from scipy.constants import c

from lasy.utils.fourier import fft
>>>>>>> 6047ebc78956349d6fd87041bb3c4649f789fd8f

from .propagator import Propagator


class AngularSpectrumDFFTPropagator(Propagator):
    """
    Angular spectrum dual FFT propagator.
    """

<<<<<<< HEAD
    def __init__(self, omega0, n=1.):

=======
    def __init__(self, n=1.0):
>>>>>>> 6047ebc78956349d6fd87041bb3c4649f789fd8f
        super().__init__()
        self.n = n  # refractive index
        self.omega0 = omega0
    def propagate(self, distance, grid, dim):
        if dim == "rt":
            print("'rt' geometry not yet supported by AngularSpectrumPropagator")

        if dim == "xyt":
            # Get the spectral field in the spatial domain
            field, omega = grid.get_spectral_field()
<<<<<<< HEAD

            omega += self.omega0
            kz = omega/c

            # get field in k-space and spatial frequency axes
            field_kspace, axes_freq = fft(arr_in=field,
                                          which="transverse",
                                          axes_in=[grid.axes[0], grid.axes[1]],
                                          from_domain="frequency",)

            kx = 2*np.pi*axes_freq[0]
            ky = 2*np.pi*axes_freq[1]

=======

            kz = omega / c

            # get field in k-space and spatial frequency axes
            field_kspace, axes_freq = fft(
                arr_in=field,
                which="transverse",
                axes_in=[grid.axes[0], grid.axes[1]],
                from_domain="real",
            )

            kx = 2 * np.pi * axes_freq[0]
            ky = 2 * np.pi * axes_freq[1]

>>>>>>> 6047ebc78956349d6fd87041bb3c4649f789fd8f
            # Calculate the refractive index if it is a function of wavelength
            if type(self.n) not in [int, float, np.ndarray]:
                wavelength = 2 * np.pi * c / omega
                n = self.n(wavelength)
                n0 = self.n(2*np.pi*c/self.omega0)
            else:
                n = self.n
                n0 = self.n

            # Calculate the phase shift in k-space
<<<<<<< HEAD
            phase = (distance * n * (kz[None, None, :]**2 - kx[:, None, None]**2 - ky[None, :, None]**2) ** 0.5)

            # compensate group delay to keep pulse centered in grid
            Nx, Ny, _ = phase.shape
            phase_onaxis = phase[Nx//2, Ny//2, :]

            order = np.argsort(omega)

            phase_onaxis = phase_onaxis[order]
            omega_sorted = omega[order]

            phase_onaxis = np.unwrap(phase_onaxis)

            gd = np.gradient(phase_onaxis, omega_sorted)
            gd = np.interp(self.omega0, omega_sorted, gd)

            phase = phase - gd*(omega-self.omega0)[None, None, :]
=======
            phase = (
                distance
                * n
                * (kz[:, :, None] ** 2 - kx[None, :, :] ** 2 - ky[:, None, :] ** 2)
                ** 0.5
            )
>>>>>>> 6047ebc78956349d6fd87041bb3c4649f789fd8f

            # Apply the phase shift to the field in k-space
            field_kspace *= np.exp(1j * phase)

            # Transform back to the spatial domain
<<<<<<< HEAD
            field, _ = fft(arr_in=field_kspace,
                                  which="transverse",
                                  axes_in=(kx/(2*np.pi), ky/(2*np.pi)),
                                  from_domain="real",)
=======
            field_kspace, _ = fft(
                arr_in=field_kspace,
                which="transverse",
                axes_in=(kx / (2 * np.pi), ky / (2 * np.pi)),
                from_domain="frequency",
            )
>>>>>>> 6047ebc78956349d6fd87041bb3c4649f789fd8f


            grid.set_spectral_field(field)
