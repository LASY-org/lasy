from .propagator import Propagator
from lasy.utils.fourier import fft

from scipy.constants import c
import numpy as np


class AngularSpectrumDFFTPropagator(Propagator):
    """
    Angular spectrum dual FFT propagator.
    """

    def __init__(self, n=1.):

        super().__init__()
        self.n = n  # refractive index

    def propagate(self, distance, grid, dim):
        
        if dim == 'rt':
            print("'rt' geometry not yet supported by AngularSpectrumPropagator")

        if dim == 'xyt':
            # Get the spectral field in the spatial domain
            field, omega = grid.get_spectral_field()

            kz = omega/c

            # get field in k-space and spatial frequency axes
            field_kspace, axes_freq = fft(arr_in=field,
                                          which="transverse",
                                          axes_in=[grid.axes[0], grid.axes[1]],
                                          from_domain="real",)

            kx = 2*np.pi*axes_freq[0]
            ky = 2*np.pi*axes_freq[1]

            # Calculate the refractive index if it is a function of wavelength
            if type(self.n) not in [int, float, np.ndarray]:
                wavelength = 2*np.pi*c/omega
                n = self.n(wavelength)
            else:
                n = self.n

            # Calculate the phase shift in k-space
            phase = (distance * n * (kz[:, :, None]**2 - kx[None, :, :]**2 - ky[:, None, :]**2) ** 0.5)
            
            # Apply the phase shift to the field in k-space
            field_kspace *= np.exp(1j * phase)

            # Transform back to the spatial domain
            field_kspace, _ = fft(arr_in=field_kspace,
                                  which="transverse",
                                  axes_in=(kx/(2*np.pi), ky/(2*np.pi)),
                                  from_domain="frequency",)

            grid.set_spectral_field(field)
