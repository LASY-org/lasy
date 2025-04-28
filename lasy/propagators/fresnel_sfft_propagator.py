from lasy.utils.fft import fft
import numpy as np
from scipy.constants import c

from .single_fft_propagator import SingleFFTPropagator


class FresnelSFFTPropagator(SingleFFTPropagator):
    """
    An implementation of the single FFT propagator using the Fresnel approximation.

    Following Goodman.
    Goodman, J. W. (2005). Introduction to Fourier Optics. Roberts and Company Publishers.

    """

    def __init__(self):
        super().__init__()

    def propagate(self, grid, distance):
        """
        Propagate the input grid using the Fresnel SFFT method.

        Parameters
        ----------
        grid : Grid
            The input grid.
        distance : float
            The distance to propagate.

        Returns
        -------
        Grid
            The propagated grid.
        """
        axes = grid.axes
        omega0 = self.omega0

        # Get the spectral field and axes from the input grid
        spectral_field, spectral_axes = grid.get_spectral_field()

        if self.dim == "rt":
            print("Fresnel SFFT propagator in rt")

        elif self.dim == "xyt":
            print("Fresnel SFFT propagator in xyt")
            axes = grid.axes
            x = axes[0]
            y = axes[1]

            X, Y, OM = np.meshgrid(axes[1], axes[0], spectral_axes + omega0)
            K = OM / c
            WAVELENGTH = 2 * np.pi * c / OM

            # Goodman pg 67
            preFactor = (
                np.exp(1j * K * distance)
                * np.exp(1j * K * (X**2 + Y**2) / (2 * distance))
                / (1j * WAVELENGTH * distance)
            )

            fftInput = spectral_field * preFactor
            
            F , axes_out = fft(fftInput, axes_in=(x,y),which="transverse",inverse=False,shift_before=True,shift_after=False)
            k_x , k_y = axes_out
            
            KY, KX, _ = np.meshgrid(k_y, k_x, spectral_axes)

            XF = KX * WAVELENGTH * distance / 2 / np.pi
            YF = KY * WAVELENGTH * distance / 2 / np.pi

            # old post factor seems to be incorrect by factor 2 from goodman pg 67
            # postFactor = np.exp( 1j*k/z * (XF**2 + YF**2) )
            postFactor = np.exp(1j * K / distance * (XF**2 + YF**2))

            diffractedField = F * postFactor

            grid.set_spectral_field(diffractedField)
            grid.axes[0] = np.unique(XF)
            grid.axes[1] = np.unique(YF)
            grid.lo = [np.unique(XF)[0], np.unique(YF)[0], self.grid.lo[-1]]
            grid.hi = [np.unique(XF)[-1], np.unique(YF)[-1], self.grid.hi[-1]]
