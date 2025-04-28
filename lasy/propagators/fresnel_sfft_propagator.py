from .single_fft_propagator import SingleFFTPropagator
from lasy.utils.fft import fft
import numpy as np
from scipy.constants import c

class FresnelSFFTPropagator(SingleFFTPropagator):
    def __init__(self):
        """
        An implementation of the single FFT propagator using the Fresnel approximation.

        Following Goodman.
        Goodman, J. W. (2005). Introduction to Fourier Optics. Roberts and Company Publishers.
        
        """
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
        spectral_field,spectral_axes = grid.get_spectral_field()

        if self.dim == 'rt':
            print("Fresnel SFFT propagator in rt")
        
        elif self.dim =='xyt':
            print("Fresnel SFFT propagator in xyt")
            axes = grid.axes
            x = axes[0]
            y = axes[1]


            X,Y,OM = np.meshgrid(axes[1],axes[0],spectral_axes+omega0)
            K = OM /c
            WAVELENGTH = 2*np.pi*c/OM

            # Goodman pg 67
            preFactor = np.exp( 1j*K*distance ) *np.exp(1j*K*(X**2 + Y**2)/(2*distance))/(1j*WAVELENGTH*distance)

            fftInput = spectral_field * preFactor
            dx = x[1]-x[0]
            dy = y[1]-y[0]

            F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(fftInput,axes=(0,1)),axes=(0,1)),axes=(0,1))*dx*dy

            X = x[-1] - x[0]
            sampleXFreq = len(x)/X
            k_x = 2*np.pi*np.linspace(-sampleXFreq/2, (sampleXFreq/2-sampleXFreq/len(x)) , len(x))
            Y = y[-1] - y[0]
            sampleYFreq = len(y)/Y
            k_y = 2*np.pi*np.linspace(-sampleYFreq/2, (sampleYFreq/2-sampleYFreq/len(y)) , len(y))

            KY,KX,_ = np.meshgrid(k_y,k_x,spectral_axes)


            XF = KX*WAVELENGTH*distance/2/np.pi
            YF = KY*WAVELENGTH*distance/2/np.pi

            # old post factor seems to be incorrect by factor 2 from goodman pg 67
            #postFactor = np.exp( 1j*k/z * (XF**2 + YF**2) ) 
            postFactor = np.exp( 1j*K/distance * (XF**2 + YF**2) ) 

            diffractedField= F*postFactor

            grid.set_spectral_field(diffractedField)
            grid.axes[0] = np.unique(XF)
            grid.axes[1] = np.unique(YF)
            grid.lo = [np.unique(XF)[0], np.unique(YF)[0], self.grid.lo[-1]]
            grid.hi = [np.unique(XF)[-1], np.unique(YF)[-1], self.grid.hi[-1]]

