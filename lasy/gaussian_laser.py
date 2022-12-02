from .laser import Laser
import numpy as np

class GaussianLaser(Laser):
    """
    Derived class for an analytic profile of a Gaussian laser pulse.
    """

    def __init__(self, box, wavelength, pol,
                laser_energy, w0, tau, t_peak, cep_phase=0):
        """
        Gaussian laser constructor

        TODO

        Re[ e^{i cep_phase - i omega(t-t_peak)}]

        ecp_phase: phase at t_peak

        Parameters:
        -----------
        TODO

        """
        super().__init__(box, wavelength, pol)

        t = box.axes[-1]
        long_profile = np.exp( -(t-t_peak)**2/tau**2 \
                               + 1.j*(cep_phase + self.omega0*t_peak) )

        if self.dim == 'xyt':
            x = box.axes[0]
            y = box.axes[1]
            transverse_profile = np.exp(
                            -(x[:,np.newaxis]**2 + y[np.newaxis, :]**2)/w0**2 )
            self.field.field[...] = transverse_profile[:,:,np.newaxis] * \
                                      long_profile[np.newaxis, np.newaxis, :]
        elif self.dim == 'rz':
            r = box.axes[0]
            transverse_profile = np.exp( -r**2/w0**2 )
            self.field.field[...] = transverse_profile[:,np.newaxis] * \
                                      long_profile[np.newaxis, :]

        # TODO: normalize to correct energy
