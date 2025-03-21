import numpy as np

from .optical_element import OpticalElement


class SpectralPhase(OpticalElement):
    r"""
    Class for an optical element that adds a spectral phase to a laser pulse.

    The amplitude multiplier corresponds to:

    .. math::

        T(\omega) = \exp(i(\phi(\omega)))

    where :math:`\phi(\omega)` is the spectral phase.

    Parameters
    ----------
    phase : 1D ndarray of floats (in rad)
        Phase that should be applied to the laser pulse. The phase is assumed to be unwrapped.
    omega : 1D ndarray of floats (in rad/s)
        Angular frequencies at which the phase is defined.
    """

    def __init__(self, phase, omega):
        self.phase = phase
        self.omega_in = omega

    def amplitude_multiplier(self, x, y, omega):
        """
        Return the amplitude multiplier.

        Parameters
        ----------
        x, y, omega : ndarrays of floats
            Define points on which to evaluate the multiplier.
            These arrays need to all have the same shape.

        Returns
        -------
        multiplier : ndarray of complex numbers
            Contains the value of the multiplier at the specified points.
        """
        # Sort the transmission function and angular frequencies to
        # eliminate problems with interpolation (e.g. due to fftshifted data)
        order = np.argsort(self.omega_in)
        self.omega_in = self.omega_in[order]
        self.phase = self.phase[order]

        # interpolate transmission function to omega axis of the laser
        phase = np.interp(omega, self.omega_in, self.phase)

        return np.exp(1j * phase)
