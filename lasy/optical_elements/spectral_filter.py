import numpy as np

from .optical_element import OpticalElement


class SpectralFilter(OpticalElement):
    r"""
    Class for an optical element that filters the spectrum of a laser pulse.

    The amplitude multiplier corresponds to:

    .. math::

        T(\omega) = \sqrt{F(\omega)}

    where :math:`F(\omega)` is the intensity transmission function of the filter.

    Parameters
    ----------
    transmission : 1D ndarray of floats
        Intensity/Energy transmission of the filter.
    omega : 1D ndarray of floats (in rad/s)
        Angular frequencies at which the transmission is defined.
    """

    def __init__(self, transmission, omega):
        self.transmission = transmission
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
        self.transmission = self.transmission[order]

        # interpolate transmission function to omega axis of the laser
        transmission = np.interp(omega, self.omega_in, self.transmission)

        # return the square root of the energy/intensity transmission
        return np.sqrt(transmission)
