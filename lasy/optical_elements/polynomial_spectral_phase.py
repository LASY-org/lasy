import numpy as np

from .optical_element import OpticalElement


class PolynomialSpectralPhase(OpticalElement):
    r"""
    Class for an optical element with added spectral phase.

    The amplitude multiplier corresponds to:

    .. math::

        T(\omega) = \exp(i(\phi(\omega)))

    where :math:`\phi(\omega)` is the spectral phase given by:

    .. math::

        \phi(\omega) = \frac{\text{GDD}}{2!} (\omega - \omega_0)^2 + \frac{\text{TOD}}{3!} (\omega - \omega_0)^3 + \frac{\text{FOD}}{4!} (\omega - \omega_0)^4

    The other parameters in this formula are defined below.

    Parameters
    ----------
    gdd : float (in s^2)
        Group Delay Dispersion.
    tod : float (in s^3)
        Third-order Dispersion.
    fod : float (in s^4)
        Fourth-order Dispersion.
    omega_0 : float (in rad/s)
        Central angular frequency.
    """

    def __init__(self, gdd, tod, fod):
        self.gdd = gdd
        self.tod = tod
        self.fod = fod

    def amplitude_multiplier(self, omega, omega0):
        """
        Return the amplitude multiplier.

        Parameters
        ----------
        omega: ndarray of floats
            Define points on which to evaluate the multiplier.

        Returns
        -------
        multiplier: ndarray of complex numbers
            Contains the value of the multiplier at the specified points
            This array has the same shape as the array omega
        """
        spectral_phase = (
            self.gdd / 2 * (omega - omega0) ** 2
            + self.tod / 6 * (omega - omega0) ** 3
            + self.fod / 24 * (omega - omega0) ** 4
        )

        return np.exp(1j * spectral_phase)
