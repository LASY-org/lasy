import numpy as np

from .optical_element import OpticalElement


class PolynomialSpectralPhase(OpticalElement):
    r"""
    Class for an optical element that adds spectral phase (e.g. a dazzler).

    The amplitude multiplier corresponds to:

    .. math::

        T(\omega) = \exp(i(\phi(\omega)))

    where :math:`\phi(\omega)` is the spectral phase given by:

    .. math::

        \phi(\omega) = \frac{\text{GDD}}{2!} (\omega - \omega_0)^2 + \frac{\text{TOD}}{3!} (\omega - \omega_0)^3 + \frac{\text{FOD}}{4!} (\omega - \omega_0)^4

    The other parameters in this formula are defined below.

    Parameters
    ----------
    gdd : float (in s^2), optional
        Group Delay Dispersion (by default: ``gdd=0``). ``gdd > 0`` corresponds to a positive
        chirp, i.e. the low-frequency part of the spectrum arriving earlier than the
        high-frequency part of the spectrum.
    tod : float (in s^3), optional
        Third-order Dispersion (by default: ``tod=0``). For a Gaussian pulse, adding a positive
        TOD (``tod > 0``) results in the apparition of post-pulses, i.e. lower intensity pulses
        arriving after the main pulse.
    fod : float (in s^4), optional
        Fourth-order Dispersion (by default: ``fod=0``).
    """

    def __init__(self, gdd=0, tod=0, fod=0):
        self.gdd = gdd
        self.tod = tod
        self.fod = fod

    def amplitude_multiplier(self, x, y, omega, omega0):
        """
        Return the amplitude multiplier.

        Parameters
        ----------
        x, y, omega : ndarrays of floats
            Define points on which to evaluate the multiplier.
            These arrays need to all have the same shape.
        omega0 : float (in rad/s)
            Central angular frequency, as used for the definition
            of the laser envelope.

        Returns
        -------
        multiplier : ndarray of complex numbers
            Contains the value of the multiplier at the specified points.
            This array has the same shape as the array omega.
        """
        spectral_phase = (
            self.gdd / 2 * (omega - omega0) ** 2
            + self.tod / 6 * (omega - omega0) ** 3
            + self.fod / 24 * (omega - omega0) ** 4
        )

        return np.exp(1j * spectral_phase)
