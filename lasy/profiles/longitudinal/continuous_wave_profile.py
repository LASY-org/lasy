import numpy as np

from .longitudinal_profile import LongitudinalProfile


class ContinuousWaveProfile(LongitudinalProfile):
    r"""
    Class representing a continous wave laser longitudinal profile.

    Specifically, the longitudinal profile will be represented by a constant value

    Parameters
    ----------
    wavelength : float (in meter)
        The main laser wavelength :math:`\lambda_0` of the laser.
    """

    def __init__(self, wavelength):
        super().__init__(wavelength)
        self.is_cw = True

    def evaluate(self, t):
        """
        Return the longitudinal envelope.

        Parameters
        ----------
        t : ndarrays of floats
            Define longitudinal points on which to evaluate the envelope

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the longitudinal envelope at the
            specified points. This array has the same shape as the array t.
        """
        return np.ones_like(t + 0 * 1j)
