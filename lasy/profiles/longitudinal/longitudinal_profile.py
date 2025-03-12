import numpy as np
from scipy.constants import c, pi


class LongitudinalProfile(object):
    """
    Base class for longitudinal profiles of laser pulses.

    Any new longitudinal profile should inherit from this class, and define
    its own `evaluate` method, using the same signature as the method below.
    """

    def __init__(self, wavelength):
        self.lambda0 = wavelength
        self.omega0 = 2 * pi * c / self.lambda0
        self.is_cw = False

    def evaluate(self, t):
        """
        Return the longitudinal envelope.

        Parameters
        ----------
        t : ndarrays of floats
            Define points on which to evaluate the envelope

        Returns
        -------
        envelope: ndarray of complex numbers
            Contains the value of the longitudinal envelope at the
            specified points. This array has the same shape as the array t.
        """
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        return np.zeros(t.shape, dtype="complex128")

    def __update_is_cw__(self, value):
        """Update state of is_cw variable."""
        self.is_cw = value
