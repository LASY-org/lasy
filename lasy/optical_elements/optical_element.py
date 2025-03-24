from abc import ABC, abstractmethod

import numpy as np


class OpticalElement(ABC):
    """
    Base class to model thin optical elements.

    Any optical element should inherit from this class, and define its own
    `amplitude_multiplier` method, using the same signature as the method below.
    """

    def __init__(self):
        pass

    @abstractmethod
    def amplitude_multiplier(self, x, y, omega):
        r"""
        Return the amplitude multiplier :math:`T`.

        This number multiplies the complex amplitude of the laser
        just before this thin element, in order to obtain the complex
        amplitude output laser just after this thin element:

        .. math::

            \tilde{\mathcal{E}}_{out}(x, y, \omega) = T(x, y, \omega)\tilde{\mathcal{E}}_{in}(x, y, \omega)

        Parameters
        ----------
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
            This array has the same shape as the array omega.
        """
        # The base class only defines dummy multiplier
        # (This should be replaced by any class that inherits from this one.)
        return np.ones_like(x, dtype="complex128")
