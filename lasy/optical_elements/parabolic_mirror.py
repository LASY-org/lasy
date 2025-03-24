import numpy as np
from scipy.constants import c

from .optical_element import OpticalElement


class ParabolicMirror(OpticalElement):
    r"""
    Class for a parabolic mirror.

    More precisely, the amplitude multiplier corresponds to:

    .. math::

        T(\boldsymbol{x}_\perp,\omega) = \exp(-i\omega (x^2+y^2)/2cf)

    where
    :math:`\boldsymbol{x}_\perp` is the transverse coordinate (orthogonal
    to the propagation direction). The other parameters in this formula
    are defined below.

    Parameters
    ----------
    f : float (in meter)
        The focal length of the parabolic mirror.
    """

    def __init__(self, f):
        self.f = f

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
            This array has the same shape as the array omega.
        """
        return np.exp(-1j * omega * (x**2 + y**2) / (2 * c * self.f))
