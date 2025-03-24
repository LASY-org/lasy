import numpy as np
from scipy.constants import c

from .optical_element import OpticalElement


class Axicon(OpticalElement):
    r"""
    Class for an axicon.

    This object technically represents a reflective axicon. However, it could
    also be used to represent a refractive axicon, if the chromatic effects of
    the refractive axicon are assumed to be negligible.

    More precisely, the amplitude multiplier corresponds to:

    .. math::

        T(\boldsymbol{x}_\perp,\omega) = \exp(-i (\omega/c) \sqrt{x^2+y^2} \tan(\gamma/2))

    where :math:`\boldsymbol{x}_\perp` is the transverse coordinate (orthogonal
    to the propagation direction). The other parameters in this formula
    are defined below.

    Parameters
    ----------
    gamma : float (in radians)
        The angle that the outcoming rays (coming from the axicon) would make with the optical axis,
        if the incoming rays (impinging on the axicon) are parallel to the optical axis.
    """

    def __init__(self, gamma):
        self.gamma = gamma

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
        return np.exp(
            -2j * (omega / c) * np.sqrt(x**2 + y**2) * np.tan(0.5 * self.gamma)
        )
