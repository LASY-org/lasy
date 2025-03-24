import numpy as np
from scipy.constants import c

from .optical_element import OpticalElement


class Axiparabola(OpticalElement):
    r"""
    Class that represents an axiparabola.

    The rays that impinge on the axiparabola at different radii are focused
    to different positions on the axis (resulting in an extended "focal range").

    For more details, see S. Smartsev et al, "Axiparabola: a long-focal-depth,
    high-resolution mirror for broadband high-intensity lasers", Optics Letters 44, 14 (2019)

    Parameters
    ----------
    f0: float (in meter)
        The focal distance, i.e. the distance, from the axiparabola,
        where the focal range starts.

    delta: float (in meter)
        The length of the focal range.

    R: float (in meter)
        The radius of the axiparabola. Rays coming from ``r=0`` focus
        at ``z=f0`` ; rays coming from ``r=R`` focus at ``z=f0+delta``
    """

    def __init__(self, f0, delta, R):
        self.f0 = f0
        self.delta = delta
        self.R = R

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
        # Implement Eq. 4 in Smatsev et al.
        r2 = x**2 + y**2
        sag = (
            (1.0 / (4 * self.f0)) * r2
            - (self.delta / (8 * self.f0**2 * self.R**2)) * r2**2
            + self.delta
            * (self.R**2 + 8 * self.f0 * self.delta)
            / (96 * self.f0**4 * self.R**4)
            * r2**3
        )

        # Calculate phase shift
        T = np.exp(-2j * (omega / c) * sag)
        # Remove intensity beyond R
        T[x**2 + y**2 > self.R**2] = 0

        return T
