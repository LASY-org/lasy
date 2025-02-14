import numpy as np
from scipy.constants import c

from lasy.optical_elements.optical_element import OpticalElement


class IntensityMask(OpticalElement):
    """
    Class for adding an radially symmetric intensity mask that acts as an aperture or a hole.

    Parameters
    ----------
    R : float (in meter)
        The Radius of the mask
    center: tuple (floats)
        Center of the mask. Default is (0,0)
    aperture_type: string
        Should be 'aperture' (default, allows light inside) or 'hole' (allows light outside).

    """

    def __init__(self, R, center=(0, 0), aperture_type="aperture"):
        assert aperture_type in ["aperture", "hole"], "aperture_type must be 'aperture' or 'hole'"
        self.R = R
        self.center = center
        self.aperture_type = aperture_type

    def amplitude_multiplier(self, x, y, omega):
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
        r_squared = (x - self.center[0])**2 + (y - self.center[1])**2
        mask = r_squared <= self.R**2  # True inside, False outside

        if self.aperture_type == "aperture":
            return mask.astype(float)  # 1 inside, 0 outside
        else:  # "hole"
            return (~mask).astype(float)  # 0 inside, 1 outside
