import numpy as np

from lasy.optical_elements.optical_element import OpticalElement


class IntensityMask(OpticalElement):
    """
    Class for a radially symmetric intensity mask.

    This creates an optical element which acts to mask out the intensity of a laser pulse.
    The mask is radially symmetric and can either mask intensity beyond a user defined radius, thus acting as
    an aperture. Alternatively, the optica can mask intensity within a user defined radius, acting in this case as
    an optic with a hole.

    Parameters
    ----------
    R : float (in meter)
        The radius of the mask
    center: tuple (floats)
        Center of the mask. Default is (0,0)
    mask_type: string
        Should be 'aperture' (default, allows light inside) or 'hole' (allows light outside).
    order: int or None
        If None (default), the mask has a hard edge. Otherwise, the edge is a
        super-Gaussian of this order.
    """

    def __init__(self, R, center=(0, 0), mask_type="aperture", order=None):
        assert mask_type in ["aperture", "hole"], (
            "mask_type must be 'aperture' or 'hole'"
        )
        self.R = R
        self.center = center
        self.mask_type = mask_type
        self.order = order

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
        r_squared = (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2

        if self.order is None:
            # Hard edge
            aperture = (r_squared <= self.R**2).astype(float)  # 1 inside, 0 outside
        else:
            # Super-Gaussian edge: exp(-(r/R)^(2n)); n=1 is a Gaussian, large n approaches a hard edge
            with np.errstate(over="ignore"):
                aperture = np.exp(-((r_squared / self.R**2) ** (self.order / 2)))

        if self.mask_type == "aperture":
            return aperture
        else:  # "hole"
            return 1.0 - aperture
