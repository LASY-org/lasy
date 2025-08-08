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
    R : float (in meter) or tuple (floats)
        The radius of the mask for round masks or half-width and half-height for rectangular masks as tuple.
        If the shape is rectangular and only one number is given, a quadratic shape is assumed. 
    center: tuple (floats)
        Center of the mask. Default is (0,0)
    mask_type: string
        Should be 'aperture' (default, allows light inside) or 'hole' (allows light outside).
    shape: string
        Should be 'round' (default, for round apertures) or 'rectangular' (for rectangular apertures).

    """

    def __init__(self, R, center=(0, 0), mask_type="aperture",shape="round"):
        assert mask_type in ["aperture", "hole"], (
            "mask_type must be 'aperture' or 'hole'"
        )
        assert shape in ["round", "rectangular"], (
            "shape must be 'round' or 'rectangular'"
        )
        
        self.R = R
        self.center = center
        self.mask_type = mask_type
        self.shape = shape
        if shape==round:
            assert R.type==float (
            "Radius cannot be a tuple'"
        )
        
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
        if shape=='round':
            r_squared = (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2
            mask = r_squared <= self.R**2  # True inside, False outside

        if shape=='rectangular':
            if R.type==float
                halfwidth=R
                halfheight=R
            if R.type==tuple:
                halfwidth=R[0]
                halfheight=R[1]
            mask = ((x-self.center[0]) <= halfwidth) and ((x-self.center[0]) >= -halfwidth) and ((y-self.center[y]) <= halfheight) and ((y-self.center[y]) >= halfheight)) # True inside, False outside

        if self.mask_type == "aperture":
            return mask.astype(float)  # 1 inside, 0 outside
        else:  # "hole"
            return (~mask).astype(float)  # 0 inside, 1 outside

            

