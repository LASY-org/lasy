from .optical_element import OpticalElement
import numpy as np
from scipy.constants import c


class AxiParabolaWithDelay(OpticalElement):
    r"""
    Class that represents the combination of an axiparabola with
    an additional optical element that provides a radially-dependent
    delay (e.g. an optical echelon) to tune the group velocity.

    The rays that impinge the axiparabola at different radii are focused
    to different positions on the axis (resulting in an extended "focal
    range"). An additional radially-dependent delay is usually applied,
    in order to tune the effective group velocity on axis.

    For more details, see K. Oubrerie et al, "Axiparabola: a new tool
    for high-intensity optics", J. Opt. 24 045503 (2022)

    Parameters
    ----------
    f0: float (in meter)
        The focal distance, i.e. the distance, from the axiparabola,
        where the focal range starts.

    delta: float (in meter)
        The length of the focal range.

    R: float (in meter)
        The radius of the axiparabola. Rays coming from r=0 focus
        at z=f0 ; rays coming from r=R focus at z=f0+delta
    """

    def __init__(self, f0, delta, R):
        self.f0 = f0
        self.delta = delta
        self.R = R

        # Function that defines the z position where rays that impinge at r focus.
        # Assuming uniform intensity on the axiparabola, and in order to get
        # a z-independent intensity over the focal range, we need
        # (see Eq. 6 in Oubrerie et al.)
        z_foc = lambda r: self.f0 + self.delta * (r / self.R) ** 2

        # Solve Eq. 2 in Oubrerie et al. to find the sag function

    def amplitude_multiplier(self, x, y, omega):
        # Interpolation
        sag = np.zeros_like(x)

        # Calculate phase shift
        T = np.exp(-2j * (omega / c) * sag)
        # Remove intensity beyond R
        T[x**2 + y**2 > self.R**2] = 0

        return T
