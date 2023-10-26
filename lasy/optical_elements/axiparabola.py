from .optical_element import OpticalElement
import numpy as np
from scipy.constants import c


class AxiParabola(OpticalElement):
    r"""
    Class that represents the combination of an axiparabola with
    an additional optical element that provides a radially-dependent
    delay (e.g. an optical echelon) to tune the group velocity.

    The rays that impinge the axiparabola at different radii are focused
    to different positions on the axis (resulting in an extended "focal
    range"). An additional radially-dependent delay is usually applied,
    in order to tune the effective group velocity on axis.

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
        The radius of the axiparabola. Rays coming from r=0 focus
        at z=f0 ; rays coming from r=R focus at z=f0+delta
    """

    def __init__(self, f0, delta, R):
        self.f0 = f0
        self.delta = delta
        self.R = R

    def amplitude_multiplier(self, x, y, omega):
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
