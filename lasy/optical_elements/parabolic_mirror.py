from . import OpticalElement
import numpy as np

class ParabolicMirror(OpticalElement):
    r"""
    Derived class for a parabolic mirror.

    More precisely, the amplitude multiplier corresponds to:

    .. math::

        T(\boldsymbol{x}_\perp,\omega) = \exp(i\omega \sqrt{x^2+y^2}/2f)

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
        return np.exp(1j*omega*(x**2 + y**2)/(2*self.f))