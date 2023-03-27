import numpy as np
from scipy.special import genlaguerre

from .transverse_profile import TransverseProfile


class LaguerreGaussianTransverseProfile(TransverseProfile):
    """Derived class for an analytic profile of a high-order Gaussian laser
    pulse expressed in the Laguerre-Gaussian formalism.

    More precisely, the transverse envelope
    (to be used in the :class:CombinedLongitudinalTransverseLaser class)
    corresponds to:

    .. math::

        \\mathcal{T}(x, y) = r^{|m|}e^{-im\\theta} \\,
        L_p^{|m|}\\left( \\frac{2 r^2 }{w_0^2}\\right )\\,
        \\exp\\left( -\\frac{r^2}{w_0^2} \\right)

    where :math:`x = r \\cos{\\theta}`,
    :math:`y = r \\sin{\\theta}`, :math:`L_p^{|m|}` is the
    Generalised Laguerre polynomial of radial order :math:`p` and
    azimuthal order :math:`|m|`

    Parameters
    ----------
    w0 : float (in meter)
        The waist of the laser pulse, i.e. :math:`w_0` in the above formula.
    p : int (dimensionless)
        The radial order of Generalized Laguerre polynomial
    m : int (dimensionless)
        Defines the phase rotation, i.e. :math:`m` in the above formula.
    """

    def __init__(self, w0, p, m):
        super().__init__()
        self.w0 = w0
        self.p = p
        self.m = m

    def _evaluate(self, x, y):
        """Returns the transverse envelope.

        Parameters
        ----------
        x, y: ndarrays of floats
            Define points on which to evaluate the envelope
            These arrays need to all have the same shape.

        Returns
        -------
        envelope: ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y
        """
        # complex_position corresponds to r e^{+/-i\theta}
        if self.m > 0:
            complex_position = x - 1j * y
        else:
            complex_position = x + 1j * y
        radius = abs(complex_position)
        scaled_rad_squared = (radius**2) / self.w0**2
        envelope = (
            complex_position ** abs(self.m)
            * genlaguerre(self.p, abs(self.m))(2 * scaled_rad_squared)
            * np.exp(-scaled_rad_squared)
        )

        return envelope
