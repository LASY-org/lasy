import numpy as np
from scipy.special import hermite
from math import factorial
from .transverse_profile import TransverseProfile


class HermiteGaussianTransverseProfile(TransverseProfile):
    r"""
    A high-order Gaussian laser pulse expressed in the Hermite-Gaussian formalism.

    Derived class for an analytic profile.
    More precisely, the transverse envelope (to be used in the
    :class:CombinedLongitudinalTransverseLaser class) corresponds to:

    .. math::

        \mathcal{T}(x, y) = \,
        \sqrt{\frac{2}{\pi}} \sqrt{\frac{1}{2^{n} n! w_0}}\,
        \sqrt{\frac{1}{2^{n} n! w_0}}\,
        H_{n_x}\left ( \frac{\sqrt{2} x}{w_0}\right )\,
        H_{n_y}\left ( \frac{\sqrt{2} y}{w_0}\right )\,
        \exp\left( -\frac{x^2+y^2}{w_0^2} \right)

    where  :math:`H_{n}` is the Hermite polynomial of order :math:`n`.

    Parameters
    ----------
    w0 : float (in meter)
        The waist of the laser pulse, i.e. :math:`w_0` in the above formula.
    n_x : int (dimensionless)
        The order of hermite polynomial in the x direction
    n_y : int (dimensionless)
        The order of hermite polynomial in the y direction
    """

    def __init__(self, w0, n_x, n_y):
        super().__init__()
        self.w0 = w0
        self.n_x = n_x
        self.n_y = n_y

    def _evaluate(self, x, y):
        """
        Return the transverse envelope.

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
        envelope = (
            np.sqrt(2 / np.pi)
            * np.sqrt(1 / (2 ** (self.n_x) * factorial(self.n_x) * self.w0))
            * np.sqrt(1 / (2 ** (self.n_y) * factorial(self.n_y) * self.w0))
            * hermite(self.n_x)(np.sqrt(2) * x / self.w0)
            * hermite(self.n_y)(np.sqrt(2) * y / self.w0)
            * np.exp(-(x**2 + y**2) / self.w0**2)
        )

        return envelope
