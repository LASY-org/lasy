import numpy as np

from .transverse_profile import TransverseProfile


class GaussianTransverseProfile(TransverseProfile):
    """
    Derived class for the analytic profile of a Gaussian laser pulse.

    More precisely, the transverse envelope
    (to be used in the :class:CombinedLongitudinalTransverseLaser class)
    corresponds to:

    .. math::

        \\mathcal{T}(x, y) = \\exp\\left( -\\frac{x^2 + y^2}{w_0^2} \\right)

    Parameters
    ----------
    w0 : float (in meter)
        The waist of the laser pulse, i.e. :math:`w_0` in the above formula.
    """

    def __init__(self, w0):
        super().__init__()
        self.w0 = w0

    def _evaluate(self, x, y):
        """
        Returns the transverse envelope

        Parameters
        ----------
        x, y : ndarrays of floats
            Define points on which to evaluate the envelope
            These arrays need to all have the same shape.

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y
        """
        envelope = np.exp(-(x**2 + y**2) / self.w0**2)

        return envelope
