import numpy as np
import scipy.special as scispe

from .transverse_profile import TransverseProfile


class JincTransverseProfile(TransverseProfile):
    r"""
    Class for the analytic profile of a Jinc laser pulse.

    The transverse envelope corresponds to:

    .. math::

        \mathcal{T}(x, y) = 2\frac{J_1(r/w_0)}{r/w_0} \textrm{, with } r=\sqrt{x^2+y^2}

    where :math:`J_1` is the Bessel function of the first kind of order one

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
        Return the transverse envelope.

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
        r_over_w0 = np.sqrt(x**2 + y**2) / self.w0

        envelope = np.ones_like(r_over_w0)
        # Avoid dividing by zero
        np.divide(
            2.0 * scispe.jv(1, r_over_w0),
            r_over_w0,
            out=envelope,
            where=r_over_w0 > 0.0,
        )

        return envelope
