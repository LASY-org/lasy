import numpy as np
import scipy.special as scispe
from .transverse_profile import TransverseProfile

class JincTransverseProfile(TransverseProfile):
    """
    Derived class for the analytic profile of a Jinc laser pulse.
    """

    def __init__(self, w0):
        """
        Defines a Jinc transverse envelope.

        More precisely, the transverse envelope
        (to be used in the :class:CombinedLongitudinalTransverseLaser class)
        corresponds to:

        .. math::

            \\mathcal{T}(x, y) = \\frac{J_1(r/w_0)}{r/w_0} \\textrm{, with } r=\\sqrt{x^2+y^2}
        where :math:`J_1` is the Bessel function of the first kind with order one

        Parameters
        ----------
        w0: float (in meter)
            The waist of the laser pulse, i.e. :math:`w_0` in the above formula.
        """
        self.w0 = w0

    def evaluate( self, x, y ):
        """
        Returns the transverse envelope

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
        r_over_w0 = np.sqrt(x**2 + y**2)/self.w0
        envelope = np.where(r_over_w0 !=0, 2.0*scispe.jv(1, r_over_w0)/r_over_w0, 1.0)

        return envelope
