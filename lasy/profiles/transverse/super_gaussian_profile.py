import numpy as np
from .transverse_profile import TransverseProfile

class SuperGaussianTransverseProfile(TransverseProfile):
    """
    Derived class for the analytic profile of a super-Gaussian laser pulse.
    """

    def __init__(self, w0, beta):
        """
        Defines a super-Gaussian transverse envelope.

        More precisely, the transverse envelope corresponds to:

        .. math::

            \\mathcal{T}(x, y) = \\exp\\left( -\\left({\\frac{\\sqrt{x^2 + y^2}}{w_0}}\\right)^{\\beta} \\right)

        Parameters
        ----------
        w0: float (in meter)
            The waist of the laser pulse, i.e. :math:`w_0` in the above formula.

        beta: float (in meter)
            The shape parameter of the super-gaussian function, i.e. :math:`\\beta` in the above formula.
            If :math:`\\beta=2` the super-Gaussian becomes a standard Gaussian function.
            If :math:`\\beta=1` the super-Gaussian becomes a Laplace function.
        """
        self.w0 = w0
        self.beta = beta

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
        envelope = np.exp( -np.power(np.sqrt(x**2 + y**2)/self.w0, self.beta) )

        return envelope