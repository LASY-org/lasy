import numpy as np
from .transverse_profile import TransverseProfile

class SuperGaussianTransverseProfile(TransverseProfile):
    """
    Derived class for the analytic profile of a super-Gaussian laser pulse.
    """

    def __init__(self, w0, n_order):
        """
        Defines a super-Gaussian transverse envelope.

        More precisely, the transverse envelope corresponds to:

        .. math::

            \\mathcal{T}(x, y) = \\exp\\left( -\\left({\\frac{{x^2 + y^2}}{w_0^2}}\\right)^{\\dfrac{n}{2}} \\right)

        Parameters
        ----------
        w0: float (in meter)
            The waist of the laser pulse, i.e. :math:`w_0` in the above formula.

        n_order: float (in meter)
            The shape parameter of the super-gaussian function, i.e. :math:`n` in the above formula.
            If :math:`n=2` the super-Gaussian becomes a standard Gaussian function.
            If :math:`n=1` the super-Gaussian becomes a Laplace function.
        """
        super().__init__()
        self.w0 = w0
        self.n_order = n_order

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
        envelope = np.exp( -np.power((x**2 + y**2)/self.w0**2, self.n_order/2) )

        return envelope
