from .transverse_profile import TransverseProfile
import numpy as np
from scipy.special.orthogonal import hermite

class HermiteGaussianTransverseProfile(TransverseProfile):
    """
    Derived class for an analytic profile of a high-order Gaussian
    laser pulse expressed in the Hermite-Gaussian formalism.
    """

    def __init__(self, w0, n_x, n_y):
        """
        Defines a Hermite-Gaussian transverse envelope

        More precisely, the transverse envelope
        (to be used in the :class:CombinedLongitudinalTransverseLaser class)
        corresponds to:

        .. math::
            \\mathcal{T}(x, y) =
            H_{n_x}\\left ( \\frac{\\sqrt{2} x}{w_0}\\right )\\,
            H_{n_y}\\left ( \\frac{\\sqrt{2} y}{w_0}\\right )\\,
            \\exp\\left( -\\frac{x^2+y^2}{w_0^2} \\right)

        where  :math:`H_{n}` is the Hermite polynomial of order :math:`n`.

        Parameters
        ----------
        w0: float (in meter)
            The waist of the laser pulse, i.e. :math:`w_0` in the above formula.
        n_x: int (dimensionless)
            The order of hermite polynomial in the x direction
        n_y: int (dimensionless)
            The order of hermite polynomial in the y direction
        """
        super().__init__()
        self.w0 = w0
        self.n_x = n_x
        self.n_y = n_y

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
        envelope = hermite(self.n_x)(np.sqrt(2)*x/self.w0) * \
                             hermite(self.n_y)(np.sqrt(2)*y/self.w0) * \
                             np.exp( -(x**2 + y**2)/self.w0**2 )

        return envelope
