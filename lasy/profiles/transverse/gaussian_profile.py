import numpy as np

from .transverse_profile import TransverseProfile


class GaussianTransverseProfile(TransverseProfile):
    """
    Derived class for the analytic profile of a Gaussian laser pulse.

    More precisely, at focus (`z_init=0`), the transverse envelope
    (to be used in the :class:CombinedLongitudinalTransverseLaser class)
    corresponds to:

    .. math::

        \\mathcal{T}(x, y) = \\exp\\left( -\\frac{x^2 + y^2}{w_0^2} \\right)

    Parameters
    ----------
    w0 : float (in meter)
        The waist of the laser pulse, i.e. :math:`w_0` in the above formula.

    wavelength : float (in meter)
        The main laser wavelength :math:`\\lambda_0` of the laser.

    z_init : float (in meter), optional
        Position at which the pulse is initialized, relative to the position
        of the focal plane. (The focal plane is at `z=0`.)
    """

    def __init__(self, w0, wavelength, z_init=0):
        super().__init__()
        self.w0 = w0
        self.z_init_over_zr = z_init * wavelength / (np.pi * w0**2)

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
        # Term for wavefront curvature + Gouy phase
        diffract_factor = 1.0 + 1j * self.z_init_over_zr
        # Calculate the argument of the complex exponential
        exp_argument = -(x**2 + y**2) / (self.w0**2 * diffract_factor)
        # Get the transverse profile
        envelope = np.exp(exp_argument) / diffract_factor

        return envelope
