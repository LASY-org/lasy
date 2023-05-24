import numpy as np
from scipy.special import genlaguerre

from .transverse_profile import TransverseProfile


class LaguerreGaussianTransverseProfile(TransverseProfile):
    r"""
    High-order Gaussian laser pulse expressed in the Laguerre-Gaussian formalism.

    Derived class for an analytic profile.
    More precisely, at focus (`z_foc=0`), the transverse envelope (to be used in the
    :class:CombinedLongitudinalTransverseLaser class) corresponds to:

    .. math::

        \mathcal{T}(x, y) = r^{|m|}e^{-im\theta} \,
        L_p^{|m|}\left( \frac{2 r^2 }{w_0^2}\right )\,
        \exp\left( -\frac{r^2}{w_0^2} \right)

    where :math:`x = r \cos{\theta}`,
    :math:`y = r \sin{\theta}`, :math:`L_p^{|m|}` is the
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
    wavelength : float (in meter)
        The main laser wavelength :math:`\\lambda_0` of the laser.
    z_foc : float (in meter), optional
        Position of the focal plane. (The laser pulse is initialized at `z=0`.)
    """

    def __init__(self, w0, p, m, wavelength, z_foc=0):
        super().__init__()
        self.w0 = w0
        self.p = p
        self.m = m
        self.z_foc_over_zr = z_foc * wavelength / (np.pi * w0**2)

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
        # Term for wavefront curvature, waist and Gouy phase
        diffract_factor = 1.0 - 1j * self.z_foc_over_zr
        w = self.w0 * abs(diffract_factor)
        psi = np.angle(diffract_factor)
        # complex_position corresponds to r e^{+/-i\theta}
        if self.m > 0:
            complex_position = x - 1j * y
        else:
            complex_position = x + 1j * y
        radius = abs(complex_position)
        envelope = (
            complex_position ** abs(self.m)
            * genlaguerre(self.p, abs(self.m))(2 * radius**2 / w**2)
            * np.exp(
                -(radius**2) / (self.w0**2 * diffract_factor)
                - 1.0j * (2 * self.p + self.m) * psi
            )  # Additional Gouy phase
            * (1.0 / diffract_factor)
        )

        return envelope
