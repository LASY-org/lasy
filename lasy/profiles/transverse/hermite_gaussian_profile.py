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
    wavelength : float (in meter), optional
        The main laser wavelength :math:`\lambda_0` of the laser.
        (Only needed if ``z_foc`` is different than 0.)
    z_foc : float (in meter), optional
        Position of the focal plane. (The laser pulse is initialized at
        ``z=0``.)

    Warnings
    --------
    In order to initialize the pulse out of focus, you can either:

    - Use a non-zero ``z_foc``
    - Use ``z_foc=0`` (i.e. initialize the pulse at focus) and then call
      ``laser.propagate(-z_foc)``

    Both methods are in principle equivalent, but note that the first
    method uses the paraxial approximation, while the second method does
    not make this approximation.
    """

    def __init__(self, w0, n_x, n_y, wavelength=None, z_foc=0):
        super().__init__()
        self.w0 = w0
        self.n_x = n_x
        self.n_y = n_y
        if z_foc == 0:
            self.z_foc_over_zr = 0
        else:
            assert (
                wavelength is not None
            ), "You need to pass the wavelength, when `z_foc` is non-zero."
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

        envelope = (
            np.sqrt(2 / np.pi)
            * np.sqrt(1 / (2 ** (self.n_x) * factorial(self.n_x) * self.w0))
            * np.sqrt(1 / (2 ** (self.n_y) * factorial(self.n_y) * self.w0))
            * hermite(self.n_x)(np.sqrt(2) * x / w)
            * hermite(self.n_y)(np.sqrt(2) * y / w)
            * np.exp(
                -(x**2 + y**2) / (self.w0**2 * diffract_factor)
                - 1.0j * (self.n_x + self.n_y) * psi
            )
            # Additional Gouy phase
            * (1.0 / diffract_factor)
        )
        return envelope
