from math import factorial

import numpy as np
from scipy.special import genlaguerre, binom

from .transverse_profile import TransverseProfile


class FlattenedGaussianTransverseProfile(TransverseProfile):
    r"""
    Define a complex transverse profile with a flattened Gaussian intensity
    distribution **far from focus** that transform into a distribution
    with rings **in the focal plane**. (See `Santarsiero et al., J.
    Modern Optics, 1997 <http://doi.org/10.1080/09500349708232927>`_)

    Increasing the parameter ``N`` increases the
    flatness of the transverse profile **far from focus**,
    and increases the number of rings **in the focal plane**.

    **In the focal plane** (:math:`z=z_f`), the profile translates to a
    laser with a transverse electric field:

    .. math::

        E(x,y,z=zf) \propto
        \exp\left(-\frac{r^2}{(N+1)w_0^2}\right)
        \sum_{n=0}^N c'_n L^0_n\left(\frac{2\,r^2}{(N+1)w_0^2}\right)

        \mathrm{with} Laguerre polynomials :math:`L^0_n` and
        \qquad c'_n = \sum_{m=n}^{N}\frac{1}{2^m}\binom{m}{n}

    - For :math:`N=0`, this is a Gaussian profile: :math:`E\propto\exp\left(-\frac{r^2}{w_0^2}\right)`.

    - For :math:`N\rightarrow\infty`, this is a Jinc profile: :math:`E\propto \frac{J_1(r/w_0)}{r/w_0}`.

    Parameters
    ----------
    w_0 : float (in meter)
        The waist of the laser pulse,
        i.e. :math:`w_{0}` in the above formula.
    N: int
        Determines the "flatness" of the transverse profile, far from
        focus (see the above formula).
        Default: ``N=6`` ; somewhat close to an 8th order supergaussian.

    p : int (dimensionless)
        The order of Laguerre polynomial in the x direction
        i.e. :math:`m` in the above formula.
    m : int (dimensionless)
        The order of Laguerre polynomial in the y direction
        i.e. :math:`n` in the above formula.
    wavelength : float (in meter)
        The main laser wavelength :math:`\lambda_0` of the laser.
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

    def __init__(self, w_0, N, wavelength, z_foc=0):
        super().__init__()
        self.w_0 = w_0
        # Ensure that N is an integer
        self.N = int(round(N))
        self.wavelength = wavelength
        self.z_foc = z_foc        
        self.z_eval = -z_foc  # this links our observation position to Siegman's definition
        self.k0 = 2 * np.pi / wavelength
        # Calculate Rayleigh Length
        zr = np.pi * w_0**2 / wavelength
        self.inv_zr = 1. / zr
        # Calculate effective waist of the Laguerre-Gauss modes, at focus
        self.w_foc = w_0*(self.N + 1)**.5

        # Calculate the coefficients for the Laguerre-Gaussian modes
        self.cn = np.empty(self.N+1)
        for n in range(self.N+1):
            m_values = np.arange(n, self.N+1)
            self.cn[n] = np.sum((1./2)**m_values * binom(m_values,n)) / (self.N+1)


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
        z_eval = self.z_eval
        diffract_factor = 1. + 1j * self.z_eval * self.inv_zr
        w = self.w_foc * np.abs( diffract_factor )
        psi = np.angle( diffract_factor )
        # Argument for the Laguerre polynomials
        scaled_radius_squared = 2*( x**2 + y**2 ) / w**2

        # Sum recursively over the Laguerre polynomials
        laguerre_sum = np.zeros_like( x, dtype=np.complex128 )
        for n in range(0, self.N+1):
            # Recursive calculation of the Laguerre polynomial
            # - `L` represents $L_n$
            # - `L1` represents $L_{n-1}$
            # - `L2` represents $L_{n-2}$
            if n==0:
                L = 1.
            elif n==1:
                L1 = L
                L = 1. - scaled_radius_squared
            else:
                L2 = L1
                L1 = L
                L = (((2*n -1) - scaled_radius_squared) * L1 - (n - 1) * L2) / n
            # Add to the sum, including the term for the additional Gouy phase
            laguerre_sum += self.cn[n] * np.exp( - (2j* n) * psi ) * L

        # Final profile: multiply by n-independent propagation factors
        exp_argument = - (x**2 + y**2) / (self.w_foc**2 * diffract_factor)
        envelope = laguerre_sum * np.exp( exp_argument ) / diffract_factor

        return envelope
