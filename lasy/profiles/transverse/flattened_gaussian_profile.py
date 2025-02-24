import math

import numpy as np
from scipy.special import binom

from .transverse_profile import TransverseProfile


class FlattenedGaussianTransverseProfile(TransverseProfile):
    r"""
    Class for the analytic profile of a Flattened-Gaussian laser pulse.

    Define a complex transverse profile with a flattened Gaussian intensity
    distribution **far from focus** that transforms into a distribution
    with rings **in the focal plane**. (See `Santarsiero et al., J.
    Modern Optics, 1997 <http://doi.org/10.1080/09500349708232927>`_)

    Increasing the parameter ``N`` increases the
    flatness of the transverse profile **far from focus**,
    and increases the number of rings **in the focal plane**.

    The implementation of this class is based on that from `FBPIC`
    <https://github.com/fbpic/fbpic/blob/dev/fbpic/lpa_utils/laser/transverse_laser_profiles.py>.

    **In the focal plane** (:math:`z=z_f`), or in the far field, the profile translates to a
    laser with a transverse electric field:

    .. math::

        E(x,y,z=zf) \propto
        \exp\left(-\frac{r^2}{(N+1)w0^2}\right)
        \sum_{n=0}^N c'_n L^0_n\left(\frac{2\,r^2}{(N+1)w0^2}\right)

    <<<<<<< HEAD
    with Laguerre polynomials :math:`L^0_n` and
    =======

    with Laguerre polynomials :math:`L^0_n` and
    >>>>>>> b0b79d8f1dccaa42345c4eda0b5660faf52b09c0

    .. math::

        c'_n=\sum_{m=n}^{N}\frac{1}{2^m}\binom{m}{n}

    - For :math:`N=0`, this is a Gaussian profile: :math:`E\propto\exp\left(-\frac{r^2}{w0^2}\right)`.
    - For :math:`N\rightarrow\infty`, this is a Jinc profile: :math:`E\propto\frac{J_1(r/w0)}{r/w0}`.

    The equivalent expression for the collimated beam in the near field which produces this focus is
    given by:

    .. math::

        E(x,y,z=\infty) \propto
        \exp\left(-\frac{(N+1)r^2}{w(z)^2}\right)
        \sum_{n=0}^N \frac{1}{n!}\left(\frac{(N+1)\,r^2}{w(z)^2}\right)^n

    with

    .. math::

        w(z) = \frac{\lambda_0}{\pi w0}|z-z_{foc}|

    - Note that a beam defined using the near field definition would be
      equivalent to a beam defined with the corresponding parameters in
      the far field, but without the parabolic phase arising from being
      defined far from the focus.

    - For :math:`N=0`, this is a Gaussian profile: :math:`E\propto\exp\left(-\frac{r^2}{w(z)^2}\right)`.
    - For :math:`N\rightarrow\infty`, this is a flat profile: :math:`E\propto\Theta(w(z)-r)`.

    Parameters
    ----------
    field_type : string
        Options: 'nearfield', when the beam is defined far from focus and
        has been collimated, or 'farfield', when the beam is in the vicinity
        of or has been directly propagated from the focus. In this case there
        can be a large defocus in the spatial phase.

    w : float (in meter)
        The waist of the laser pulse. If ``field_type == 'farfield'`` then this
        variable corresponds to :math:`w_{0}` in the above far field formula.
        If ``field_type == 'nearfield'`` then this variable corresponds to
        :math:`w(z)` in the above near field formula.

    N : int
        Determines the "flatness" of the transverse profile, far from
        focus (see the above formula).
        Default: ``N=6`` ; somewhat close to an 8th order supergaussian.

    wavelength : float (in meter)
        The main laser wavelength :math:`\lambda_0` of the laser.

    z_foc : float (in meter), optional
        Only required if defining the pulse in the far field. Gives the position
        of the focal plane. (The laser pulse is initialized at ``z=0``.)

    Warnings
    --------

    In order to initialize the pulse out of focus, you can either:

    - Use a non-zero ``z_foc``.
    - Use ``z_foc=0`` (i.e., initialize the pulse at focus) and then call
      ``laser.propagate(-z_foc)``.

    Both methods are in principle equivalent, but note that the first
    method uses the paraxial approximation, while the second method does
    not make this approximation.
    """

    def __init__(self, field_type, w, N, wavelength, z_foc=0):
        super().__init__()
        # Ensure that N is an integer
        self.N = int(round(N))
        assert field_type in ["nearfield", "farfield"]
        self.field_type = field_type

        if field_type == "farfield":
            w0 = w
            # Calculate effective waist of the Laguerre-Gauss modes, at focus
            self.w_foc = w0 * (self.N + 1) ** 0.5
            # Calculate Rayleigh Length
            self.zr = np.pi * self.w_foc**2 / wavelength
            # Evaluation distance w.r.t focal position
            self.z_eval = z_foc
            # Calculate the coefficients for the Laguerre-Gaussian modes
            self.cn = np.empty(self.N + 1)
            for n in range(self.N + 1):
                m_values = np.arange(n, self.N + 1)
                self.cn[n] = np.sum((1.0 / 2) ** m_values * binom(m_values, n)) / (
                    self.N + 1
                )
        else:
            self.w = w

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
        if self.field_type == "farfield":
            # Term for wavefront curvature + Gouy phase
            diffract_factor = 1.0 - 1j * self.z_eval / self.zr
            w = self.w_foc * np.abs(diffract_factor)
            psi = np.angle(diffract_factor)
            # Argument for the Laguerre polynomials
            scaled_radius_squared = 2 * (x**2 + y**2) / w**2

            # Sum recursively over the Laguerre polynomials
            laguerre_sum = np.zeros_like(x, dtype=np.complex128)
            for n in range(0, self.N + 1):
                # Recursive calculation of the Laguerre polynomial
                # - `L` represents $L_n$
                # - `L1` represents $L_{n-1}$
                # - `L2` represents $L_{n-2}$
                if n == 0:
                    L = 1.0
                elif n == 1:
                    L1 = L
                    L = 1.0 - scaled_radius_squared
                else:
                    L2 = L1
                    L1 = L
                    L = (((2 * n - 1) - scaled_radius_squared) * L1 - (n - 1) * L2) / n
                # Add to the sum, including the term for the additional Gouy phase
                laguerre_sum += self.cn[n] * np.exp(-(2j * n) * psi) * L

            # Final envelope: multiply by n-independent propagation factors
            exp_argument = -(x**2 + y**2) / (self.w_foc**2 * diffract_factor)
            envelope = laguerre_sum * np.exp(exp_argument) / diffract_factor

            return envelope

        else:
            N = self.N
            w = self.w

            sumseries = 0
            if N > 0:
                for n in range(N):
                    sumseries += (
                        1 / math.factorial(n) * ((N + 1) * (x**2 + y**2) / w**2) ** n
                    )

            envelope = np.exp(-(N + 1) * (x**2 + y**2) / w**2) * sumseries

            return envelope
