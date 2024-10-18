import numpy as np

from .longitudinal_profile import LongitudinalProfile


class GaussianLongitudinalProfile(LongitudinalProfile):
    r"""
    Class for the analytic profile of a longitudinally-Gaussian laser pulse.

    More precisely, the longitudinal envelope
    (to be used in the :class:`.CombinedLongitudinalTransverseProfile` class)
    corresponds to:

    .. math::

        \mathcal{L}(t) = \exp\left( - \frac{(t-t_{peak})^2}{\tau^2}
                        + i\omega_0t_{peak} \right)

    Parameters
    ----------
    tau : float (in second)
        The duration of the laser pulse, i.e. :math:`\tau` in the above
        formula. Note that :math:`\tau = \tau_{FWHM}/\sqrt{2\log(2)}`,
        where :math:`\tau_{FWHM}` is the Full-Width-Half-Maximum duration
        of the intensity distribution of the pulse.

    t_peak : float (in second)
        The time at which the laser envelope reaches its maximum amplitude,
        i.e. :math:`t_{peak}` in the above formula.

    cep_phase : float (in radian), optional(default '0')
        The Carrier Enveloppe Phase (CEP), i.e. :math:`\phi_{cep}`
        in the above formula (i.e. the phase of the laser
        oscillation, at the time where the laser envelope is maximum).

    beta : float (in second), optional
        The angular dispersion parameterized by
        .. math::
            \beta = \frac{d\theta_0}{d\omega}
        Here :math:`\theta_0` is the propagation angle of this component.

    phi2 : float (in second^2), optional (default '0')
        The group-delay dispertion parameterized by
        .. math::
            \phi^{(2)} = \frac{dt}{d\omega}

    zeta : float (in meter * second) optional (defalut '0')
        The spatio-chirp parameterized by
        .. math::
         \zeta = \frac{x_0}{d\omega}
        Here :math:`x_0` is the beam center position.

    stc_theta :  float (in rad) optional (default '0')
        Transeverse direction along which spatio-temperal field couples.
        0 is along x axis.

    z_foc : float (in meter), necessary if beta is not 0
        Position of the focal plane. (The laser pulse is initialized at
        ``z=0``.)

     w0 : float (in meter), necessary if beta is not 0
        The waist of the laser pulse.
    """

    def __init__(
        self,
        wavelength,
        tau,
        t_peak,
        cep_phase=0,
        beta=0,
        phi2=0,
        zeta=0,
        stc_theta=0,
        w0=0,
        z_foc=0,
    ):
        super().__init__(wavelength)
        self.tau = tau
        self.t_peak = t_peak
        self.cep_phase = cep_phase
        self.beta = beta
        self.phi2 = phi2
        self.zeta = zeta
        self.w0 = w0
        self.stc_theta = stc_theta
        if z_foc == 0:
            self.z_foc_over_zr = 0
        else:
            assert (
                wavelength is not None
            ), "You need to pass the wavelength, when `z_foc` is non-zero."
            self.z_foc_over_zr = z_foc * wavelength / (np.pi * w0**2)

    def evaluate(self, t, x=0, y=0):
        """
        Return the longitudinal envelope.

        Parameters
        ----------
        t : ndarrays of floats
            Define longitudinal points on which to evaluate the envelope

        x,y : ndarrays of floats, necessray if spatio-temperal coulping exists
            Define transverse points on which to evaluate the envelope

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the longitudinal envelope at the
            specified points. This array has the same shape as the array t.
        """
        inv_tau2 = self.tau ** (-2)
        inv_complex_waist_2 = 1.0 / (
            self.w0**2 * (1.0 + 2.0j * self.z_foc_over_zr / (self.k0 * self.w0**2))
        )
        stretch_factor = (
            1
            + 4.0
            * (self.zeta + self.beta * self.z_foc_over_zr * inv_tau2)
            * (self.zeta + self.beta * self.z_foc_over_zr * inv_complex_waist_2)
            + 2.0j
            * (self.phi2 - self.beta**2 * self.k0 * self.z_foc_over_zr)
            * inv_tau2
        )
        stc_exponent = (
            1.0
            / stretch_factor
            * inv_tau2
            * (
                t
                - self.t_peak
                - self.beta
                * self.k0
                * (x * np.cos(self.stc_theta) + y * np.sin(self.stc_theta))
                - 2.0j
                * (x * np.cos(self.stc_theta) + y * np.sin(self.stc_theta))
                * (self.zeta - self.beta * self.z_foc_over_zr)
                * inv_complex_waist_2
            )
            ** 2
        )
        envelope = np.exp(
            -stc_exponent + 1.0j * (self.cep_phase + self.omega0 * (t - self.t_peak))
        )
        return envelope
