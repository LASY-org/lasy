import numpy as np

from .profile import Profile


class SpaceTimeProfile(Profile):
    r"""
    Class that can evaluate a pulse that has certain space-time couplings

    More precisely, the electric field corresponds to:

        .. math::

            E_u(\\boldsymbol{x}_\\perp,t) = Re\\left[ E_0\\,
            \\exp\\left(-\\frac{\\boldsymbol{x}_\\perp^2}{w_0^2}
            - \\frac{(t-t_{peak}+2ibx/w_0^2)^2}{\\tau_{eff}^2}
            - i\\omega_0(t-t_{peak}) + i\\phi_{cep}\\right) \\times p_u \\right]

        where :math:`u` is either :math:`x` or :math:`y`, :math:`p_u` is
        the polarization vector, :math:`Re` represent the real part.
        The other parameters in this formula are defined below.

        Parameters
        ----------
        wavelength: float (in meter)
            The main laser wavelength :math:`\\lambda_0` of the laser, which
            defines :math:`\\omega_0` in the above formula, according to
            :math:`\\omega_0 = 2\\pi c/\\lambda_0`.

        pol: list of 2 complex numbers (dimensionless)
            Polarization vector. It corresponds to :math:`p_u` in the above
            formula ; :math:`p_x` is the first element of the list and
            :math:`p_y` is the second element of the list. Using complex
            numbers enables elliptical polarizations.

        laser_energy: float (in Joule)
            The total energy of the laser pulse. The amplitude of the laser
            field (:math:`E_0` in the above formula) is automatically
            calculated so that the pulse has the prescribed energy.

        tau: float (in second)
            The duration of the laser pulse, i.e. :math:`\\tau` in the above
            formula. Note that :math:`\\tau = \\tau_{FWHM}/\\sqrt{2\\log(2)}`,
            where :math:`\\tau_{FWHM}` is the Full-Width-Half-Maximum duration
            of the intensity distribution of the pulse.

        w0: float (in meter)
            The waist of the laser pulse, i.e. :math:`w_0` in the above formula.

        sc: spatial chirp, b in the above formula, that results in a mixing
            of the longitudinal and transverse profiles. Must be in units
            of [x/omega]. An imaginary value for this property is possible,
            which would represent a pulse-front tilt in the focus.
            A representative real value is b = w0 * tau.

        t_peak: float (in second)
            The time at which the laser envelope reaches its maximum amplitude,
            i.e. :math:`t_{peak}` in the above formula.

        cep_phase: float (in radian), optional
            The Carrier Enveloppe Phase (CEP), i.e. :math:`\\phi_{cep}`
            in the above formula (i.e. the phase of the laser
            oscillation, at the time where the laser envelope is maximum)
    """

    def __init__(self, wavelength, pol, laser_energy, w0, tau, sc, t_peak, cep_phase=0):
        super().__init__(wavelength, pol)
        self.laser_energy = laser_energy
        self.w0 = w0
        self.tau = tau
        self.sc = sc
        self.t_peak = t_peak
        self.cep_phase = cep_phase

    def evaluate(self, x, y, t):
        """
        Returns the envelope field of the laser

        Parameters
        ----------
        x, y, t: ndarrays of floats
            Define points on which to evaluate the envelope
            These arrays need to all have the same shape.

        Returns
        -------
        envelope: ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y, t
        """
        transverse = np.exp(-(x**2 + y**2) / self.w0**2)

        tau_eff = np.sqrt(self.tau**2 + (2 * self.sc / self.w0) ** 2)

        spacetime = np.exp(
            -((t - self.t_peak + (2 * 1j * self.sc * x / self.w0**2)) ** 2)
            / tau_eff**2
        )

        oscillatory = *np.exp(1.0j * (self.cep_phase - self.omega0 * (t - self.t_peak)))

        envelope = transverse * spacetime * oscillatory

        return envelope
