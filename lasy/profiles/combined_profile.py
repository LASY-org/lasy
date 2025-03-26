from .profile import Profile


class CombinedLongitudinalTransverseProfile(Profile):
    r"""
    Class that combines a longitudinal and transverse laser profile.

    The combined profile is defined as the product of the longitudinal and transverse
    profile.

    More precisely, the electric field corresponds to:

    .. math::

        E_u(\boldsymbol{x}_\perp,t) = Re\left[ E_0\, \mathcal{T}(x, y)
        \times \mathcal{L}(t) e^{-i\omega_0 t} \times p_u \right]

    where :math:`u` is either :math:`x` or :math:`y`, :math:`p_u` is
    the polarization vector, :math:`Re` represent the real part.
    The other parameters in this formula are defined below.

    Parameters
    ----------
    wavelength : float (in meter)
        The main laser wavelength :math:`\lambda_0` of the laser, which
        defines :math:`\omega_0` in the above formula, according to
        :math:`\omega_0 = 2\pi c/\lambda_0`.

    pol : list of 2 complex numbers (dimensionless)
        Polarization vector. It corresponds to :math:`p_u` in the above
        formula ; :math:`p_x` is the first element of the list and
        :math:`p_y` is the second element of the list. Using complex
        numbers enables elliptical polarizations.

    long_profile : :class:`.LongitudinalProfile`
        Defines the longitudinal envelope of the laser, i.e. the
        function :math:`\mathcal{L}(t)` in the above formula.

    transverse_profile : :class:`.TransverseProfile`
        Defines the transverse envelope of the laser, i.e. the
        function :math:`\mathcal{T}(x, y)` in the above formula.

    laser_energy : float (in Joule)
        The total energy of the laser pulse. The amplitude of the laser
        field (:math:`E_0` in the above formula) is automatically
        calculated so that the pulse has the prescribed energy.

    peak_fluence : float (in J/m^2)
        The peak fluence of the laser pulse. The amplitude of the laser
        field (:math:`E_0` in the above formula) is automatically
        calculated so that the pulse has the prescribed peak fluence.
        Required for a CW (monochromatic) profile.

    peak_power : float (in W)
        The peak power of the laser pulse. The amplitude of the laser
        field (:math:`E_0` in the above formula) is automatically
        calculated so that the pulse has the prescribed peak power.
        Required for a plane wave profile.
    """

    def __init__(
        self,
        wavelength,
        pol,
        long_profile,
        trans_profile,
        laser_energy=None,
        peak_fluence=None,
        peak_power=None,
    ):
        super().__init__(wavelength, pol)
        assert (
            (laser_energy is not None)
            ^ (peak_fluence is not None)
            ^ (peak_power is not None)
        ), "Exactly one of laser_energy, peak_fluence, or peak_power must be specified"
        if long_profile.is_cw:
            assert peak_fluence is not None
            self.peak_fluence = peak_fluence
            self.__update_is_cw__(long_profile.is_cw)

        elif trans_profile.is_plane_wave:
            assert peak_power is not None
            self.peak_power = peak_power
            self.__update_is_plane_wave__(trans_profile.is_plane_wave)
        else:
            assert laser_energy is not None
            self.laser_energy = laser_energy
        self.long_profile = long_profile
        self.trans_profile = trans_profile

    def evaluate(self, x, y, t):
        """
        Return the envelope field of the laser.

        Parameters
        ----------
        x, y, t : ndarrays of floats
            Define points on which to evaluate the envelope
            These arrays need to all have the same shape.

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y, t
        """
        envelope = self.trans_profile.evaluate(x, y) * self.long_profile.evaluate(t)
        return envelope
