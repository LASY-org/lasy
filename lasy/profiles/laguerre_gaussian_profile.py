from .profile import Profile
import numpy as np
from scipy.special import genlaguerre

class LaguerreGaussianProfile(Profile):
    """
    Derived class for an analytic profile of a high-order Gaussian
    laser pulse expressed in the Laguerre-Gaussian formalism.
    """

    def __init__(self, wavelength, pol,
                laser_energy, w0, p, m, tau, t_peak, cep_phase=0):
        """
        Defines a Laguerre-Gaussian laser pulse.
        More precisely, the electric field corresponds to:
        .. math::
            E_u(r,\theta,t) = Re\left[ E_0\, r^{|m|}e^{-im\theta} \,
            L_p^{|m|}\left( \frac{2 r^2 }{w_0^2}\right )\,
            \exp\left( -\frac{r^2}{w_0^2}
            - \frac{(t-t_{peak})^2}{\tau^2} -i\omega_0(t-t_{peak})
            + i\phi_{cep}\right) \times p_u \right]
        where :math:`u` is either :math:`x = r \cos{\theta}` or
        :math:`y = r \sin{\theta}`, :math:`L_p^{|m|}` is the
        Generalised Laguerre polynomial of radial order :math:`p` and
        azimuthal order :math:`|m|`, :math:`p_u` is the polarization
        vector, :math:`Re` represent the real part, and :math:`r` is the radial
        coordinate (orthogonal to the propagation direction) and :math:`\theta`
        is the azmiuthal coordinate and :math:`t` is time. The other parameters
        in this formula are defined below.

        Parameters:
        -----------
        wavelength: float (in meter)
            The main laser wavelength :math:`\lambda_0` of the laser, which
            defines :math:`\omega_0` in the above formula, according to
            :math:`\omega_0 = 2\pi c/\lambda_0`.
        pol: list of 2 complex numbers (dimensionless)
            Polarization vector. It corresponds to :math:`p_u` in the above
            formula ; :math:`p_x` is the first element of the list and
            :math:`p_y` is the second element of the list. Using complex
            numbers enables elliptical polarizations.
        laser_energy: float (in Joule)
            The total energy of the laser pulse. The amplitude of the laser
            field (:math:`E_0` in the above formula) is automatically
            calculated so that the pulse has the prescribed energy.
        w0: float (in meter)
            The waist of the laser pulse, i.e. :math:`w_0` in the above formula.
        p: int (dimensionless)
            The radial order of Generalized Laguerre polynomial
        m: int (dimensionless)
            Defines the phase rotation, i.e. :math:`m` in the above formula.
        tau: float (in second)
            The duration of the laser pulse, i.e. :math:`\tau` in the above
            formula. Note that :math:`\tau = \tau_{FWHM}/\sqrt{2\log(2)}`,
            where :math:`\tau_{FWHM}` is the Full-Width-Half-Maximum duration
            of the intensity distribution of the pulse.
        t_peak: float (in second)
            The time at which the laser envelope reaches its maximum amplitude,
            i.e. :math:`t_{peak}` in the above formula.
        cep_phase: float (in radian), optional
            The Carrier Enveloppe Phase (CEP), i.e. :math:`\phi_{cep}`
            in the above formula (i.e. the phase of the laser
            oscillation, at the time where the laser envelope is maximum)
        """
        super().__init__(wavelength, pol)
        self.laser_energy = laser_energy
        self.w0 = w0
        self.p = p
        self.m = m
        self.tau = tau
        self.t_peak = t_peak
        self.cep_phase = cep_phase

    def evaluate( self, x, y, t ):
        """
        Returns the envelope field of the laser

        Parameters:
        -----------
        x, y, t: ndarrays of floats
            Define points on which to evaluate the envelope
            These arrays need to all have the same shape.

        Returns:
        --------
        envelope: ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y, t
        """
        long_profile = np.exp( -(t-self.t_peak)**2/self.tau**2 \
                              + 1.j*(self.cep_phase + self.omega0*self.t_peak))
        # complex_position corresponds to r e^{+/-i\theta}
        if self.m > 0:
            complex_position = x - 1j*y
        else:
            complex_position = x + 1j*y
        radius = abs(complex_position)
        scaled_rad_squared = (radius**2)/self.w0**2
        transverse_profile = complex_position**abs(self.m) * \
            genlaguerre(self.p, abs(self.m))(2*scaled_rad_squared) * \
            np.exp(-scaled_rad_squared)
        envelope = transverse_profile * long_profile

        return envelope
