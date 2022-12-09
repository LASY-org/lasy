from .profile import Profile
import numpy as np
from scipy.special.orthogonal import hermite

class HermiteGaussianProfile(Profile):
    """
    Derived class for an analytic profile of a high-order Gaussian
    laser pulse expressed in the Hermite-Gaussian formalism.
    """

    def __init__(self, wavelength, pol,
                laser_energy, w0, n_x, n_y, tau, t_peak, cep_phase=0):
        """
        Defines a Hermite-Gaussian laser pulse.
        More precisely, the electric field corresponds to:
        .. math::
            E_u(\boldsymbol{x}_\perp,t) = Re\left[ E_0\,
            H_{n_x}\left ( \frac{\sqrt{2} x}{w_0}\right )\,
            H_{n_y}\left ( \frac{\sqrt{2} y}{w_0}\right )\,
            \exp\left( -\frac{\boldsymbol{x}_\perp^2}{w_0^2}
            - \frac{(t-t_{peak})^2}{\tau^2} -i\omega_0(t-t_{peak})
            + i\phi_{cep}\right) \times p_u \right]
        where :math:`u` is either :math:`x` or :math:`y`, :math:`H_{n}` is the
        Hermite polynomial of order :math:`n`, :math:`p_u` is the polarization
        vector, :math:`Re` represent the real part, and
        :math:`\boldsymbol{x}_\perp` is the transverse coordinate (orthogonal
        to the propagation direction). The other parameters in this formula
        are defined below.

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
        n_x: int (dimensionless)
            The order of hermite polynomial in the x direction
        n_y: int (dimensionless)
            The order of hermite polynomial in the y direction
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
        self.n_x = n_x
        self.n_y = n_y
        self.tau = tau
        self.t_peak = t_peak
        self.cep_phase = cep_phase

    def evaluate( self, envelope, box ):
        """
        Fills the envelope field of the laser

        Parameters
        -----------
        envelope: ndarrays (V/m)
            Contains the values of the envelope field, to be filled
        box: an object of type lasy.utils.Box
            Defines the points at which evaluate the laser
        """
        t = box.axes[-1]
        long_profile = np.exp( -(t-self.t_peak)**2/self.tau**2 \
                            + 1.j*(self.cep_phase + self.omega0*self.t_peak))

        if box.dim == 'xyt':
            x = box.axes[0]
            y = box.axes[1]
            transverse_profile = hermite(self.n_x)(
                np.sqrt(2)*x[:,np.newaxis]/self.w0) * hermite(self.n_y)(
                    np.sqrt(2)*y[np.newaxis,:]/self.w0) * np.exp(
                    -(x[:,np.newaxis]**2 + y[np.newaxis, :]**2)/self.w0**2 )
            envelope[...] = transverse_profile[:,:,np.newaxis] * \
                    long_profile[np.newaxis, np.newaxis, :]
        elif box.dim == 'rt':
            r = box.axes[0]
            """
            For these modes to make sense in rt, we need to
            have multiple modes and likely represent as a
            series of Laguerre Gauss Modes for accurate
            reresentation. Until then, we pass a simple gaussian
            and a warning
            """
            assert ((self.n_x ==0) and (self.n_y == 0 )),"Modes for \
                n_x > 0 or n_y > 0 are not yet implemented in dimension rt"

            transverse_profile = np.exp( -r**2/self.w0**2 )

            envelope[0,:,:] = transverse_profile[:,np.newaxis] * \
                            long_profile[np.newaxis, :]
