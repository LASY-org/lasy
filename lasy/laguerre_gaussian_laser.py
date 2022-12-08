from ..utils.laser_energy import compute_laser_energy
from .laser_profile import LaserProfile
import numpy as np
from scipy.special import genlaguerre

class LaguerreGaussianLaser(LaserProfile):
    """
    Derived class for an analytic profile of a high-order Gaussian
    laser pulse expressed in the Laguerre-Gaussian formalism.
    """

    def __init__(self, wavelength, pol,
                laser_energy, w0, p, m, tau, t_peak, cep_phase=0,theta0=0):
        """
        Defines a Laguerre-Gaussian laser pulse.
        More precisely, the electric field corresponds to:
        .. math::
            E_u(r,\theta,t) = Re\left[ E_0\,
            L_p^m\left( \frac{2 r^2 }{w_0^2}\right )\, 
            \cos\left ( m(\theta-\theta_0)\right)\,
            \exp\left( -\frac{\boldsymbol{x}_\perp^2}{w_0^2}
            - \frac{(t-t_{peak})^2}{\tau^2} -i\omega_0(t-t_{peak})
            + i\phi_{cep}\right) \times p_u \right]
        where :math:`u` is either :math:`x` or :math:`y`, :math:`L_p^m` is the
        Generalised Laguerre polynomial of radial order :math:`p` and 
        azimuthal order :math:`m`, :math:`p_u` is the polarization 
        vector, :math:`Re` represent the real part, and :math:`r` is the radial 
        coordinate (orthogonal to the propagation direction) and :math:`theta` 
        is the azmiuthal coordinate. The other parameters in this formula
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
        p: int (dimensionless)
            The radial order of Generalized Laguerre polynomial 
        m: int (dimensionless)
            The azimuthal order of Generalized Laguerre polynomial 
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
        theta0: float (in radian), optional
            The azimuthal offset of the mode
        """
        super().__init__(wavelength, pol)
        self.laser_energy = laser_energy
        self.w0 = w0
        self.p = p 
        self.m = m 
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
            scaled_rad_squared = (x[:,np.newaxis]**2 + y[np.newaxis, :]**2)/ \
                self.w0**2
            theta = np.arctan2(y[np.newaxis, :],x[:,np.newaxis])
            angle_term = np.exp(1.j*self.m*(theta-self.theta0))
            transverse_profile = genlaguerre(self.p, self.m)(
                2*scaled_rad_squared) * angle_term * \
                np.exp(-scaled_rad_squared)
            envelope[...] = transverse_profile[:,:,np.newaxis] * \
                    long_profile[np.newaxis, np.newaxis, :]
        elif box.dim == 'rt':
            r = box.axes[0]
            if self.m > 0:
                import warnings
                warnings.warn(
                "In rt, Laguerre-Gauss is not currently defined for m > 0"
                )
            scaled_rad_squared = r**2/self.w0**2
            transverse_profile = genlaguerre(self.p, self.m)(
                2*scaled_rad_squared) * \
                np.exp( -scaled_rad_squared )
            # Store field purely in mode 0
            envelope[0,:,:] = transverse_profile[:,np.newaxis] * \
                            long_profile[np.newaxis, :]

        # Normalize to the correct energy
        current_energy = compute_laser_energy(envelope, box)
        norm_factor = (self.laser_energy/current_energy)**.5
        envelope *= norm_factor