from .laser import Laser
import numpy as np

class GaussianLaser(Laser):
    """
    Derived class for the analytic profile of a Gaussian laser pulse.
    """

    def __init__(self, box, wavelength, pol,
                laser_energy, w0, tau, t_peak, cep_phase=0):
        """
        Defines a Gaussian laser pulse.

        More precisely, the electric field corresponds to:

        .. math::

            E_u(\boldsymbol{x}_\perp,t) = Re\left[ E_0\,
            \exp\left( -\frac{\boldsymbol{x}_\perp^2}{w_0^2}
            - \frac{(t-t_{peak})^2}{\tau^2} -i\omega_0(t-t_{peak})
            + i\phi_{cep}\right) \times p_u \right]

        where :math:`u` is either :math:`x` or :math:`y`, :math:`p_u` is
        the polarization vector, :math:`Re` represent the real part, and
        :math:`\boldsymbol{x}_\perp` is the transverse coordinate (orthogonal
        to the propagation direction). The other parameters in this formula
        are defined below.

        Parameters:
        -----------
        box: an object of type lasy.utils.box.Box
            Defines the grid over which the laser will be computed

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
        super().__init__(box, wavelength, pol)

        t = box.axes[-1]
        long_profile = np.exp( -(t-t_peak)**2/tau**2 \
                               + 1.j*(cep_phase + self.omega0*t_peak) )

        if self.dim == 'xyt':
            x = box.axes[0]
            y = box.axes[1]
            transverse_profile = np.exp(
                            -(x[:,np.newaxis]**2 + y[np.newaxis, :]**2)/w0**2 )
            self.field.field[...] = transverse_profile[:,:,np.newaxis] * \
                                      long_profile[np.newaxis, np.newaxis, :]
        elif self.dim == 'rt':
            r = box.axes[0]
            transverse_profile = np.exp( -r**2/w0**2 )
            # Store field purely in mode 0
            self.field.field[0,:,:] = transverse_profile[:,np.newaxis] * \
                                      long_profile[np.newaxis, :]

        # Normalize to the correct energy
        current_energy = self._compute_laser_energy()
        norm_factor = (laser_energy/current_energy)**.5
        self.field.field *= norm_factor