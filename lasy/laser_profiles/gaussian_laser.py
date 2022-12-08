import numpy as np
from .laser_profile import LaserProfile

class GaussianLaser(LaserProfile):
    """
    Derived class for the analytic profile of a Gaussian laser pulse.
    """

    def __init__(self, wavelength, pol,
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
        super().__init__(wavelength, pol)
        self.laser_energy = laser_energy
        self.w0 = w0
        self.tau = tau
        self.t_peak = t_peak
        self.cep_phase = cep_phase

    def evaluate( self, dim, envelope, *axes ):
        """
        Fills the envelope field of the laser
        Usage: evaluate(dim, envelope, x, y, t) (3D Cartesian) or
               evaluate(dim, envelope, r, t) (2D cylindrical)
        t is always the last axis to be read.

        Example use cases:
            evaluate('rt', env, 0, 0) # evaluate at a single point (0,0)
            X, Y, T = np.meshgrid(x,y,t,indexing='ij') where x, y and t are 1darray-s
            evaluate('rt', env, X, Y, T)

        Parameters
        -----------
        dim: string
            'rt' or 'xyt'

        envelope: ndarrays (V/m)
            Contains the values of the envelope field, to be filled

        axes: Coordinates at which the envelope should be evaluated.
            Can be 2 elements in cylindrical geometry (r,t) or
            3 elements in Cartesian geometry (x,y,t).
            elements should be scalars or numpy arrays. In the latter case,
            all elements should have the same shape.
        """

        t = axes[-1]
        long_profile = np.exp( -(t-self.t_peak)**2/self.tau**2 \
                              + 1.j*(self.cep_phase + self.omega0*self.t_peak))

        if dim == 'xyt':
            assert( all(axis.shape == envelope.shape for axis in axes) )
            x = axes[0]
            y = axes[1]
            transverse_profile = np.exp( -(x**2 + y**2)/self.w0**2 )
            envelope[:] = transverse_profile * long_profile
        else:
            r = axes[0]
            transverse_profile = np.exp( -r**2/self.w0**2 )
            # Store field purely in mode 0
            envelope[0,:,:] = transverse_profile * long_profile
