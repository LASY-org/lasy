import numpy as np

from .profile import Profile


class GaussianProfile(Profile):
    r"""
    Class for the analytic profile of a Gaussian laser pulse.

    This includes space-time couplings: pulse-front tilt and spatial chirp

    More precisely, the electric field corresponds to:

    .. math::

        E_u(\\boldsymbol{x}_\\perp,t) = Re\\left[ E_0\\,
        \\exp\\left(-\\frac{\\boldsymbol{x}_\\perp^2}{w_0^2}
        - \\frac{(t-t_{peak}-ax+2ibx/w_0^2)^2}{\\tau_{eff}^2}
        - i\\omega_0(t-t_{peak}) + i\\phi_{cep}\\right) \\times p_u \\right]

    where :math:`u` is either :math:`x` or :math:`y`, :math:`p_u` is
    the polarization vector, :math:`Re` represent the real part, and
    :math:`\boldsymbol{x}_\perp` is the transverse coordinate (orthogonal
    to the propagation direction). The other parameters in this formula
    are defined below.

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

    laser_energy : float (in Joule)
        The total energy of the laser pulse. The amplitude of the laser
        field (:math:`E_0` in the above formula) is automatically
        calculated so that the pulse has the prescribed energy.

    w0 : float (in meter)
        The waist of the laser pulse, i.e. :math:`w_0` in the above formula.

    tau : float (in second)
        The duration of the laser pulse, i.e. :math:`\tau` in the above
        formula. Note that :math:`\tau = \tau_{FWHM}/\sqrt{2\log(2)}`,
        where :math:`\tau_{FWHM}` is the Full-Width-Half-Maximum duration
        of the intensity distribution of the pulse.

    t_peak : float (in second)
        The time at which the laser envelope reaches its maximum amplitude,
        i.e. :math:`t_{peak}` in the above formula.

    a: float (in second/meter), optional
        Pulse-front tilt, i.e. :math:`a` in the above formula, that results in the laser arrival
        time varying as a function of `x`. A representative real value is a = tau / w0.

    b: float (in meter.second), optional
        Spatial chirp, i.e. :math:`b` in the above formula, that results in the laser frequency
        varying as a function of `x`. A representative real value is b = w0 * tau.

    GDD: float (in second.second), optional
        Group-delay dispersion, i.e. :math:`gdd` in the formula for tau_eff, that results
        in temporal chirp. A representative real value is gdd = tau * tau.

    cep_phase : float (in radian), optional
        The Carrier Envelope Phase (CEP), i.e. :math:`\phi_{cep}`
        in the above formula (i.e. the phase of the laser
        oscillation, at the time where the laser envelope is maximum)

    z_foc : float (in meter), optional
        Position of the focal plane. (The laser pulse is initialized at `z=0`.)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from lasy.laser import Laser
    >>> from lasy.profiles.gaussian_profile import GaussianProfile
    >>> from lasy.utils.laser_utils import get_full_field
    >>> # Create profile.
    >>> profile = GaussianProfile(
    ...     wavelength=0.6e-6,  # m
    ...     pol=(1, 0),
    ...     laser_energy=1.,  # J
    ...     w0=5e-6,  # m
    ...     tau=30e-15,  # s
    ...     t_peak=0.  # s
    ... )
    >>> # Create laser with given profile in `rt` geometry.
    >>> laser = Laser(
    ...     dim="rt",
    ...     lo=(0e-6, -60e-15),
    ...     hi=(10e-6, +60e-15),
    ...     npoints=(50, 400),
    ...     profile=profile
    ... )
    >>> # Visualize field.
    >>> E_rt, extent = get_full_field(laser)
    >>> extent[2:] *= 1e6
    >>> extent[:2] *= 1e15
    >>> tmin, tmax, rmin, rmax = extent
    >>> vmax = np.abs(E_rt).max()
    >>> plt.imshow(
    ...     E_rt,
    ...     origin="lower",
    ...     aspect="auto",
    ...     vmax=vmax,
    ...     vmin=-vmax,
    ...     extent=[tmin, tmax, rmin, rmax],
    ...     cmap='bwr',
    ... )
    >>> plt.xlabel('t (fs)')
    >>> plt.ylabel('r (µm)')
    """

    def __init__(
        self,
        wavelength,
        pol,
        laser_energy,
        w0,
        tau,
        t_peak,
        a=0.0,
        b=0.0,
        gdd=0.0,
        cep_phase=0.0,
        z_foc=0.0,
    ):
        super().__init__(wavelength, pol)
        self.laser_energy = laser_energy
        self.w0 = w0
        self.tau = tau
        self.t_peak = t_peak
        self.a = a
        self.b = b
        self.gdd = gdd
        self.cep_phase = cep_phase
        self.z_foc = z_foc

    def evaluate(self, x, y, t):
        """
        Return the envelope field of the laser.

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

        tau_eff = np.sqrt(
            self.tau**2 + (2 * self.b / self.w0) ** 2 + 2 * 1j * self.gdd
        )

        spacetime = np.exp(
            -(
                (t - self.t_peak - self.a * x + (2 * 1j * self.b * x / self.w0**2))
                ** 2
            )
            / tau_eff**2
        )

        oscillatory = np.exp(1.0j * (self.cep_phase - self.omega0 * (t - self.t_peak)))

        envelope = transverse * spacetime * oscillatory

        return envelope
