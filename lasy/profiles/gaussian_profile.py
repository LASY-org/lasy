import numpy as np

from .profile import Profile


class GaussianProfile(Profile):
    r"""
    Class for the analytic profile of a Gaussian laser pulse.

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
    This profile also supports some chirp parameters that are omitted in the expression above for clarity.

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

    x0 , y0 : float (in meter) optional (default: '0')
        Transverse centroid for the laser pulse

    cep_phase : float (in radian), optional
        The Carrier Envelope Phase (CEP), i.e. :math:`\phi_{cep}`
        in the above formula (i.e. the phase of the laser
        oscillation, at the time where the laser envelope is maximum)

    z_foc : float (in meter), optional
        Position of the focal plane. (The laser pulse is initialized at `z=0`.)

    phi2 : float (in second^2), optional (default: '0')
        The group-delay dispersion defined as :math:`\phi^{(2)} = \frac{dt_0}{d\omega}`. Here :math:`t_0` is the temporal position of this frequency component.

    beta : float (in second), optional (default: '0')
        The angular dispersion defined as :math:`\beta = \frac{d\theta_0}{d\omega}`. Here :math:`\theta_0` is the propagation angle of this frequency component.

    zeta : float (in meter * second), optional (default: '0')
        A spatial chirp defined as :math:`\zeta = \frac{dx_0}{d\omega}`. Here :math:`x_0` is the transverse beam center position of this frequency component.
        The definitions of beta, phi2, and zeta are taken from [S. Akturk et al., Optics Express 12, 4399 (2004)].

    stc_theta : float (in radian), optional (default: '0')
        Transverse direction along which there are chirps and spatio-temporal couplings.
        A value of 0 corresponds to the x-axis.

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
    ...     laser_energy=1.0,  # J
    ...     w0=5e-6,  # m
    ...     tau=30e-15,  # s
    ...     t_peak=0.0,  # s
    ... )
    >>> # Create laser with given profile in `rt` geometry.
    >>> laser = Laser(
    ...     dim="rt",
    ...     lo=(0e-6, -60e-15),
    ...     hi=(10e-6, +60e-15),
    ...     npoints=(50, 400),
    ...     profile=profile,
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
    ...     cmap="bwr",
    ... )
    >>> plt.xlabel("t (fs)")
    >>> plt.ylabel("r (µm)")
    """

    def __init__(
        self,
        wavelength,
        pol,
        laser_energy,
        w0,
        tau,
        t_peak,
        x0=0,
        y0=0,
        cep_phase=0,
        z_foc=0,
        phi2=0,
        beta=0,
        zeta=0,
        stc_theta=0,
    ):
        super().__init__(wavelength, pol)
        self.laser_energy = laser_energy
        self.w0 = w0
        self.tau = tau
        self.t_peak = t_peak
        self.cep_phase = cep_phase
        self.z_foc = z_foc
        self.z_foc_over_zr = z_foc * wavelength / (np.pi * w0**2)
        self.phi2 = phi2
        self.beta = beta
        self.zeta = zeta
        self.stc_theta = stc_theta
        self.x0 = x0
        self.y0 = y0

    def evaluate(self, x, y, t):
        """
        Return the longitudinal envelope.

        Parameters
        ----------
        t : ndarrays of floats
            Define longitudinal points on which to evaluate the envelope

        x,y : ndarrays of floats
            Define transverse points on which to evaluate the envelope

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the longitudinal envelope at the
            specified points. This array has the same shape as the array t.
        """
        inv_tau2 = self.tau ** (-2)
        inv_complex_waist_2 = 1.0 / (
            self.w0**2 * (1.0 + 2.0j * self.z_foc / (self.k0 * self.w0**2))
        )
        stretch_factor = (
            1
            + 4.0
            * (-self.zeta + self.beta * self.z_foc)
            * inv_tau2
            * (-self.zeta + self.beta * self.z_foc)
            * inv_complex_waist_2
            + 2.0j * (-self.phi2 - self.beta**2 * self.k0 * self.z_foc) * inv_tau2
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
                * (-self.zeta - self.beta * self.z_foc)
                * inv_complex_waist_2
            )
            ** 2
        )
        # Term for wavefront curvature + Gouy phase
        diffract_factor = 1.0 - 1j * self.z_foc_over_zr
        # Calculate the argument of the complex exponential
        exp_argument = -((x - self.x0) ** 2 + (y - self.y0) ** 2) / (
            self.w0**2 * diffract_factor
        )
        # Get the profile
        envelope = (
            np.exp(
                -stc_exponent
                + exp_argument
                + 1.0j * (self.cep_phase + self.omega0 * self.t_peak)
            )
            / diffract_factor
        )

        return envelope
