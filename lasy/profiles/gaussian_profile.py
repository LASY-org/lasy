from . import CombinedLongitudinalTransverseProfile
from .longitudinal import GaussianLongitudinalProfile
from .transverse import GaussianTransverseProfile


class GaussianProfile(CombinedLongitudinalTransverseProfile):
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

    cep_phase : float (in radian), optional
        The Carrier Envelope Phase (CEP), i.e. :math:`\phi_{cep}`
        in the above formula (i.e. the phase of the laser
        oscillation, at the time where the laser envelope is maximum)

    z_foc : float (in meter), optional
        Position of the focal plane. (The laser pulse is initialized at `z=0`.)

    beta : float (in second), optional
        The angular dispersion parameterized by

    .. math::

            \beta = \frac{d\theta_0}{d\omega}
            
        Here :math:`\theta_0` is the propagation angle of this component.

    phi2 : float (in second^2), optional (default '0')
        The group-delay dispertion parameterized by

    .. math::

            \phi^{(2)} = \frac{dt}{d\omega}

    zeta : float (in meter * second) optional (default '0')
        The spatio-chirp parameterized by

    .. math::

         \zeta = \frac{x_0}{d\omega}
         
        Here :math:`x_0` is the beam center position.

    stc_theta :  float (in rad) optional (default '0')
        Transeverse direction along which spatio-temperal field couples.
        0 is along x axis.


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
        cep_phase=0,
        z_foc=0,
        phi2=0,
        beta=0,
        zeta=0,
        stc_theta=0,
    ):
        super().__init__(
            wavelength,
            pol,
            laser_energy,
            GaussianLongitudinalProfile(
                wavelength,
                tau,
                t_peak,
                cep_phase,
                beta=beta,
                phi2=phi2,
                zeta=zeta,
                stc_theta=stc_theta,
                w0=w0,
                z_foc=z_foc,
            ),
            GaussianTransverseProfile(w0, z_foc, wavelength),
        )
