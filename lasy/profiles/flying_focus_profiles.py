from math import factorial

import numpy as np
from scipy.special import genlaguerre, hermite

from lasy.profiles.profile import Profile


class ParaxialFlyingFocusGaussianProfile(Profile):
    r"""
    Class for the analytic profile of a flying focus Gaussian laser pulse in 2 and 3 dimensions.

    More precisely, at focus (``z_foc=0``), the transverse and longtudinal envelope
    corresponds to:
    .. math::

        E_{3D}(x, y, t) =
            \Rea \left(
                \frac{iz_r}{q}
                \exp\left(-ik\frac{x^2+y^2}{2q}
                - \left(\frac{(t-t_p)^2}{\tau^2}\right)^{n_{order}/2}
                + i(\phi_{cep}+\text{omega}_0t_p)\right)
            \right)
        E_{2D}(x, t) =
            \Rea \left(
                \sqrt{\frac{iz_r}{q}}
                \exp\left(-ik\frac{x^2}{2q}
                - \left(\frac{(t-t_p)^2}{\tau^2}\right)^{n_{order}/2}
                + i(\phi_{cep}+\text{omega}_0t_p)\right)
            \right)

    Where z_r is the Rayleigh Range, and q is the complex factor:
    .. math::
        z_r = \frac{\pi w_0^2}{\lambda}
        q = z_{init} + iz_r

    Parameters
    ----------
    n_dims : int
        The number of dimensions to render
        Choose 2 (x, t) or 3 (x, y, t)

    w_0 : float (in meter)
        The waist of the laser pulse, i.e. :math:`w_0` in the above formula.

    wavelength : float (in meter)
        The main laser wavelength :math:`\lambda_0` of the laser.
        Which defines :math:`\omega_0` in the above formula, according to
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

    tau : float (in second)
        The duration of the laser pulse, i.e. :math:`\tau` in the above
        formula. Note that :math:`\tau = \tau_{FWHM}/\sqrt{2\log(2)}`,
        where :math:`\tau_{FWHM}` is the Full-Width-Half-Maximum duration
        of the intensity distribution of the pulse.

    t_peak : float (in second)
        The time at which the laser envelope reaches its maximum amplitude,
        i.e. :math:`t_{peak}` in the above formula.

    vf : float (in meters / second), optional
        The velocity of the point of peak intensity in the pulse
        Default value is 0

    cep_phase : float (in radian), optional
        The Carrier Envelope Phase (CEP), i.e. :math:`\phi_{cep}`
        in the above formula (i.e. the phase of the laser
        oscillation, at the time where the laser envelope is maximum)

    z_init : float (in meter), optional
        Initial position of the focal plane. (The laser pulse is initialized at
        ``z=0``.)

    n_order : int, optional
        the exponent for the super gaussian time envelope
        default is two (standard gaussian)


    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from lasy.laser import Laser
    >>> from lasy.profiles.flying_focus_profiles import FlyingFocusGaussianProfile
    >>> from lasy.utils.laser_utils import get_full_field
    >>> from scipy.constants import c
    >>> # Create profile.
    >>> profile = FlyingFocusGaussianProfile(
    ...     n_dims=3 # 3 Dimensional case
    ...     wavelength=0.6e-6,  # m
    ...     pol=(1, 0),
    ...     laser_energy=1.0,  # J
    ...     w0=5e-6,  # m
    ...     tau=30e-15,  # s
    ...     t_peak=0.0,  # s
            vf=0.5*c
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
        n_dims,
        w_0,
        wavelength,
        pol,
        laser_energy,
        tau,
        t_peak,
        vf=0,
        cep_phase=0,
        z_init=0,
        n_order=2,
    ):
        super().__init__(wavelength, pol)
        self.n_dims = n_dims  # maybe add a check
        self.w_0 = w_0
        self.wavelength = wavelength
        self.laser_energy = laser_energy
        self.tau = tau
        self.t_peak = t_peak
        self.vf = vf
        self.cep_phase = cep_phase
        self.z_init = z_init
        self.n_order = n_order

    def evaluate(self, x, y, t):
        """
        Return the transverse and longitudinal envelope.

        Parameters
        ----------
        t : ndarrays of floats
            Define longitudinal points on which to evaluate the envelope

        x,y : ndarrays of floats
            Define transverse points on which to evaluate the envelope

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y, t
        """
        # Rayleigh Range
        z_r = np.pi * self.w_0**2 / self.wavelength
        # complex beam parameter (z + iz_r)
        q = self.z_init - self.vf * (t - self.t_peak) + (1.0j * z_r)
        # wavenumber
        k0 = 2 * np.pi / self.wavelength

        # transverse radius squared
        if self.n_dims == 2:
            r_sq = x**2
        else:
            r_sq = x**2 + y**2

        # Term for wavefront curvature + Gouy phase
        diffract_factor = np.pow((1.0j * z_r / q), (self.n_dims / 2 - 0.5))

        # Calculate the argument of the complex exponential
        exp_argument = -1.0j * k0 * r_sq / 2 / q
        # Get the profile
        envelope = (
            np.exp(exp_argument)  # transverse envelope
            * np.exp(
                -np.power(((t - self.t_peak) ** 2) / self.tau**2, self.n_order / 2)
            )  # longitudal envelope
            * np.exp(
                1.0j * (self.cep_phase + self.omega0 * self.t_peak)
            )  # phase factor
            * diffract_factor  # normalization
        )

        return envelope


class ParaxialFlyingFocusHermiteGaussianProfile(Profile):
    r"""
    A high-order flying focus Gaussian laser pulse expressed in the Hermite-Gaussian formalism.

    Definition is according to Siegman "Lasers" pg. 646 eq. 60, as explicitly given
    in https://doi.org/10.1364/JOSAB.489884 eq. 3, with the beam center upon the
    optical axis, :math:`x_0,y_0 = (0,0)`.

    More precisely, the combined transverse and longitdual envelope corresponds to:
    .. math::
        \mathcal{T_{3D}}(x, y, t) = \,
                \mathcal{H}_m (x) \, \mathcal{H}_n(y) \, \exp(i \left ( \Phi_x(z) + \Phi_y(z) -(\frac{(t-t_{peak})^2}{\tau^2})^{n_{order}/2}+i(\phi_{cep} + w_0t_{peak})\right ) )

        \mathcal{T_{2D}}(x, t) = \,
                \mathcal{H}_m (x) \, \exp(i \left ( \Phi_x(z) -(\frac{(t-t_{peak})^2}{\tau^2})^{n_{order}/2}+i(\phi_{cep} + w_0t_{peak})\right ) )

    with

    .. math::
        \mathcal{H}_p(q) = A_p h_p \left ( \frac{\sqrt{2}q}{w_q(z)} \right) \exp{\left( -\frac{q^2}{w_q^2(z)}\right)} \exp{ \left ( -i k_0 \frac{q^2}{2 R_q(z)} \right )}

        \Phi_q(z) = \left(p+\frac{1}{2}\right) \arctan\left({\frac{z_init - v_f(t-t_{peak})}{Z_q}}\right)

        w_q(z) = w_{0,q} \sqrt{1 + \left( \frac{z_init - v_f(t-t_{peak})}{Z_q}\right)^2}

        Z_q = \frac{\pi w_{0,q}^2}{\lambda_0}

        A_p = \frac{1}{\sqrt{w_q(z) 2^{p-1/2} p!\sqrt{\pi}}}

        R_q(z) = z_{init} - v_f{t-t_{peak}} + \frac{Z_q^2}{z_{init} - v_f(t-t_{peak})}





    where  :math:`h_{p}` is the Hermite polynomial of order :math:`p`, :math:`w_q(z)` is the
    spot size of the laser along the :math:`q` axis (:math:`q` is :math:`x` or :math:`y`), :math:`Z_q` is the corresponding Rayleigh
    length and, :math:`\lambda_0` and :math:`k_0` are the wavelength and central wavenumber of the
    laser respectively.

    Parameters
    ----------
    n_dims : int
        The number of dimensions to render
        Choose 2 (x, t) or 3 (x, y, t)

    w_0x : float (in meter)
        The waist of the laser pulse in the x direction

    w_0y : float (in meter)
        The waist of the laser pulse in the y direction

    m : int (dimensionless)
        The order of hermite polynomial in the x direction

    n : int (dimensionless)
        The order of hermite polynomial in the y direction

    wavelength : float (in meter)
        The main laser wavelength :math:`\lambda_0` of the laser.
        Which defines :math:`\omega_0` in the above formula, according to
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

    tau : float (in second)
        The duration of the laser pulse, i.e. :math:`\tau` in the above
        formula. Note that :math:`\tau = \tau_{FWHM}/\sqrt{2\log(2)}`,
        where :math:`\tau_{FWHM}` is the Full-Width-Half-Maximum duration
        of the intensity distribution of the pulse.

    t_peak : float (in second)
        The time at which the laser envelope reaches its maximum amplitude,
        i.e. :math:`t_{peak}` in the above formula.

    vf : float (in meters / second), optional
        The velocity of the point of peak intensity in the pulse
        Default value is 0

    cep_phase : float (in radian), optional
        The Carrier Envelope Phase (CEP), i.e. :math:`\phi_{cep}`
        in the above formula (i.e. the phase of the laser
        oscillation, at the time where the laser envelope is maximum)

    z_init : float (in meter), optional
        Position of the initial focal plane. (The laser pulse is initialized at
        ``z=0``.)

    n_order : int, optional
        the exponent for the super gaussian time envelope
        default is two (standard gaussian)

    Examples
    --------
    # Transverse Profile
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from lasy.profiles.flying_focus_profiles import FlyingFocusHGProfile
    >>> from scipy.constants import c
    >>> # Create evaluation grid
    >>> xy = np.linspace(-30e-6, 30e-6, 200)
    >>> X, Y = np.meshgrid(xy, xy)
    >>> # Create an array of plots
    >>> fig, ax = plt.subplots(3, 6, figsize=(10, 5), tight_layout=True)
    >>> extent = (1e6 * xy[0], 1e6 * xy[-1], 1e6 * xy[0], 1e6 * xy[-1])
    >>> for m in range(3):
    >>>     for n in range(3):
    >>>         ff_hg_profile = FlyingFocusHGProfile(
    ...             n_dims = 3 # 3D case
    ...             w_0x = 10e-6, # m
    ...             w_0y = 15e-6, # m
    ...             m = m, #
    ...             n = n, #
    ...             wavelength = 0.8e-6, # m
    ...             pol= (1,0) ,
    ...             laser_energy = 1, # J
    ...             tau = 30e-12, # s
    ...             t_peak = 0.0, # s
    ...             vf = 0.5*c , # m/s
    ...         )
    ...         intensity = np.abs(ff_hg_profile.evaluate(X,Y,0))**2
    ...         vmax_intensity = np.max(intensity)
    >>>         ax[m,n].imshow(intensity,extent=extent,cmap='bone_r',vmin=0,vmax=vmax_intensity)
    >>>         ax[m,n].set_title('Inten: m,n = %i,%i' %(m,n))
    >>>         phase = np.angle(ff_hg_profile.evaluate(X,Y,0))
    ...         vmax_phase = np.max(np.abs(phase))
    >>>         ax[m,n+3].imshow(phase,extent=extent,cmap='seismic',vmin=-vmax_phase,vmax=vmax_phase)
    >>>         ax[m,n+3].set_title('Phase: m,n = %i,%i' %(m,n))
    >>>         if m==2:
    >>>             ax[m,n].set_xlabel("x (µm)")
    >>>             ax[m,n+3].set_xlabel("x (µm)")
    >>>         else:
    >>>             ax[m,n].set_xticks([])
    >>>             ax[m,n+3].set_xticks([])
    >>>         if n==0:
    >>>             ax[m,n].set_ylabel("y (µm)")
    >>>             ax[m,n+3].set_yticks([])
    >>>         else:
    >>>             ax[m,n].set_yticks([])
    >>>             ax[m,n+3].set_yticks([])

    # Longitudal profile
    >>> import matplotlib.pyplot as plt
    >>> from lasy.laser import Laser
    >>> from lasy.profiles.flying_focus_profiles import FlyingFocusHGProfile
    >>> from lasy.utils.laser_utils import get_full_field
    >>> from scipy.constants import c
    >>> # Create profile.
    >>> profile = FlyingFocusHGProfile(
    ...     n_dims = 3 # 3 Dimensional case
    ...     w_0x = 10e-6, # m
    ...     w_0y = 15e-6, # m
    ...     m = 3, #
    ...     n = 2, #
    ...     wavelength = 0.8e-6, # m
    ...     pol = (1,0) ,
    ...     laser_energy = 1, # J
    ...     tau = 30e-12, # s
    ...     t_peak = 0.0, # s
    ...     vf = 0.5*c , # m/s
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
        n_dims,
        w_0x,
        w_0y,
        m,
        n,
        wavelength,
        pol,
        laser_energy,
        tau,
        t_peak,
        vf=0,
        cep_phase=0,
        z_init=0,
        n_order=2,
    ):
        super().__init__(wavelength, pol)
        self.n_dims = n_dims
        self.w_0x = w_0x
        self.w_0y = w_0y
        self.m = m
        self.n = n
        self.wavelength = wavelength
        self.pol = pol
        self.laser_energy = laser_energy
        self.tau = tau
        self.t_peak = t_peak
        self.vf = vf
        self.cep_phase = cep_phase
        self.z_init = z_init
        self.n_order = n_order

    def evaluate(self, x, y, t):
        """
        Return the transverse envelope.

        Parameters
        ----------
        t : ndarrays of floats
            Define longitudinal points on which to evaluate the envelope

        x,y : ndarrays of floats
            Define transverse points on which to evaluate the envelope

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y, t

        """
        # Calculation for x terms
        # z value for flying focus
        z_eval = self.z_init - self.vf * (
            t - self.t_peak
        )  # this links our observation position to Siegmann's definition
        # wavenumber
        k0 = 2 * np.pi / self.wavelength
        # Calculate Rayleigh Lengths
        Zx = np.pi * self.w_0x**2 / self.wavelength
        # Calculate Size at Location Z
        wxZ = self.w_0x * np.sqrt(1 + (z_eval / Zx) ** 2)
        # Calculate Multiplicative Factors
        Anx = 1 / np.sqrt(
            wxZ * 2 ** (self.m - 1 / 2) * factorial(self.m) * np.sqrt(np.pi)
        )
        # Calculate the Phase contributions from propagation
        phiXz = (self.m + 1 / 2) * np.arctan2(z_eval, Zx)

        HGnx = (
            Anx
            * hermite(self.m)(np.sqrt(2) * (x) / wxZ)
            * np.exp(-((x) ** 2) / wxZ**2)
            * np.exp(-1j * k0 * (x) ** 2 / 2 / (z_eval**2 + Zx**2) * z_eval)
        )

        # calculations for y terms, skipped for 2D
        if self.n_dims > 2:
            Zy = np.pi * self.w_0y**2 / self.wavelength
            wyZ = self.w_0y * np.sqrt(1 + (z_eval / Zy) ** 2)
            Any = 1 / np.sqrt(
                wyZ * 2 ** (self.n - 1 / 2) * factorial(self.n) * np.sqrt(np.pi)
            )
            phiYz = (self.n + 1 / 2) * np.arctan2(z_eval, Zy)

            HGny = (
                Any
                * hermite(self.n)(np.sqrt(2) * (y) / wyZ)
                * np.exp(-((y) ** 2) / wyZ**2)
                * np.exp(-1j * k0 * (y) ** 2 / 2 / (z_eval**2 + Zy**2) * z_eval)
            )
        # terms for 2D where these values are constants
        else:
            HGny = 1
            phiYz = 0

        # Put it altogether
        envelope = (
            HGnx  # x transverse envelope
            * HGny  # y transverse envelope
            * np.exp(1j * (phiXz + phiYz))  # guoy terms
            * np.exp(
                -np.power(((t - self.t_peak) ** 2) / self.tau**2, self.n_order / 2)
            )  # longitudal envelope
            * np.exp(1.0j * (self.cep_phase + self.omega0 * self.t_peak))  # phase
        )

        return envelope


class ParaxialFlyingFocusLaguerreGaussianProfile(Profile):
    r"""
    A high-order flying focus Gaussian laser pulse expressed in the Laguerre-Gaussian formalism.

    Definition is according to Siegman "Lasers" pg. 646 eq. 64.

    More precisely, the transverse and longtudinal envelope corresponds to:
    .. math::
        \mathcal{T}(x, y, t) = \,
                \mathcal{L}_{p,m} (x) \, \exp(i l\Phi -(\frac{(t-t_{peak})^2}{\tau^2})^{n_{order}/2}+i(\phi_{cep} + w_0t_{peak}))

    with

    .. math::
        \mathcal{L}_{p,m}(x) = A \left ( \frac{\sqrt{2}r}{w(z)} \right)^m l_{p,m} \left ( \frac{2 r^2}{w^2(z)} \right)
        \exp{\left( -\frac{r^2}{w^2(z)}\right)} \exp{ \left ( -i k_0 \frac{r^2}{2 R(z)} \right )}

        w(z) = w_{0} \sqrt{1 + \left( \frac{z_{init} - v_f(t-t_{peak})}{Z_R}\right)^2}

        A = \frac{1}{w(z)} \sqrt{\frac{2 p!}{\pi (p + m)!}}

        R(z) = z_{init} - v_f(t-t_{peak}) + \frac{Z_R^2}{z_{init} - v_f(t-t_{peak})}

        \Phi(z) = \left(2 p + m + 1\right) \arctan\left({\frac{z_{init} - v_f(t-t_{peak})}{Z_R}}\right)

        Z_R = \frac{\pi w_0^2}{\lambda_0}


    where  :math:`l_{p,m}` is the Laguerre polynomial of radial order :math:`p` and azimuthal order :math:`m`.

    The z-depedence shown in the above equations is required to correctly define the
    electric field of the transverse profile relative to that of the pulse at the focus.
    The absolute z position will be overwritten when creating a laser object.

    Parameters
    ----------
    w_0 : float (in meter)
        The waist of the laser pulse,
        i.e. :math:`w_{0}` in the above formula.

    p : int (dimensionless)
        The order of Laguerre polynomial in the x direction
        i.e. :math:`m` in the above formula.
    m : int (dimensionless)
        The order of Laguerre polynomial in the y direction
        i.e. :math:`n` in the above formula.

    wavelength : float (in meter)
        The main laser wavelength :math:`\lambda_0` of the laser.
        Which defines :math:`\omega_0` in the above formula, according to
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

    tau : float (in second)
        The duration of the laser pulse, i.e. :math:`\tau` in the above
        formula. Note that :math:`\tau = \tau_{FWHM}/\sqrt{2\log(2)}`,
        where :math:`\tau_{FWHM}` is the Full-Width-Half-Maximum duration
        of the intensity distribution of the pulse.

    t_peak : float (in second)
        The time at which the laser envelope reaches its maximum amplitude,
        i.e. :math:`t_{peak}` in the above formula.

    vf : float (in meters / second), optional
        The velocity of the point of peak intensity in the pulse
        Default value is 0

    cep_phase : float (in radian), optional
        The Carrier Envelope Phase (CEP), i.e. :math:`\phi_{cep}`
        in the above formula (i.e. the phase of the laser
        oscillation, at the time where the laser envelope is maximum)

    z_init : float (in meter), optional
        Position of the initial focal plane. (The laser pulse is initialized at
        ``z=0``.)

    n_order : int, optional
        the exponent for the super gaussian time envelope
        default is two (standard gaussian)


    Examples
    --------
    # Transverse Profile
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from lasy.flying_focus_profiles import FlyingFocusLGProfile
    >>> from scipy.constants import c
    >>> # Create evaluation grid
    >>> xy = np.linspace(-30e-6, 30e-6, 200)
    >>> X, Y = np.meshgrid(xy, xy)
    >>> # Create an array of plots
    >>> fig, ax = plt.subplots(3, 6, figsize=(10, 5), tight_layout=True)
    >>> extent = (1e6 * xy[0], 1e6 * xy[-1], 1e6 * xy[0], 1e6 * xy[-1])
    >>> for p in range(3):
    >>>     for m in range(3):
    >>>         ff_lg_profile = FlyingFocusLGProfile(
    ...             w_0 = 10e-6, # m
    ...             p = p, #
    ...             m = m, #
    ...             wavelength = 0.8e-6, # m
    ...             pol = (1,0),
    ...             laser_energy = 1, # J
    ...             tau = 30e-12, # s
    ...             t_peak = 0.0, # s
    ...             vf = 0.5 * c, # m/s
    ...         )
    ...         intensity = np.abs(ff_lg_profile.evaluate(X,Y,0))**2
    ...         vmax_intensity = np.max(intensity)
    >>>         ax[p,m].imshow(intensity,extent=extent,cmap='bone_r',vmin=0,vmax=vmax_intensity)
    >>>         ax[p,m].set_title('Inten: p,m = %i,%i' %(p,m))
    >>>         phase = np.angle(ff_lg_profile.evaluate(X,Y,0))
    ...         vmax_phase = np.max(np.abs(phase))
    >>>         ax[p,m+3].imshow(phase,extent=extent,cmap='seismic',vmin=-vmax_phase,vmax=vmax_phase)
    >>>         ax[p,m+3].set_title('Phase: p,m = %i,%i' %(p,m))
    >>>         if p==2:
    >>>             ax[p,m].set_xlabel("x (µm)")
    >>>             ax[p,m+3].set_xlabel("x (µm)")
    >>>         else:
    >>>             ax[p,m].set_xticks([])
    >>>             ax[p,m+3].set_xticks([])
    >>>         if m==0:
    >>>             ax[p,m].set_ylabel("y (µm)")
    >>>             ax[p,m+3].set_yticks([])
    >>>         else:
    >>>             ax[p,m].set_yticks([])
    >>>             ax[p,m+3].set_yticks([])


    # Longitudal profile
    >>> import matplotlib.pyplot as plt
    >>> from lasy.laser import Laser
    >>> from lasy.profiles.flying_focus_profiles import FlyingFocusLGProfile
    >>> from lasy.utils.laser_utils import get_full_field
    >>> from scipy.constants import c
    >>> # Create profile.
    >>> profile = FlyingFocusLGProfile(
    ...     w_0=10e-6,  # m
    ...     p=3,  #
    ...     m=2,  #
    ...     wavelength=0.8e-6,  # m
    ...     pol=(1, 0),
    ...     laser_energy=1,  # J
    ...     tau=30e-12,  # s
    ...     t_peak=0.0,  # s
    ...     vf=0.5 * c,  # m/s
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
        w_0,
        p,
        m,
        wavelength,
        pol,
        laser_energy,
        tau,
        t_peak,
        vf=0,
        cep_phase=0,
        z_init=0,
        n_order=2,
    ):
        super().__init__(wavelength, pol)
        self.w_0 = w_0
        self.p = p
        self.m = m
        self.wavelength = wavelength
        self.pol = pol
        self.laser_energy = laser_energy
        self.tau = tau
        self.t_peak = t_peak
        self.vf = vf
        self.cep_phase = cep_phase
        self.z_init = z_init
        self.n_order = n_order

    def evaluate(self, x, y, t):
        """
        Return the transverse envelope.

        Parameters
        ----------
        t : ndarrays of floats
            Define longitudinal points on which to evaluate the envelope

        x,y : ndarrays of floats
            Define transverse points on which to evaluate the envelope

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y, t
        """
        # z eval
        z_eval = self.z_init - self.vf * (
            t - self.t_peak
        )  # this links our observation position to Siegmann's definition
        # wavenumber
        k0 = 2 * np.pi / self.wavelength
        # Calculate Rayleigh Length
        z_r = np.pi * self.w_0**2 / self.wavelength
        # Calculate Size at Location Z
        w0Z = self.w_0 * np.sqrt(1 + (z_eval / z_r) ** 2)
        # Calculate Multiplicative Factors
        A = (
            np.sqrt(2.0 * factorial(self.p) / (np.pi * factorial(self.m + self.p)))
            / w0Z
        )
        # Calculate the Phase contributions from propagation
        phiZ = (2.0 * self.p + self.m + 1) * np.arctan2(z_eval, z_r)

        # Calculate the LG in each plane
        LG = (
            A
            * (np.sqrt(2.0) * np.sqrt(x**2 + y**2) / w0Z) ** np.abs(self.m)
            * genlaguerre(self.p, np.abs(self.m))(2.0 * (x**2 + y**2) / w0Z**2)
            * np.exp(-(x**2 + y**2) / w0Z**2)
            * np.exp(-1j * k0 * (x**2 + y**2) / 2 / (z_eval**2 + z_r**2) * z_eval)
        )

        # Put it altogether
        envelope = (
            LG  # transverse envelope and laguerre constants
            * np.exp(1j * phiZ)  # guoy phase
            * np.exp(
                -np.power(((t - self.t_peak) ** 2) / self.tau**2, self.n_order / 2)
            )  # longitdual envelope
            * np.exp(-1j * self.m * np.arctan2(y, x))  # orbital angular momentum
            * np.exp(1j * (self.cep_phase + self.omega0 * self.t_peak))  # phase
        )

        return envelope
