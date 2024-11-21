from math import factorial

import numpy as np
from scipy.special import hermite

from .transverse_profile import TransverseProfile


class HermiteGaussianTransverseProfile(TransverseProfile):
    r"""
    A high-order Gaussian laser pulse expressed in the Hermite-Gaussian formalism.

    Definition is according to Siegman "Lasers" pg. 646 eq. 60, as explicitly given
    in https://doi.org/10.1364/JOSAB.489884 eq. 3, with the beam center upon the
    optical axis, :math:`x_0,y_0 = (0,0)`.

    More precisely, the transverse envelope (to be used in the
    :class:`.CombinedLongitudinalTransverseLaser` class) corresponds to:

    .. math::
        \mathcal{T}(x, y) = \,
                \mathcal{H}_m (x) \, \mathcal{H}_n(y) \, \exp(i \left ( \Phi_x(z) + \Phi_y(z)\right ) )

    with

    .. math::
        \mathcal{H}_p(q) = A_p h_p \left ( \frac{\sqrt{2}q}{w_q(z)} \right) \exp{\left( -\frac{q^2}{w_q^2(z)}\right)} \exp{ \left ( -i k_0 \frac{q^2}{2 R_q(z)} \right )}

        \Phi_q(z) = \left(p+\frac{1}{2}\right) \arctan\left({\frac{z}{Z_q}}\right)

        w_q(z) = w_{0,q} \sqrt{1 + \left( \frac{z}{Z_q}\right)^2}

        Z_q = \frac{\pi w_{0,q}^2}{\lambda_0}

        A_p = \frac{1}{\sqrt{w_q(z) 2^{p-1/2} p!\sqrt{\pi}}}

        R_q(z) = z + \frac{Z_q^2}{z}





    where  :math:`h_{p}` is the Hermite polynomial of order :math:`p`, :math:`w_q(z)` is the
    spot size of the laser along the :math:`q` axis (:math:`q` is :math:`x` or :math:`y`), :math:`Z_q` is the corresponding Rayleigh
    length and, :math:`\lambda_0` and :math:`k_0` are the wavelength and central wavenumber of the
    laser respectively.


    The z-dependence shown in the above equations is required to correctly define the
    electric field of the transverse profile relative to that of the pulse at the focus.
    The absolute z position will be overwritten when creating a laser object.

    Parameters
    ----------
    w_0x : float (in meter)
        The waist of the laser pulse in the x direction,
    w_0y : float (in meter)
        The waist of the laser pulse in the y direction,
    m : int (dimensionless)
        The order of hermite polynomial in the x direction
    n : int (dimensionless)
        The order of hermite polynomial in the y direction
    wavelength : float (in meter), optional
        The main laser wavelength :math:`\lambda_0` of the laser.
    z_foc : float (in meter), optional
        Position of the focal plane. (The laser pulse is initialized at
        ``z=0``.)

    Warnings
    --------
    In order to initialize the pulse out of focus, you can either:

    - Use a non-zero ``z_foc``
    - Use ``z_foc=0`` (i.e. initialize the pulse at focus) and then call
      ``laser.propagate(-z_foc)``

    Both methods are in principle equivalent, but note that the first
    method uses the paraxial approximation, while the second method does
    not make this approximation.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from lasy.profiles.transverse.hermite_gaussian_profile import (
    ...     HermiteGaussianTransverseProfile,
    ... )
    >>> # Create evaluation grid
    >>> xy = np.linspace(-30e-6, 30e-6, 200)
    >>> X, Y = np.meshgrid(xy, xy)
    >>> # Create an array of plots
    >>> fig, ax = plt.subplots(3, 6, figsize=(10, 5), tight_layout=True)
    >>> extent = (1e6 * xy[0], 1e6 * xy[-1], 1e6 * xy[0], 1e6 * xy[-1])
    >>> for m in range(3):
    >>>     for n in range(3):
    >>>         transverse_profile = HermiteGaussianTransverseProfile(
    ...             w_0x = 10e-6, # m
    ...             w_0y = 15e-6, # m
    ...             m = m, #
    ...             n = n, #
    ...             wavelength = 0.8e-6, # m
    ...         )
    ...         intensity = np.abs(transverse_profile.evaluate(X,Y))**2
    ...         vmax_intensity = np.max(intensity)
    >>>         ax[m,n].imshow(intensity,extent=extent,cmap='bone_r',vmin=0,vmax=vmax_intensity)
    >>>         ax[m,n].set_title('Inten: m,n = %i,%i' %(m,n))
    >>>         phase = np.angle(transverse_profile.evaluate(X,Y))
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



    """

    def __init__(self, w_0x, w_0y, m, n, wavelength, z_foc=0):
        super().__init__()
        self.w_0x = w_0x
        self.w_0y = w_0y
        self.m = m
        self.n = n
        self.wavelength = wavelength
        self.z_foc = z_foc
        z_eval = -z_foc  # this links our observation position to Siegmann's definition

        self.k0 = 2 * np.pi / wavelength

        # Calculate Rayleigh Lengths
        Zx = np.pi * w_0x**2 / wavelength
        Zy = np.pi * w_0y**2 / wavelength

        # Calculate Size at Location Z
        wxZ = w_0x * np.sqrt(1 + (z_eval / Zx) ** 2)
        wyZ = w_0y * np.sqrt(1 + (z_eval / Zy) ** 2)

        # Calculate Multiplicative Factors
        Anx = 1 / np.sqrt(wxZ * 2 ** (m - 1 / 2) * factorial(m) * np.sqrt(np.pi))
        Any = 1 / np.sqrt(wyZ * 2 ** (n - 1 / 2) * factorial(n) * np.sqrt(np.pi))

        # Calculate the Phase contributions from propagation
        phiXz = (m + 1 / 2) * np.arctan2(z_eval, Zx)
        phiYz = (n + 1 / 2) * np.arctan2(z_eval, Zy)

        self.z_eval = z_eval
        self.Zx = Zx
        self.Zy = Zy
        self.wxZ = wxZ
        self.wyZ = wyZ
        self.Anx = Anx
        self.Any = Any
        self.phiXz = phiXz
        self.phiYz = phiYz

    def _evaluate(self, x, y):
        """
        Return the transverse envelope.

        Parameters
        ----------
        x, y : ndarrays of floats
            Define points on which to evaluate the envelope
            These arrays need to all have the same shape.

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y
        """
        z_eval = self.z_eval
        Zx = self.Zx
        Zy = self.Zy
        k0 = self.k0
        wxZ = self.wxZ
        wyZ = self.wyZ
        Anx = self.Anx
        Any = self.Any
        m = self.m
        n = self.n
        phiXz = self.phiXz
        phiYz = self.phiYz

        # Calculate the HG in each plane
        HGnx = (
            Anx
            * hermite(m)(np.sqrt(2) * (x) / wxZ)
            * np.exp(-((x) ** 2) / wxZ**2)
            * np.exp(-1j * k0 * (x) ** 2 / 2 / (z_eval**2 + Zx**2) * z_eval)
        )
        HGny = (
            Any
            * hermite(n)(np.sqrt(2) * (y) / wyZ)
            * np.exp(-((y) ** 2) / wyZ**2)
            * np.exp(-1j * k0 * (y) ** 2 / 2 / (z_eval**2 + Zy**2) * z_eval)
        )

        # Put it altogether
        envelope = HGnx * HGny * np.exp(1j * (phiXz + phiYz))

        return envelope
