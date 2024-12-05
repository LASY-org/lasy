from math import factorial

import numpy as np
from scipy.special import genlaguerre

from .transverse_profile import TransverseProfile


class LaguerreGaussianTransverseProfile(TransverseProfile):
    r"""
    A high-order Gaussian laser pulse expressed in the Laguerre-Gaussian formalism.

    Definition is according to Siegman "Lasers" pg. 646 eq. 64.

    More precisely, the transverse envelope (to be used in the
    :class:`.CombinedLongitudinalTransverseLaser` class) corresponds to:

    .. math::
        \mathcal{T}(x, y) = \,
                \mathcal{L}_{p,m} (x) \, \exp(i \Phi)

    with

    .. math::
        \mathcal{L}_{p,m}(x) = A \left ( \frac{\sqrt{2}r}{w(z)} \right)^m l_{p,m} \left ( \frac{2 r^2}{w^2(z)} \right)
        \exp{\left( -\frac{r^2}{w^2(z)}\right)} \exp{ \left ( -i k_0 \frac{r^2}{2 R(z)} \right )}

        w(z) = w_{0} \sqrt{1 + \left( \frac{z}{Z_R}\right)^2}

        A = \frac{1}{w(z)} \sqrt{\frac{2 p!}{\pi (p + m)!}}

        R(z) = z + \frac{Z_R^2}{z}

        \Phi(z) = \left(2 p + m + 1\right) \arctan\left({\frac{z}{Z_R}}\right)

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
    >>> from lasy.profiles.transverse.laguerre_gaussian_profile import (
    ...     LaguerreGaussianTransverseProfile,
    ... )
    >>> # Create evaluation grid
    >>> xy = np.linspace(-30e-6, 30e-6, 200)
    >>> X, Y = np.meshgrid(xy, xy)
    >>> # Create an array of plots
    >>> fig, ax = plt.subplots(3, 6, figsize=(10, 5), tight_layout=True)
    >>> extent = (1e6 * xy[0], 1e6 * xy[-1], 1e6 * xy[0], 1e6 * xy[-1])
    >>> for p in range(3):
    >>>     for m in range(3):
    >>>         transverse_profile = LaguerreGaussianTransverseProfile(
    ...             w_0 = 10e-6, # m
    ...             p = p, #
    ...             m = m, #
    ...             wavelength = 0.8e-6, # m
    ...         )
    ...         intensity = np.abs(transverse_profile.evaluate(X,Y))**2
    ...         vmax_intensity = np.max(intensity)
    >>>         ax[p,m].imshow(intensity,extent=extent,cmap='bone_r',vmin=0,vmax=vmax_intensity)
    >>>         ax[p,m].set_title('Inten: p,m = %i,%i' %(p,m))
    >>>         phase = np.angle(transverse_profile.evaluate(X,Y))
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



    """

    def __init__(self, w_0, p, m, wavelength, z_foc=0):
        super().__init__()
        self.w_0 = w_0
        self.p = p
        self.m = m
        self.wavelength = wavelength
        self.z_foc = z_foc
        z_eval = -z_foc  # this links our observation position to Siegman's definition

        self.k0 = 2 * np.pi / wavelength

        # Calculate Rayleigh Length
        Zr = np.pi * w_0**2 / wavelength

        # Calculate Size at Location Z
        w0Z = w_0 * np.sqrt(1 + (z_eval / Zr) ** 2)

        # Calculate Multiplicative Factors
        A = np.sqrt(2.0 * factorial(p) / (np.pi * factorial(m + p))) / w0Z

        # Calculate the Phase contributions from propagation
        phiZ = (2.0 * p + m + 1) * np.arctan2(z_eval, Zr)

        self.z_eval = z_eval
        self.Zr = Zr
        self.w0Z = w0Z
        self.A = A
        self.phiZ = phiZ

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
        Zr = self.Zr
        k0 = self.k0
        w0Z = self.w0Z
        A = self.A
        p = self.p
        m = self.m
        phiZ = self.phiZ

        # Calculate the LG in each plane
        LG = (
            A
            * (np.sqrt(2.0) * np.sqrt(x**2 + y**2) / w0Z) ** (m)
            * genlaguerre(p, m)(2.0 * (x**2 + y**2) / w0Z**2)
            * np.exp(-(x**2 + y**2) / w0Z**2)
            * np.exp(-1j * k0 * (x**2 + y**2) / 2 / (z_eval**2 + Zr**2) * z_eval)
        )

        # Put it altogether
        envelope = LG * np.exp(1j * phiZ) * np.exp(-1j * m * np.arctan2(y, x))

        return envelope
