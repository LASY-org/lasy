import copy

import numpy as np
from scipy.constants import c
from scipy.special import j0 as scipy_j0

from lasy.backend import xp

from .propagator import Propagator


class FresnelChirpZPropagatorRT(Propagator):
    r"""Class that represents a Fresnel propagator based upon the zoomed quasi-discrete Hankel transform.

    For an azimuthally symmetric scalar field :math:`E_0(r',0,\omega)`, the
    propagated field at distance :math:`z` under the Fresnel approximation is:

    .. math::

        E(r,z,\omega) =
        \frac{-ik}{z}
        \exp\!\left(ikz\right)
        \exp\!\left(\frac{ikr^2}{2z}\right)
        \int_0^\infty E_0(r',0,\omega)\,
        \exp\!\left(\frac{ikr'^2}{2z}\right)
        J_0\!\left(\frac{krr'}{z}\right) r' \,dr'

    This is derived from the standard 2D Huygens–Fresnel integral by integrating
    out the azimuthal angle, using
    :math:`\int_0^{2\pi} e^{-ik\rho\rho'\cos\theta/z}\,d\theta = 2\pi J_0(k\rho\rho'/z)`.

    The result can be written compactly as a scaled zeroth-order Hankel transform
    :math:`\mathcal{H}_0`:

    .. math::

        E(r,z,\omega) = G \times
        \mathcal{H}_0\!\left[E_0 \times H\right]\!\!\left(\frac{kr}{z}\right)

    where

    .. math::

        G = \frac{-ik}{z}\exp(ikz)\exp\!\left(\frac{ikr^2}{2z}\right),
        \qquad
        H = \exp\!\left(\frac{ikr'^2}{2z}\right).

    The Hankel transform is evaluated at the specific radial spatial frequencies
    :math:`kr/z` that map directly onto the desired output grid positions :math:`r`.
    This is the cylindrical counterpart of the Chirp-Z (zoom FFT) idea: instead of
    being restricted to the reciprocal of the input grid, the transform is sampled
    at an arbitrary set of output frequencies — here determined by the output grid.
    The transform is computed via direct quadrature (matrix–vector product with the
    :math:`J_0` Bessel kernel).

    Note that, consistent with lasy's co-moving frame convention, the overall
    carrier-wave phase :math:`\exp(ikz)` is cancelled by a matching
    :math:`\exp(-i\omega z/c)` correction applied after the loop.

    .. note::
        On a GPU (CuPy backend) the :math:`J_0` matrix is computed on the CPU
        with ``scipy.special`` and then transferred to the device.  For large
        grids this round-trip can be a bottleneck; a future version could use
        ``cupyx.scipy.special.j0`` for a fully on-device path.

    Parameters
    ----------
    (none required at construction; ``update`` is called by ``propagate``)

    Examples
    --------
    >>> from lasy.laser import Laser
    >>> from lasy.profiles.gaussian_profile import GaussianProfile
    >>> from lasy.optical_elements import ParabolicMirror
    >>> from lasy.propagators import FresnelChirpZPropagatorRT
    >>> from lasy.utils.grid import Grid
    >>> from lasy.backend import xp
    >>> # Create profile.
    >>> profile = GaussianProfile(
    ...     wavelength=0.8e-6,  # m
    ...     pol=(1, 0),
    ...     laser_energy=1.0,  # J
    ...     w0=5e-3,  # m
    ...     tau=30e-15,  # s
    ...     t_peak=0.0,  # s
    ... )
    >>> # Create laser with given profile in `rt` geometry.
    >>> laser = Laser(
    ...     dim="rt",
    ...     lo=(0, -60e-15),
    ...     hi=(15e-3, +60e-15),
    ...     npoints=(100, 500),
    ...     profile=profile,
    ... )
    >>> # Add a focusing parabolic mirror.
    >>> focal_length = 1  # m
    >>> laser.apply_optics(ParabolicMirror(focal_length))
    >>> # Attach the Fresnel Chirp-Z RT propagator.
    >>> laser.add_propagator(FresnelChirpZPropagatorRT())
    >>> # Build a finer output grid near the focal spot.
    >>> rLimNew = 150e-6  # m
    >>> newGrid = Grid(
    ...     laser.dim,
    ...     (0, laser.grid.lo[1]),
    ...     (rLimNew, laser.grid.hi[1]),
    ...     (50, laser.grid.npoints[1]),
    ... )
    >>> # Propagate to the focal plane and visualise.
    >>> laser.propagate(focal_length, grid_out=newGrid)
    >>> laser.show(envelope_type="intensity")
    >>> w0theory = 0.8e-6 * focal_length / (xp.pi * 5e-3)
    >>> print("w0 theoretical: %.2e m" % w0theory)
    """

    def update(self, dim, omega0):
        r"""
        Initialize or update the propagator state.

        Parameters
        ----------
        dim : string
            Dimensionality of the array.  Options are:

            - ``'rt'``: The laser pulse is represented on a 2D grid:
                        radial (r) transversely, and temporal (t) longitudinally.

        omega0 : float (in rad s⁻¹)
            The central angular frequency :math:`\omega_0 = 2\pi c/\lambda_0`.
        """
        self.dim = dim
        self.omega0 = omega0

        assert dim in ["rt"], "Invalid dimension. Only 'rt' is currently supported."

    def _zoomHankelTransform(self, r, f, k_r):
        r"""
        Zeroth-order Hankel transform evaluated at arbitrary output spatial freqs.

        Cylindrical analogue of the zoom (Chirp-Z) FFT computes the discrete approximation:

        .. math::

            \mathcal{H}_0[f](k_r) =
            \int_0^\infty f(r')\,J_0(k_r\,r')\,r'\,dr'
            \;\approx\; \sum_j f(r_j)\,J_0(k_r\,r_j)\,r_j\,\Delta r

        by constructing the :math:`J_0` kernel matrix and performing a
        matrix–vector product.

        Setting ``k_r = k * r_out / z`` evaluates the transform at exactly
        the spatial frequencies corresponding to the output grid positions,
        mirroring how the Chirp-Z transform evaluates the DFT at a freely
        chosen set of frequencies rather than the standard FFT grid.

        Parameters
        ----------
        r : array_like, shape (N,)
            Uniformly-spaced radial coordinates of the input field
            (must start at or very near zero).

        f : array_like, shape (N,)
            Complex field values at the radial positions ``r``.

        k_r : array_like, shape (M,)
            Radial spatial frequencies (rad m⁻¹) at which to evaluate the
            transform.  Typically ``k_r = k * r_out / z``.

        Returns
        -------
        H : array, shape (M,)
            Complex Hankel transform sampled at each frequency in ``k_r``.
        """
        # Always compute the J0 matrix on the CPU with scipy.special.
        # If xp is CuPy, np.asarray() pulls the arrays to CPU memory;
        # the result is transferred back at the end via xp.asarray().
        r_np = np.asarray(r)
        k_r_np = np.asarray(k_r)
        f_np = np.asarray(f)

        dr = r_np[1] - r_np[0]

        # J0 kernel matrix: shape (M, N)
        # Entry (i, j) = J0(k_r[i] * r[j])
        J_mat = scipy_j0(k_r_np[:, np.newaxis] * r_np[np.newaxis, :])

        # H[i] = sum_j  J_mat(k_r[i]·r[j]) · f[j] · r[j] · dr
        # The Hankel transformation becomes matrix-vector multiplication
        H_np = xp.matmul(J_mat, (f_np * r_np * dr))

        # Move result back to the active backend (no-op when xp is NumPy)
        return xp.asarray(H_np)

    def propagate(self, grid_in, dim=None, omega0=None, distance=None, grid_out=None):
        r"""
        Propagate the laser field in the z direction using zoomed Hankel transform method.

        Parameters
        ----------
        grid_in : Grid
            Grid object containing the laser field to propagate.

        dim : string (optional)
            Dimensionality of the array.  If not provided, uses the propagator's
            stored dimension.

        omega0 : float (in rad s⁻¹) (optional)
            Central angular frequency.  If not provided, uses the propagator's
            stored value.

        distance : float (in m)
            Distance by which the laser is propagated.

        grid_out : Grid (optional)
            Output grid.  May have a different radial extent and/or resolution
            from ``grid_in``.  If ``None``, a deep copy of ``grid_in`` is used.

        Returns
        -------
        Grid
            Grid object containing the propagated laser field.
        """
        self.update(dim, omega0)

        initial_position = grid_in.position

        # Obtain the spectral representation of the field.
        # field_in has shape (Nr, Nomega) for the rt geometry.
        field_in, omega = grid_in.get_spectral_field()
        if grid_out is None:
            grid_out = copy.deepcopy(grid_in)
            grid_out.set_spectral_field(xp.zeros_like(field_in))
        field_out = grid_out.spectral_field

        # Shift from relative to absolute angular frequencies.
        omega += omega0
        indxs = xp.argsort(omega)

        # Radial axes of the input and output grids.
        r = grid_in.axes[0]  # shape (Nr,)
        rF = grid_out.axes[0]  # shape (NrF,)

        assert xp.all(r >= -1e-15 * xp.abs(r[-1])), (
            "Input grid r-axis must be non-negative."
        )
        assert xp.all(rF >= -1e-15 * xp.abs(rF[-1])), (
            "Output grid r-axis must be non-negative."
        )

        # Propagate each frequency component independently.
        for indx in indxs:
            om = omega[indx]
            k = om / c  # wave-number  [/m]

            # --- Quadratic phase prefactor H = exp(i k r'^2 / 2z) ---
            prefactor = xp.exp(1j * k / 2 / distance * r**2)

            # --- Spatial frequencies k_r = k · r_out / z ---
            # These are the Hankel-transform frequencies that map onto the
            # output radial grid, playing the role of the zoom in Chirp-Z.
            k_r = k * rF / distance

            # --- Zoomed Hankel transform H0[(E0 · H)](k_r) ---
            H = self._zoomHankelTransform(
                r,
                xp.squeeze(field_in[:, :, indx]) * prefactor,
                k_r,
            )

            # --- Post-factor G = (−ik/z) exp(ikz) exp(ik r^2/2z) ---
            # The factor −ik/z = 2 pi/(i lambda z) comes from integrating out the
            # azimuthal angle relative to the 2D Cartesian prefactor 1/(i lambda z).
            # The exp(ikz) carrier is later cancelled by the phase correction
            # below, consistent with lasy's co-moving frame convention.
            postFactor = (
                -1j
                * k
                / distance
                * xp.exp(1j * k * distance)
                * xp.exp(1j * k / 2 / distance * rF**2)
            )

            field_out[:, :, indx] = H * postFactor

        # Cancel the carrier-wave propagation phase exp(i omega z/c) introduced by
        # postFactor above, restoring the co-moving-frame representation.
        field_out *= xp.exp(-1j * omega[xp.newaxis, xp.newaxis, :] * distance / c)

        # Set the result as the spectral field and advance the longitudinal position.
        grid_out.set_spectral_field(field_out)
        grid_out.position = initial_position + distance

        return grid_out
