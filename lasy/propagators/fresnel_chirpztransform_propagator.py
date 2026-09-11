import copy

from scipy.constants import c

from lasy.backend import xp, zoom_fft, j0

from .propagator import Propagator


class FresnelChirpZPropagator(Propagator):
    r"""Class that represents a Fresnel propagator based upon the Chirp-Z Transform.

    The propagated field is calculated via the following method in Cartesian coordinates:

    Given a scalar field :math:`E_0(x',y',0,\omega)`, one writes the propagated field
    at a distance :math:`z`, under the Fresnel approximation, as:

    .. math::

        E (x,y,z,\omega) =
        \frac{ \omega \exp{(\frac{i \omega z}{c}) \exp(i\omega\frac{x^2+y^2}{2 c z})}}{i 2 \pi c z}
        \int \int E_0(x',y',0,\omega) \times \exp{\left [\frac{i\omega}{2 c z}(x'^2 + y'^2) \right ]}
        \times \exp{\left[ \frac{i \omega}{c z} (xx' +yy')\right]} dx' dy'

    which can be rewritten as a 2D Fourier transform :math:`\mathcal{F}`:

    .. math::

        E (x,y,z,\omega) = G \times \mathcal{F}(E_0 \times H)

    where :math:`G` is given by:

    .. math::

        G = \frac{ \omega \exp{(\frac{i \omega z}{c}) \exp(i\omega\frac{x^2+y^2}{2 c z})}}{i 2 \pi c z}

    and where :math:`H` is given by:

    .. math::

        H = \exp{\left [\frac{i\omega}{2 c z}(x'^2 + y'^2) \right ]}


    Normally, the Fourier transform is computed using the Fast Fourier Transform (FFT) algorithm.
    However, in this case, the Chirp-Z Transform (or Zoom FFT) is used to compute the Fourier transform.
    This allows for more flexibility in choosing both the initial and final sampling of the Fourier transform.

    The algorithm is based upon the work by Hu et al., https://www.nature.com/articles/s41377-020-00362-z
    and the implementation of the Chirp-Z Transform in SciPy, specifically `scipy.signal.zoom_fft`.


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

    Parameters
    ----------
    omega0 : float (in rad/s)
        The center frequency of the laser field.

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    cylindrical (r) transversely, and temporal (t) longitudinally.

    Examples
    --------
    >>> from lasy.laser import Laser
    >>> from lasy.profiles.gaussian_profile import GaussianProfile
    >>> from lasy.optical_elements import ParabolicMirror
    >>> from lasy.propagators import FresnelChirpZPropagator
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
    >>> # Create laser with given profile in `xyt` geometry.
    >>> laser = Laser(
    ...     dim="xyt",
    ...     lo=(-15e-3, -15e-3, -60e-15),
    ...     hi=(15e-3, 15e-3, +60e-15),
    ...     npoints=(200, 200, 500),
    ...     profile=profile,
    ... )
    >>> # Add Focusing Phase.
    >>> focal_length = 1  # m
    >>> laser.apply_optics(ParabolicMirror(focal_length))
    >>> # Add Fresnel Chirp-Z propagator.
    >>> laser.add_propagator(FresnelChirpZPropagator())
    >>> # Create a new resampled grid for propagation.
    >>> xLimNew = 150e-6  # m
    >>> newGrid = Grid(
    ...     laser.dim,
    ...     (-xLimNew, -xLimNew, laser.grid.lo[2]),
    ...     (xLimNew, xLimNew, laser.grid.hi[2]),
    ...     (100, 100, laser.grid.npoints[2]),
    ... )
    >>> # Propagate the laser pulse to the focal plane and visualise.
    >>> laser.propagate(focal_length, grid_out=newGrid)
    >>> laser.show(envelope_type="intensity")
    >>> w0theory = 0.8e-6 * focal_length / (xp.pi * 5e-3)
    >>> print("w0 theoretical: %.2e m" % (w0theory))
    """

    def update(self, dim, omega0):
        r"""
        Initialize or update the propagator if needed.

        Parameters
        ----------
        dim : string
            Dimensionality of the array. Options are:
            - ``'xyt'``: Laser pulse represented on a 3D Cartesian grid.
            - ``'rt'`` : Laser pulse represented on a 2D cylindrical grid.

        omega0 : float (in rad.s^-1)
            The main frequency :math:`\omega_0`, which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.
        """
        self.dim = dim
        self.omega0 = omega0

    def _zoomFourierTransform2D(self, x, y, f, k_x, k_y):
        # Get initial grid spacing in each axis
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # Calculate the sample frequency in each axis
        x_range = x[-1] - x[0]
        y_range = y[-1] - y[0]
        sample_frequency_x = (len(x) - 1) / x_range
        sample_frequency_y = (len(y) - 1) / y_range

        # Convert desired frequency from rad/m to cycles/m
        freq_x = k_x / (2 * xp.pi)
        freq_y = k_y / (2 * xp.pi)

        FreqX, FreqY = xp.meshgrid(freq_x, freq_y)

        # Perform the 2D Zoom FFT as a set of 2x 1D Zoom FFTs
        F = (
            zoom_fft(
                zoom_fft(
                    f,
                    [freq_x[0], freq_x[-1]],
                    m=len(freq_x),
                    fs=sample_frequency_x,
                    endpoint=True,
                    axis=1,
                )
                * dx,
                [freq_y[0], freq_y[-1]],
                m=len(freq_y),
                fs=sample_frequency_y,
                endpoint=True,
                axis=0,
            )
            * dy
        )

        # Apply phase shift to account for non-zero grid origin (analogous to FFT shift)
        F *= xp.exp(1j * FreqX * xp.pi * x_range) * xp.exp(1j * FreqY * xp.pi * y_range)

        return F

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
        dr = r[1] - r[0]

        # J0 kernel matrix: shape (M, N), entry (i,j) = J0(k_r[i] * r[j])
        J_mat = j0(k_r[:, xp.newaxis] * r[xp.newaxis, :])

        # H[i] = sum_j J_mat[i,j] * f[j] * r[j] * dr  (matrix-vector product)
        return J_mat @ (f * r * dr)

    def propagate(self, grid_in, dim=None, omega0=None, distance=None, grid_out=None):
        r"""
        Propagates the laser field in z direction by a given distance using the Chirp-Z Transform method.

        Parameters
        ----------
        grid_in : Grid
            Grid object containing the laser to propagate.

        dim : string (optional)
            Dimensionality of the array. If not provided, uses the propagator's dimension.

        omega0 : float (in rad/s) (optional)
            The center frequency of the laser field. If not provided, uses the propagator's frequency.

        distance : scalar
            Distance by which the laser is propagated.

        grid_out : Grid object (optional)
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.

        Returns
        -------
        Grid object with laser data after propagation.
        """
        self.update(dim, omega0)

        # --- Common setup ---
        initial_position = grid_in.position
        field_in, omega = grid_in.get_spectral_field()

        if grid_out is None:
            grid_out = copy.deepcopy(grid_in)
            grid_out.set_spectral_field(xp.zeros_like(field_in))
        field_out = grid_out.spectral_field

        omega = omega + omega0  # avoid mutating the grid's internal array
        indxs = xp.argsort(omega)

        # --- Geometry-specific propagation ---
        if self.dim == 'xyt':
            x, y   = grid_in.axes[0],  grid_in.axes[1]
            xF, yF = grid_out.axes[0], grid_out.axes[1]

            assert xp.isclose(xp.mean(x),  0, atol=1e-8 * xp.abs(x[-1]  - x[0])),  \
                "Input grid x-axis is not centered around zero."
            assert xp.isclose(xp.mean(y),  0, atol=1e-8 * xp.abs(y[-1]  - y[0])),  \
                "Input grid y-axis is not centered around zero."
            assert xp.isclose(xp.mean(xF), 0, atol=1e-8 * xp.abs(xF[-1] - xF[0])), \
                "Output grid x-axis is not centered around zero."
            assert xp.isclose(xp.mean(yF), 0, atol=1e-8 * xp.abs(yF[-1] - yF[0])), \
                "Output grid y-axis is not centered around zero."

            X,  Y  = xp.meshgrid(x,  y,  indexing="ij")
            XF, YF = xp.meshgrid(xF, yF, indexing="ij")

            for indx in indxs:
                om = omega[indx]
                k  = om / c
                wavelength = 2 * xp.pi / k

                prefactor = xp.exp(1j * k / (2 * distance) * (X**2 + Y**2))
                k_x = k * xF / distance
                k_y = k * yF / distance

                F = self._zoomFourierTransform2D(
                    x, y, xp.squeeze(field_in[:, :, indx]) * prefactor, k_x, k_y
                )

                postFactor = (
                    xp.exp(1j * k * distance)
                    * xp.exp(1j * k / (2 * distance) * (XF**2 + YF**2))
                    / (1j * wavelength * distance)
                )

                field_out[:, :, indx] = F * postFactor

        elif self.dim == 'rt':
            r  = grid_in.axes[0]
            rF = grid_out.axes[0]

            for indx in indxs:
                om = omega[indx]
                k  = om / c

                prefactor = xp.exp(1j * k / (2 * distance) * r**2)
                k_r = k * rF / distance

                # field_in has shape (Nr, n_azimuthal_modes, Nω); squeeze out the
                # modes axis (=1) so _zoomHankelTransform receives a 1-D vector.
                F = self._zoomHankelTransform(
                    r, xp.squeeze(field_in[:, :, indx]) * prefactor, k_r
                )

                postFactor = (
                    (-1j * k / distance)
                    * xp.exp(1j * k * distance)
                    * xp.exp(1j * k / (2 * distance) * rF**2)
                )

                field_out[:, :, indx] = F * postFactor

        # --- Common teardown ---
        # Shift pulse back to centre of time axis; broadcast omega over all spatial axes
        omega_bc = omega.reshape((1,) * (field_out.ndim - 1) + (-1,))
        field_out *= xp.exp(-1j * omega_bc * distance / c)

        grid_out.set_spectral_field(field_out)
        grid_out.position = initial_position + distance

        return grid_out
