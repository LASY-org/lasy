import copy

import numpy as np
from scipy.constants import c
from scipy.signal import zoom_fft

from .propagator import Propagator


class FresnelChirpZPropagator(Propagator):
    r"""Class that represents a Fresnel propagator based upon the Chirp-Z Transform.

    The propagated field is calculated via the following method:

    Given a scalar field :math:`E_0(x',y',0,\omega)`, one writes the propagated field
    at a distance :math:`z`, under the Fresnel approximation, as:

    .. math::

        E (x,y,z,\omega) =
        \frac{ \omega \exp{(\frac{i \omega z}{c}) \exp(i\omega\frac{x^2+y^2}{2 c z})}}{i 2 \pi c z} \int \int E_0(x',y',0,\omega) \times \exp{\left [\frac{i\omega}{2 c z}(x'^2 + y'^2) \right ]}\times \exp{\left[ \frac{i \omega}{c z} (xx' +yy')\right]} dx' dy'

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

    Parameters
    ----------
    omega0 : float (in rad/s)
        The center frequency of the laser field.

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.


    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from lasy.laser import Laser
    >>> from lasy.profiles.gaussian_profile import GaussianProfile
    >>> from lasy.optical_elements import ParabolicMirror
    >>> from lasy.propagators import FresnelChirpZPropagator
    >>> from lasy.utils.grid import Grid
    >>> import numpy as np
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
    >>> w0theory = 0.8e-6 * focal_length / (np.pi * 5e-3)
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

        assert dim in ["xyt"], "Invalid dimension. Only 'xyt' is currently supported."

    def _zoomFourierTransform2D(self, x, y, f, k_x, k_y):
        # Get initial grid spacing in each axis
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # Calculate the sample frequency in each axis
        x_range = x[-1] - x[0]
        y_range = y[-1] - y[0]
        sample_frequency_x = (len(x) - 1) / x_range
        sample_frequency_y = (len(y) - 1) / y_range

        # Convert desired frequency from rad/s to Hz
        freq_x = k_x / 2 / np.pi
        freq_y = k_y / 2 / np.pi

        FreqX, FreqY = np.meshgrid(
            freq_x,
            freq_y,
        )

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

        # Apply the phase factor to shift the transform. Similar to a Fourier Transform shift.
        F *= np.exp(1j * FreqX * np.pi * x_range) * np.exp(1j * FreqY * np.pi * y_range)

        return F

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

        initial_position = grid_in.position

        # Get the spectral field from the grid objects
        field_in, omega = grid_in.get_spectral_field()
        if grid_out is None:
            # Create a new grid for the output if not provided
            grid_out = copy.deepcopy(grid_in)
            grid_out.set_spectral_field(np.zeros_like(field_in))
        field_out = grid_out.spectral_field
        omega += omega0
        indxs = np.argsort(omega)

        # Extract the initial and final axes from the grids
        x = grid_in.axes[0]
        y = grid_in.axes[1]
        xF = grid_out.axes[0]
        yF = grid_out.axes[1]

        assert np.isclose(np.mean(x), 0, atol=1e-8 * np.abs((x[-1] - x[0]))), (
            "Input grid x-axis is not centered around zero."
        )
        assert np.isclose(np.mean(y), 0, atol=1e-8 * np.abs((y[-1] - y[0]))), (
            "Input grid y-axis is not centered around zero."
        )
        assert np.isclose(np.mean(xF), 0, atol=1e-8 * np.abs((xF[-1] - xF[0]))), (
            "Output grid x-axis is not centered around zero."
        )
        assert np.isclose(np.mean(yF), 0, atol=1e-8 * np.abs((yF[-1] - yF[0]))), (
            "Output grid y-axis is not centered around zero."
        )

        X, Y = np.meshgrid(x, y, indexing="ij")
        XF, YF = np.meshgrid(xF, yF, indexing="ij")

        for indx in indxs:
            om = omega[indx]
            wavelength = 2 * np.pi * c / om
            k = om / c

            prefactor = np.exp(1j * k / 2 / distance * (X**2 + Y**2))

            # Calculate the required fourier frequencies from output grid
            k_x = 2 * np.pi * xF / wavelength / distance
            k_y = 2 * np.pi * yF / wavelength / distance

            # Perform the 2D Zoom FFT
            F = self._zoomFourierTransform2D(
                x, y, np.squeeze(field_in[:, :, indx]) * prefactor, k_x, k_y
            )

            postFactor = (
                np.exp(1j * k * distance)
                * np.exp(1j * k / 2 / distance * (XF**2 + YF**2))
                / (1j * wavelength * distance)
            )

            # Add output field to array
            field_out[:, :, indx] = F * postFactor

        # Shift the pulse back to the center of the time axis
        field_out *= np.exp(-1j * omega[np.newaxis, np.newaxis, :] * distance / c)

        # Update output grid parameters
        grid_out.set_spectral_field(field_out)
        grid_out.position = initial_position + distance

        return grid_out
