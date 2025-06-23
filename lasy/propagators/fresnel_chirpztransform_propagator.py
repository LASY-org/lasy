import numpy as np
from scipy.constants import c
from scipy.signal import zoom_fft

from .propagator import Propagator


class FresnelChirpZPropagator(Propagator):
    r"""Class that represents a Fresnel propagator based upon the Chirp-Z Transform.

    The propagated field is calculated via the following method:

    Given a scalar field :math:`E_0(x',y',0,\omega)`, one write the propagated field
    at a distance :math:`z`, under the Fresnel approximation, as:

    .. math::

        E (x,y,z,\omega) =
        \frac{ \omega \exp{(\frac{i \omega z}{c}) \exp(i\omega\frac{x^2+y^2}{2 c z})}}{i 2 \pi c z} \int \int E_0(x',y',0,\omega) \times \exp{\left [\frac{i\omega}{2 c z}(x'^2 + y'^2) \right ]}\times \exp{\left[ \frac{i \omega}{c z} (xx' +yy')\right]} dx' dy'

    which can be rewritten as a 2D Fourier transform :math:`\mathcal{F}`:

    .. math::

        E (x,y,z,\omega) = G \times \mathcal{F}(E_0)

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
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

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

        assert dim in ["rt", "xyt"], "Invalid dimension. Choose 'rt' or 'xyt'."

    def _zoomFourierTransform2D(self, x, y, f, k_x, k_y):
        # Here we scale by dt, the discrete spacing in time
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # Calculate Frequency Axws. The 2pi is because we're returning the axis like k_x rather than 1/x
        X = x[-1] - x[0]
        sampleXFreq = len(x) / X

        # Calculate Desired Frequency Window
        freq_x = k_x / 2 / np.pi
        freq_y = k_y / 2 / np.pi

        Y = y[-1] - y[0]
        sampleYFreq = len(y) / Y

        # Do the ZoomFFT
        F = (
            zoom_fft(
                zoom_fft(
                    f,
                    [np.min(freq_x), np.max(freq_x)],
                    m=len(freq_x),
                    fs=sampleXFreq,
                    endpoint=True,
                    axis=1,
                ),
                [np.min(freq_y), np.max(freq_y)],
                m=len(freq_y),
                fs=sampleYFreq,
                endpoint=True,
                axis=0,
            )
            * dx
            * dy
        )

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

        # Get the spectral field from the grid_out object
        field_in, omega = grid_in.get_spectral_field()
        field_out = grid_out.spectral_field
        omega += omega0

        x = grid_in.axes[1]
        y = grid_in.axes[0]

        xF = grid_out.axes[1]
        yF = grid_out.axes[0]

        X, Y = np.meshgrid(x, x)
        XF, YF = np.meshgrid(xF, yF)

        for i, om in enumerate(omega):
            wavelength = 2 * np.pi * c / om
            k = om / c

            prefactor = np.exp(1j * k / 2 / distance * (X**2 + Y**2))
            k_x = 2 * np.pi * xF / wavelength / distance
            k_y = 2 * np.pi * yF / wavelength / distance

            F = self._zoomFourierTransform2D(
                x, y, field_in[:, :, i] * prefactor, k_x, k_y
            )

            postFactor = (
                np.exp(1j * k * distance)
                * np.exp(1j * k / distance * (XF**2 + YF**2))
                / (1j * wavelength * distance)
            )
            field_out[:, :, i] = F * postFactor

        grid_out.set_spectral_field(field_out)

        grid_out.position += distance

        return grid_out
