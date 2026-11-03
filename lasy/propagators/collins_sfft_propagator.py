import warnings
from copy import deepcopy

from scipy.constants import c, epsilon_0

from lasy.backend import xp
from lasy.utils.laser_utils import get_w0

from .propagator import Propagator


def _q(z, z_0, z_R):  # Defines the q-parameter of a Gaussian beam
    return z - z_0 - 1j * z_R


class CollinsSFFTPropagator(Propagator):
    r"""
    Class that represents a single FFT propagator using the Collins method.

    The propagated field is calculated using the following method:

    .. math::

        E_\mathrm{propagated} (x,y,\omega) =
        \frac{1}{i\lambda B} e^{ik(z-z_0)}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}
        E_{i} (x,y,\omega) e^{ikS}dx_0dy_0,

    where :math:`E_{i} (x,y,\omega)` is the complex field envelope of the input field
    and :math:`S` is the propagator term

    .. math::

        S = \bigg\{\frac{1}{2B}\Big[A(x_0^2+y_0^2)+D(x^2+y^2)-2(xx_0+yy_0)\Big]\bigg\},

    defined in terms of the elements of the ``'ABCD'`` optical ray matrix.

    The integral is evaluated with a single centred 2D FFT. The output
    coordinates follow the exact Collins mapping :math:`x = \lambda_0 B f_x`,
    where :math:`f_x` is the FFT spatial frequency.
    Note that this is currently monochromatic.

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

    abcd : 2d array
        The 2D ray matrix of the optical system through which the beam propagates.
        This is defined in the ``'ABCD'`` class as:

        .. math::

            O =
            \begin{pmatrix}
            A & B \\
            C & D
            \end{pmatrix}.

    """

    def __init__(self, dim, omega0):
        super().__init__()
        self.update(dim=dim, omega0=omega0)

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
        dim = dim if dim is not None else self.dim
        assert dim in ["rt", "xyt"]

        self.dim = dim
        self.omega0 = omega0 if omega0 is not None else self.omega0

    def _get_abcd(self):
        """Return the elements A, B, C, D of the ray matrix and check its determinant."""
        M = self.abcd.abcd
        A, B, C, D = M[0][0], M[0][1], M[1][0], M[1][1]
        if not xp.isclose(A * D - B * C, 1.0):
            warnings.warn("Ray matrix does not conserve energy (det(ABCD) != 1).")
        if B == 0:
            raise ValueError(
                "B = 0 (imaging condition): the Collins single-FFT method is undefined."
            )
        return A, B, C, D

    def add_output_grid(self, dim, grid_in):
        """
        Calculate the output grid automatically.

        The output axes follow the exact Collins mapping x = lambda0 * B * f,
        centred on the optical axis.

        Parameters
        ----------
        dim : string
            Dimensionality of the array. Options are:
            - ``'xyt'``: Laser pulse represented on a 3D Cartesian grid.
            - ``'rt'`` : Laser pulse represented on a 2D cylindrical grid.

        grid_in : Grid
            Grid object at the input plane.

        Returns
        -------
        grid_out : Grid
            Grid object for the output plane.
        """
        grid_out = deepcopy(grid_in)  # Make a copy of the input grid

        if self.dim == "rt":
            print(
                "'rt' geometry not yet supported by CollinsSFFTPropagator, "
                "skipping grid calculation."
            )
            return grid_out

        A, B, C, D = self._get_abcd()

        x = grid_in.axes[0]
        y = grid_in.axes[1]
        dx = xp.abs(x[1] - x[0])  # Actual grid spacing, L / (N - 1)
        dy = xp.abs(y[1] - y[0])

        lambda0 = 2.0 * xp.pi * c / self.omega0
        k0 = self.omega0 / c

        # Sanity check that the system is a focusing geometry
        w0 = get_w0(grid_in, self.dim)  # Input spot size
        z_R = xp.pi * w0**2 / lambda0  # Input Rayleigh range
        q1 = _q(0, 0, z_R)
        z_R2 = -xp.imag((A * q1 + B) / (C * q1 + D))  # Output Rayleigh range
        f0 = xp.sqrt(k0 * w0**2 / 2.0 * z_R2)  # Effective focal length
        assert f0 < 100, (
            "CollinsSFFTPropagator is for focusing geometries, please specify a lens."
        )
        x_out = xp.fft.fftshift(xp.fft.fftfreq(len(x), dx)) * lambda0 * xp.abs(B)
        y_out = xp.fft.fftshift(xp.fft.fftfreq(len(y), dy)) * lambda0 * xp.abs(B)

        grid_out.lo[0], grid_out.hi[0] = x_out[0], x_out[-1]
        grid_out.lo[1], grid_out.hi[1] = y_out[0], y_out[-1]
        grid_out.axes[0] = x_out
        grid_out.axes[1] = y_out
        return grid_out

    def propagate(self, grid_in, abcd, dim=None, omega0=None, grid_out=None):
        r"""
        Calculate the output field from the input field and the optical ray matrix of the system.

        Parameters
        ----------
        grid_in : Grid
            Grid object containing the laser to propagate.

        abcd : 2d array
            The 2D ray matrix of the optical system through which the beam propagates.

        dim : string (optional)
            Dimensionality of the array. Options are:
            - ``'xyt'``: Laser pulse represented on a 3D Cartesian grid.
            - ``'rt'`` : Laser pulse represented on a 2D cylindrical grid.

        omega0 : float (optional)
            The main frequency :math:`\omega_0` (in rad.s^-1).

        grid_out : Grid object (optional)
            Grid object on which the propagated laser pulse is defined.
            If not given, it is calculated from the ABCD matrix. Note that the
            field is always computed on the FFT-determined grid; a user grid
            must match that sampling.

        Returns
        -------
        Grid object with laser data after propagation.
        """
        self.update(omega0=omega0, dim=dim)
        self.abcd = abcd
        self.grid_out = grid_out

        if grid_out is None:  # Determine output grid from the ABCD matrix
            grid_out = self.add_output_grid(dim, grid_in)

        if self.dim == "rt":
            field = self._propagate_mrt(grid_in, grid_out)
        else:  # self.dim == "xyt"
            field = self._propagate_xyt(grid_in, grid_out)

        grid_in.set_spectral_field(field)

    def _propagate_xyt(self, grid_in, grid_out):
        # Get the spectral field and axes from the input grid
        spectral_field, spectral_axes = grid_in.get_spectral_field()

        x0, y0 = grid_in.axes[0], grid_in.axes[1]  # Input axes
        x, y = grid_out.axes[0], grid_out.axes[1]  # Output axes

        X0, Y0, OM = xp.meshgrid(y0, x0, spectral_axes + self.omega0)
        X, Y, OM = xp.meshgrid(y, x, spectral_axes + self.omega0)
        R0 = xp.sqrt(X0**2 + Y0**2)
        R = xp.sqrt(X**2 + Y**2)

        A, B, C, D = self._get_abcd()

        # Input quadratic phase
        field = spectral_field * xp.exp(1j * OM / (2 * c) * (A / B) * R0**2)

        # Centred 2D FFT over the transverse axes:
        #   ifftshift -> move the input centre to index 0
        #   fftshift  -> move zero frequency (optical axis) to the array
        #                centre, matching the fftshift-ed output axes
        field = xp.fft.ifftshift(field, axes=(0, 1))
        field = xp.fft.fft2(field, axes=(0, 1))
        field = xp.fft.fftshift(field, axes=(0, 1))

        # Negative B: x = lambda*B*f, so the image is inverted. Map f -> -f
        # exactly (flip + 1-pixel roll, since even-length FFT grids are
        # asymmetric).
        if B < 0:
            field = xp.roll(xp.flip(field, axis=(0, 1)), 1, axis=(0, 1))

        # Output quadratic phase and phase of the Collins prefactor 1/(i*lambda*B)
        prefac = OM / (2j * xp.pi * c * B)
        field = (
            field * xp.exp(1j * OM / (2 * c) * (D / B) * R**2) * prefac / xp.abs(prefac)
        )

        # Energy normalisation
        dA_in = xp.abs(x0[1] - x0[0]) * xp.abs(y0[1] - y0[0])
        dA_out = xp.abs(x[1] - x[0]) * xp.abs(y[1] - y[0])
        field *= xp.sqrt(
            xp.sum(c * epsilon_0 * xp.abs(spectral_field) ** 2 * dA_in)
            / xp.sum(c * epsilon_0 * xp.abs(field) ** 2 * dA_out)
        )

        # Update the grid to the output plane
        grid_in.lo[0], grid_in.hi[0] = x[0], x[-1]
        grid_in.lo[1], grid_in.hi[1] = y[0], y[-1]
        grid_in.axes[0] = x
        grid_in.axes[1] = y
        return field

    def _propagate_mrt(self, grid_in, grid_out):
        print(
            "'rt' geometry not yet supported by CollinsSFFTPropagator, skipping propagation."
        )
        return grid_in.field
