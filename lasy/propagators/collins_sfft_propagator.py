from copy import deepcopy

import numpy as np
from numpy.fft import fftfreq, fftshift
from scipy.constants import c, epsilon_0

from lasy.utils.fft_wrapper import fft
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

    def add_output_grid(self, dim, grid_in):
        """
        Calculate the output grid automatically.

        Resolution and size are determined based on the focusing geometry calculated from the ABCD optical ray matrix.

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
        if self.dim == "rt":
            print(
                "'rt' geometry not yet supported by CollinsSFFTPropagator, skipping grid calculation."
            )
            grid_out = deepcopy(grid_in)  # Make a copy of the input grid

        else:  # self.dim == "xyt"
            grid_out = deepcopy(grid_in)  # Make a copy of the input grid

            try:  # Get the elements of the optical matrix
                A = self.abcd.abcd[0][0]
                B = self.abcd.abcd[0][1]
                C = self.abcd.abcd[1][0]
                D = self.abcd.abcd[1][1]
                det = A * D - B * C
            except det != 1:
                print("Ray matrix does not conserve energy.")

            x = grid_in.axes[0]
            L0_width = np.abs(x[-1] - x[0])
            N_points = len(x)

            lambda0 = 2.0 * np.pi * c / self.omega0
            k0 = self.omega0 / c

            w0 = get_w0(grid_in, self.dim)  # Calculate input spot size
            z_R = np.pi * w0**2 / lambda0  # Calculate input Rayleigh range

            q1 = _q(0, 0, z_R)

            # Calculate output Rayleigh range
            z_R2 = -np.imag((A * q1 + B) / (C * q1 + D))
            f0 = np.sqrt(k0 * w0**2 / 2.0 * z_R2)  # Calculate effective focal length
            assert f0 < 100, (
                "CollinsSFFTPropagator is for focusing geometries, please specify a lens."
            )

            r0_step = L0_width / N_points  # Note: D gridpoints means D-1 intervals

            x_out = fftshift(fftfreq(N_points, r0_step) * lambda0 * f0)
            y_out = fftshift(fftfreq(N_points, r0_step) * lambda0 * f0)

            grid_out.lo[0] = x_out[0]
            grid_out.lo[1] = y_out[0]
            grid_out.hi[0] = x_out[-1]
            grid_out.hi[1] = y_out[-1]
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
            By default, this is initialised to be the unitary matrix:

            .. math::

                O =
                \begin{pmatrix}
                A & B \\
                C & D
                \end{pmatrix}=
                \begin{pmatrix}
                1 & 0 \\
                0 & 1
                \end{pmatrix}.

        dim : string (optional)
            Dimensionality of the array. Options are:
            - ``'xyt'``: Laser pulse represented on a 3D Cartesian grid.
            - ``'rt'`` : Laser pulse represented on a 2D cylindrical grid.

        omega0 : float (optional)
            The main frequency :math:`\omega_0` (in rad.s^-1), which is defined by the laser
            wavelength :math:`\lambda_0`, as :math:`\omega_0 = 2\pi c/\lambda_0`.

        grid_out : Grid object (optional)
            Grid object on which the propagated laser pulse is defined.
            Can be different from laser grid before propagation.

        Returns
        -------
        Grid object with laser data after propagation.

        """
        self.update(omega0=omega0, dim=dim)
        self.abcd = abcd
        self.grid_out = grid_out

        if (
            grid_out is None
        ):  # Call routine to determine output grids from focusing geometry
            grid_out = self.add_output_grid(dim, grid_in)
        else:
            grid_out = self.grid_out  # Use user-specified grid

        if self.dim == "rt":
            field = self._propagate_mrt(grid_in, grid_out)

        else:  # self.dim == "xyt"
            field = self._propagate_xyt(grid_in, grid_out)

        grid_in.set_spectral_field(field)

    def _propagate_xyt(self, grid_in, grid_out):
        # Get the spectral field and axes from the input grid
        spectral_field, spectral_axes = grid_in.get_spectral_field()

        x0 = grid_in.axes[0]  # Input axes
        y0 = grid_in.axes[1]

        x = grid_out.axes[0]  # Output axes
        y = grid_out.axes[1]

        X0, Y0, OM = np.meshgrid(y0, x0, spectral_axes + self.omega0)
        X, Y, OM = np.meshgrid(y, x, spectral_axes + self.omega0)
        R0 = np.sqrt(X0**2 + Y0**2)
        R = np.sqrt(X**2 + Y**2)

        try:  # Get the elements of the optical matrix
            A = self.abcd.abcd[0][0]
            B = self.abcd.abcd[0][1]
            C = self.abcd.abcd[1][0]
            D = self.abcd.abcd[1][1]
            det = A * D - B * C
        except det != 1:
            print("Ray matrix does not conserve energy.")

        propagator = np.exp(1j * OM / (2 * c) * (A / B) * R0**2)

        # Take the convolution to the output plane
        field, _ = fft(
            arr_in=spectral_field * propagator,
            which="transverse",
            axes_in=(x0, y0),
            from_domain="frequency",
        )

        # Return field in spectral domain
        field = (
            field
            * np.exp(1j * OM / (2 * c) * (D / B) * R**2)
            * OM
            / (2j * np.pi * c * B)
            / np.abs(OM / (2j * np.pi * c * B))
        )
        field *= np.sqrt(
            np.sum(
                c
                * epsilon_0
                * np.abs(spectral_field) ** 2
                * np.abs(x0[1] - x0[0])
                * np.abs(y0[1] - y0[0])
            )
            / np.sum(
                c
                * epsilon_0
                * np.abs(field) ** 2
                * np.abs(x[1] - x[0])
                * np.abs(y[1] - y[0])
            )
        )

        # Update the grid
        grid_in.lo[0] = x[0]
        grid_in.lo[1] = y[0]
        grid_in.hi[0] = x[-1]
        grid_in.hi[1] = y[-1]
        grid_in.axes[0] = x
        grid_in.axes[1] = y
        return field

    def _propagate_mrt(self, grid_in, grid_out):
        print(
            "'rt' geometry not yet supported by CollinsSFFTPropagator, skipping propagation."
        )
        field = grid_in.field
        return field
