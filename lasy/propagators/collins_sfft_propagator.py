import numpy as np
from numpy.fft import fftfreq, fftshift, ifft2, ifftshift

from lasy.utils.laser_utils import get_w0

from .single_fft_propagator import SingleFFTPropagator

"DEFINE CONSTANTS"
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9
ps = 1e-12
fs = 1e-15
mJ = 1e-3
c = 2.998e8


class CollinsSFFTPropagator(SingleFFTPropagator):
    r"""
    Class that represents a single FFT propagator using the Collins method.

    The propagated field is calculated in the following method:

    .. math::

        E_\mathrm{propagated} (x,y,\omega) =
        \frac{1}{i\lambda B} e^{ik(z-z_0)}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}
        E_{i} (x,y,\omega) e^{ikS}dx_0dy_0,

    where :math:`E_{i} (x,y,\omega)` is the initial/propagated fields complex field envelope
    and :math:`S` is the propagator term

    :math:
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
    """

    
    def __init__(self, omega0, dim, abcd=np.array([[1,0],[0,1]])):
        super().__init__()
        self.update(dim=dim, omega0=omega0, abcd=abcd)

    def update(self, dim, omega0, abcd):
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
        """
        dim = dim if dim is not None else self.dim
        assert isinstance(abcd, (np.ndarray))
        assert dim in ["rt", "xyt"]

        self.dim = dim
        self.omega0 = omega0 if omega0 is not None else self.omega0
        self.abcd = abcd  # optical matrix

    def add_vacuum(self, distance):
        vacuum = np.array([[1, distance], [0, 1]])
        self.abcd = np.matmul(vacuum, self.abcd)
        return self.abcd

    def add_lens(self, focal_length):
        self.f0 = focal_length
        lens = np.array([[1, 0], [-1.0 / focal_length, 1]])
        self.abcd = np.matmul(lens, self.abcd)
        return self.abcd

    def add_output_grid(self, grid):
        """
        Function to calculate the output grids
        for a focusing / defocusing beam

        Parameters
        ----------
        grid_in : meshgrid (in meter)
            2D meshgrid for the input coordinates

        """
        # AT THE MOMENT THIS ASSUMES SYMMETRIC CARTESIAN GRIDS
        axes = grid.axes
        x = axes[0]
        y = axes[1]
        L0_width = np.abs(x[-1] - x[0])
        N_points = len(x)

        w0 = get_w0(grid, self.dim)
        f0 = self.f0
        NA = w0 / f0
        lambda0 = 2.0 * np.pi * c / self.omega0
        k0 = self.omega0 / c
        print("Numerical aperture: ", NA, "\nf/#: ", 1 / NA)

        # Spot size and Rayleigh range after lens
        z_Rf0 = 2.0 * f0**2 / (k0 * w0**2)  # Estimated Rayleigh range
        w_0f = 2.0 * f0 / (k0 * w0)  # Estimated focal spot-size
        print("Waist size [um]: ", w_0f / um, "\nRayleigh [um]: ", z_Rf0 / um)

        r0_step = L0_width / N_points  # Note: D gridpoints means D-1 intervals

        x = fftshift(fftfreq(N_points, r0_step) * lambda0 * f0)
        y = fftshift(fftfreq(N_points, r0_step) * lambda0 * f0)
        return [x, y]

    def propagate(self, grid_in, dim=None, omega0=None, distance=None, grid_out=None):
        """
        Function to calculate an output field from
        input field and optical ray matrix of the system

        Parameters
        ----------
        profile : array
            The input field profile with arbitrary intensity and phase

        grid_in : meshgrid (in meter)
            2D meshgrid for the input coordinates

        grid_out : meshgrid (in meter)
            2D meshgrid for the output coordinates

        abcd : array
            The ray matrix of the optical system

        omega_0 : float (in meter)
            The wavelength of the electric field

        """
        

        self.update(omega0=omega0, dim=dim, abcd=self.abcd)

        axes = grid_in.axes
        if grid_out == None:
            axes_out = self.add_output_grid(
                grid_in
            )  # Call routine to determine output grid
        else:
            axes_out = self.grid_out.axes  # Use user-specified grid

        # Get the spectral field and axes from the input grid
        spectral_field, spectral_axes = grid_in.get_spectral_field()

        try:
            abcd = self.abcd
            A = abcd[0][0]
            B = abcd[0][1]
            C = abcd[1][0]
            D = abcd[1][1]
        except:
            print("Missing the ray matrix for the optical system.")
        print("Determinant of optical matrix: ", A*D - B*C)

        if self.dim == "rt":
            print("Collins SFFT propagator in rt")
            profile_out = None

        elif self.dim == "xyt":
            print("Collins SFFT propagator in xyt")
            axes = grid_in.axes  # Input axes
            x0 = axes[0]
            y0 = axes[1]

            x = axes_out[0]  # Output axes
            y = axes_out[1]

            X0, Y0, OM = np.meshgrid(y0, x0, spectral_axes + self.omega0)
            X, Y, OM = np.meshgrid(y, x, spectral_axes + self.omega0)
            R0 = np.sqrt(X0**2 + Y0**2)
            R = np.sqrt(X**2 + Y**2)

            propagator = np.exp(1j * OM / (2 * c) * (A / B) * R0**2)
            profile_out = fftshift(
                ifft2(
                    ifftshift(spectral_field * propagator, axes=(0, 1)),
                    axes=(0, 1),
                ),
                axes=(0, 1),
            ) * np.sqrt(np.shape(R)[0] * np.shape(R)[1])
            profile_out = (
                profile_out
                * np.exp(1j * OM / (2 * c) * (D / B) * R**2)
                * OM
                / (2j * np.pi * c * B)
                / np.abs(OM / (2j * np.pi * c * B))
            )

        grid_in.lo[0] = x[0]
        grid_in.lo[1] = y[0]
        grid_in.hi[0] = x[-1]
        grid_in.hi[1] = y[-1]
        grid_in.axes[0] = x
        grid_in.axes[1] = y
        grid_in.set_spectral_field(profile_out)
