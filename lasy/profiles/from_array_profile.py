from scipy.interpolate import RegularGridInterpolator
from .profile import Profile


class FromArrayProfile(Profile):
    r"""
    Profile defined from numpy array directly.

    Parameters
    ----------
    wavelength : float (in meter)
        The main laser wavelength :math:`\\lambda_0` of the laser, which
        defines :math:`\\omega_0` in the above formula, according to
        :math:`\\omega_0 = 2\\pi c/\\lambda_0`.

    pol : list of 2 complex numbers (dimensionless)
        Polarization vector. It corresponds to :math:`p_u` in the above
        formula ; :math:`p_x` is the first element of the list and
        :math:`p_y` is the second element of the list. Using complex
        numbers enables elliptical polarizations.

    array : Array of the electric field of the laser pulse.

    axes : Python dictionary containing the axes vectors.
        Keys are 'x', 'y', 't' if dim=xyt and 'r', 't' if dim=rt, respectively.
        Values are the 1D arrays of each axis.
        array.shape = (axes['x'].size, axes['y'].size, axes['t'].size) in 3D,
        and similar in cylindrical geometry.

    dim : Dimension of the data, 'xyt' or 'rt'
    """

    def __init__(self, wavelength, pol, array, axes, dim):
        super().__init__(wavelength, pol)

        assert dim == "xyt" or dim == "rt", "dim must be 'xyt' or 'rt'"

        assert dim == "xyt", "Only dim='xyt' currently implemented"

        self.array = array
        self.axes = axes
        self.dim = dim

        if dim == "xyt":
            self.field_interp = RegularGridInterpolator(
                (axes["x"], axes["y"], axes["t"]),
                array,
                bounds_error=False,
                fill_value=0.0,
            )

    def evaluate(self, x, y, t):
        """Return the envelope field of the scaled profile."""
        envelope = self.field_interp((x, y, t))
        return envelope
