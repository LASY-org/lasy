from scipy.interpolate import RegularGridInterpolator
import numpy as np
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
        Keys are 'x', 'y', 't'.
        Values are the 1D arrays of each axis.
        array.shape = (axes['x'].size, axes['y'].size, axes['t'].size) in 3D,
        and similar in cylindrical geometry.

    axes_order : List of strings, giving the name and ordering of the axes in the array.
        Currently, only implemented for 3D, and supported values are
        ['x', 'y', 't'] and ['t', 'y', 'x'].
    """

    def __init__(self, wavelength, pol, array, dim, axes, axes_order=["x", "y", "t"]):
        super().__init__(wavelength, pol)

        assert dim in ["xyt", "rt"]
        self.axes = axes
        self.dim = dim

        if dim == "xyt":
            assert axes_order in [["x", "y", "t"], ["t", "y", "x"]]

            if axes_order == ["t", "y", "x"]:
                self.array = np.swapaxes(array, 0, 2)
            else:
                self.array = array

            self.field_interp = RegularGridInterpolator(
                (axes["x"], axes["y"], axes["t"]),
                array,
                bounds_error=False,
                fill_value=0.0,
            )

        else:  # dim = "rt"
            assert axes_order in [["r", "t"], ["t", "r"]]

            if axes_order == ["t", "r"]:
                self.array = np.swapaxes(array, 0, 2)
            else:
                self.array = array

            # The first point is at dr/2.
            # To make correct interpolation within the first cell, mirror
            # field and axes along r.
            r = np.concatenate((-axes["r"][::-1], axes["r"]))
            array = np.concatenate((array[::-1], array))
            self.field_interp = RegularGridInterpolator(
                (r, axes["t"]),
                array,
                bounds_error=False,
                fill_value=0.0,
            )

    def evaluate(self, x, y, t):
        """Return the envelope field of the scaled profile."""
        if self.dim == "xyt":
            envelope = self.field_interp((x, y, t))
        else:
            envelope = self.field_interp((np.sqrt(x**2 + y**2), t))
        return envelope
