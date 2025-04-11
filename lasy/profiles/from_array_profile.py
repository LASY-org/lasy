import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .profile import Profile


class FromArrayProfile(Profile):
    r"""
    Profile defined from numpy array directly.

    The numpy array contains the envelope of the electric field of the laser pulse, defined as :math:`\mathcal{E}` in:

    .. math::
        \begin{aligned}
        E_x(x,y,t) = \operatorname{Re} \left( \mathcal{E}(x,y,t) e^{-i \omega_0t}p_x \right)\\
        E_y(x,y,t) = \operatorname{Re} \left( \mathcal{E}(x,y,t) e^{-i \omega_0t}p_y \right)\end{aligned}

    where :math:`\operatorname{Re}` stands for real part, :math:`E_x` (resp. :math:`E_y`) is the laser electric field in the :math:`x` (resp. :math:`y`) direction.

    Parameters
    ----------
    wavelength : float (in meter)
        The main laser wavelength :math:`\lambda_0` of the laser, which
        defines :math:`\omega_0` in the above formula, according to
        :math:`\omega_0 = 2\pi c/\lambda_0`.

    pol : list of 2 complex numbers (dimensionless)
        Polarization vector. It corresponds to :math:`p_u` in the above
        formula ; :math:`p_x` is the first element of the list and
        :math:`p_y` is the second element of the list. Using complex
        numbers enables elliptical polarizations.

    array : 3darray of complex numbers
        Contains the values of the envelope, defined as :math:`\mathcal{E}` in the above formula.

    dim : string
        "xyt" or "rt"

    axes : Python dictionary containing the axes vectors.
        Keys are 'x', 'y', 't'.
        Values are the 1D arrays of each axis.
        array.shape = (axes['x'].size, axes['y'].size, axes['t'].size) in 3D,
        and similar in cylindrical geometry.

    axes_order : List of strings
        Gives the name and ordering of the axes in the array.
        Currently, only implemented for 3D, and supported values are
        ['x', 'y', 't'] and ['t', 'y', 'x'].
    """

    def __init__(self, wavelength, pol, array, dim, axes, axes_order=["x", "y", "t"]):
        super().__init__(wavelength, pol)

        assert dim in ["xyt", "rt"]
        assert array.ndim == 3, "array must be 3D, [x,y,t] or [modes,r,t]."
        self.axes = axes
        self.dim = dim
        self.array = array

        if dim == "xyt":
            assert axes_order == ["x", "y", "t"]

            self.combined_field_interp = RegularGridInterpolator(
                (axes["x"], axes["y"], axes["t"]),
                np.abs(self.array) + 1.0j * np.unwrap(np.angle(self.array), axis=-1),
                bounds_error=False,
                fill_value=0.0,
            )
        else:  # dim = "rt"
            assert axes_order == ["r", "t"]

            # If the first point of radial axis is not 0, we "mirror" it,
            # to make correct interpolation within the first cell
            if axes["r"][0] != 0.0:
                # add mirrored point to the axis
                r = np.concatenate(([-axes["r"][0]], axes["r"]))
                # takes first element of the array in the radial dimension
                subarray = self.array[:, 0, :]
                # add it at the beginning to be the value at the mirrored point
                self.array = np.concatenate(
                    (subarray[:, np.newaxis, :], self.array), axis=1
                )
            else:
                r = axes["r"]

            self.field_interp_modes = []
            # Loop over the 2*m-1 elements of the array and create a separate
            # interpolator object for each of them.
            # Note that the field_interp_modes is not directly the complex
            # envelope because interpolating separately real and imag is not
            # accurate enough. Instead, the real part of field_interp_modes
            # represents the mode's modulus and its imag the mode's phase.
            for imode in range(self.array.shape[0]):
                self.field_interp_modes.append(
                    RegularGridInterpolator(
                        (r, axes["t"]),
                        np.abs(self.array[imode, :, :])
                        + 1.0j * np.unwrap(np.angle(self.array[imode, :, :]), axis=-1),
                        bounds_error=False,
                        fill_value=0.0,
                    )
                )

    def evaluate(self, x, y, t):
        """Return the envelope field of the scaled profile."""
        if self.dim == "xyt":
            combined_field = self.combined_field_interp((x, y, t))
        else:
            r = np.sqrt(x**2 + y**2)
            theta = np.angle(x + 1j * y)
            combined_field = np.zeros_like(x, dtype="complex128")
            nmodes = (len(self.field_interp_modes) + 1) // 2
            for imode in range(-nmodes + 1, nmodes):
                combined_field += self.field_interp_modes[imode]((r, t)) * np.exp(
                    -1j * imode * theta
                )

        return np.abs(np.real(combined_field)) * np.exp(1.0j * np.imag(combined_field))

    def evaluate_mrt(self, mode, r, t):
        """Return the envelope field of the scaled profile."""
        assert self.dim == "rt"
        combined_field = self.field_interp_modes[mode]((r, t))
        return np.abs(np.real(combined_field)) * np.exp(1.0j * np.imag(combined_field))
