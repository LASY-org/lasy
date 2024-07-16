import h5py
from scipy.constants import c

from lasy.backend import xp

from .from_array_profile import FromArrayProfile


class FromInsightFile(FromArrayProfile):
    r"""
    Profile defined from insight measurement.

    Parameters
    ----------
    file_path: string
        Path to the file created by INSIGHT that contains the full field (e.g. Exyt_0.h5)

    pol: list of 2 complex numbers (dimensionless)
        Polarization vector. It corresponds to :math:`p_u` in the above
        formula ; :math:`p_x` is the first element of the list and
        :math:`p_y` is the second element of the list. Using complex
        numbers enables elliptical polarizations.

    omega0: string or float
        Set the central frequency for the envelope construction. Can be a float value
        in [rad/s], or a string defining the method for automatic frequency detection:
        "barycenter" frequency is averaged over the power profile, "peak" frequency
        corresponding to the location of the maximum of the on-axis spectrum.
    """

    def __init__(self, file_path, pol, omega0="barycenter"):
        # read the data from H5 filed
        with h5py.File(file_path, "r") as hf:
            data = xp.asanyarray(hf["data/Exyt_0"][()], dtype=xp.complex128, order="C")
            t = xp.asanyarray(hf["scales/t"][()], dtype=xp.float64, order="C")
            x = xp.asanyarray(hf["scales/x"][()], dtype=xp.float64, order="C")
            y = xp.asanyarray(hf["scales/y"][()], dtype=xp.float64, order="C")

        # convert data and axes to SI units
        t *= 1e-15
        x *= 1e-3
        y *= 1e-3

        # get the field on axis and local frequencies
        field_onaxis = data[data.shape[0] // 2, data.shape[1] // 2, :]
        omega_array = -xp.gradient(xp.unwrap(xp.angle(field_onaxis)), t)

        # choose the central frequency
        if omega0 == "peak":
            # using peak field frequency
            omega0 = omega_array[xp.abs(field_onaxis).argmax()]
        elif omega0 == "barycenter":
            # or "center of mass" frequency
            omega0 = xp.average(omega_array, weights=xp.abs(field_onaxis) ** 2)
        else:
            assert type(omega0) == float

        # check the complex field convention and correct if needed
        if omega0 < 0:
            omega0 *= -1
            data = xp.conj(data)
            print("Warning: input field will be conjugated")

        # remove the envelope frequency
        data *= xp.exp(1j * omega0 * t[None, None, :])

        # created LASY profile using FromArrayProfile class
        wavelength = 2 * xp.pi * c / omega0
        dim = "xyt"
        axes = {"x": x, "y": y, "t": t}
        super().__init__(
            wavelength=wavelength,
            pol=pol,
            array=data,
            dim=dim,
            axes=axes,
            axes_order=["x", "y", "t"],
        )
