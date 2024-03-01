import numpy as np
import h5py
from scipy.constants import c
from .from_array_profile import FromArrayProfile


class FromInsightFile(FromArrayProfile):
    r"""
    Profile defined from insight measurement.

    Parameters
    ----------
    file_path: string
        Path to the file created by insight ....

    pol : list of 2 complex numbers (dimensionless)
        Polarization vector. It corresponds to :math:`p_u` in the above
        formula ; :math:`p_x` is the first element of the list and
        :math:`p_y` is the second element of the list. Using complex
        numbers enables elliptical polarizations.
    """

    def __init__(self, file_path, pol):
        # read the data from H5 filed
        with h5py.File(file_path, "r") as hf:
            data = np.asanyarray(hf["data/Exyt_0"][()], dtype=np.complex128, order="C")
            t = np.asanyarray(hf["scales/t"][()], dtype=np.float64, order="C")
            x = np.asanyarray(hf["scales/x"][()], dtype=np.float64, order="C")
            y = np.asanyarray(hf["scales/y"][()], dtype=np.float64, order="C")

        # convert data and axes to SI units
        t *= 1e-15
        x *= 1e-3
        y *= 1e-3

        # get central frequency from the field on axis,
        field_onaxis = data[data.shape[0] // 2, data.shape[1] // 2, :]
        omega_array = -np.gradient(np.unwrap(np.angle(field_onaxis)), t)
        # using peak field frequency
        omega0 = omega_array[np.abs(field_onaxis).argmax()]
        # or "center of mass" frequency
        ## omega0 = np.average(omega_array, weights=np.abs(env)**2)

        # check the complex field convention and correct if needed
        if omega0<0:
            omega0 *= -1
            data = np.conj(data)
            print('Warning: input field will be conjugated')

        # remove the envelope frequencey
        data *= np.exp(1j * omega0 * t[None, None, :])

        # created LASY profile using FromArrayProfile class
        wavelength = 2 * np.pi * c / omega0
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
