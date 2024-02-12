import numpy as np
import h5py
from scipy.constants import c

from .from_array_profile import FromArrayProfile


class FromInsightFile(FromArrayProfile):
    r"""
    Profile defined from numpy array directly.

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
        with h5py.File(file_path, "r") as hf:
            data = np.asanyarray(hf["data/Exyt_0"][()], dtype=np.complex128, order="C")

            t = 1e-15 * np.asanyarray(hf["scales/t"][()], dtype=np.float64, order="C")

            x = 1e-3 * np.asanyarray(hf["scales/x"][()], dtype=np.float64, order="C")

            y = 1e-3 * np.asanyarray(hf["scales/x"][()], dtype=np.float64, order="C")

        # get central frequency from the field on axis
        env = data[data.shape[0] // 2, data.shape[1] // 2, :]
        omega_array = np.gradient(np.unwrap(np.angle(env)), t)

        # frequency at `t=0`
        # omega0 = omega_array[np.abs(t).argmin()]

        # "center of mass" frequency
        ## omega0 = np.average(omega_array, weights=np.abs(env)**2)

        # peak field frequency
        omega0 = omega_array[np.abs(env).argmax()]

        wavelength = 2 * np.pi * c / omega0

        data = np.conjugate(data)
        data *= np.exp(1j * omega0 * t[None, None, :])

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
