import numpy as np
from scipy.signal import hilbert
from scipy.constants import c
import openpmd_api as io
from openpmd_viewer import OpenPMDTimeSeries
from .from_array_profile import FromArrayProfile


class FromOpenPMDProfile(FromArrayProfile):
    r"""
    Profile defined from an openPMD file.

    Parameters
    ----------
    path : string
        Path to the openPMD file containing the laser field or envelope.
        Passed directly OpenPMDTimeSeries.

    iteration : int
        Iteration at which the argument is read.
        Passed directly OpenPMDTimeSeries.

    pol : list of 2 complex numbers (dimensionless)
        Polarization vector. It corresponds to :math:`p_u` in the above
        formula ; :math:`p_x` is the first element of the list and
        :math:`p_y` is the second element of the list. Using complex
        numbers enables elliptical polarizations.

    field : string
        Name of the field containing the laser pulse
        Passed directly OpenPMDTimeSeries.

    coord : string
        Name of the field containing the laser pulse
        Passed directly OpenPMDTimeSeries.

    envelope : boolean
        Whether the file represents a laser envelope.
        If not, the envelope is obtained from the electric field
        using a Hilbert transform

    prefix : string
        Prefix of the openPMD file from which the envelope is read.
        Only used when envelope=True.
        The provided iteration is read from <path>/<prefix>_%T.h5.

    wavelength : float (in meter)
        The main laser wavelength :math:`\\lambda_0` of the laser, which defines
        :math:`\\omega_0`, according to :math:`\\omega_0 = 2\\pi c/\\lambda_0`.
        This argument is optional, and will overwrite the following default
        behavior:
        - if envelope is True, the central wavelength is read from the openPMD
        file at openPMD envelope standard.
        - if envelope is False, the central wavelength is obtained from the
        Hilbert transform.
    """

    def __init__(
        self,
        path,
        iteration,
        pol,
        field,
        coord=None,
        envelope=False,
        prefix=None,
        wavelength=None,
    ):
        ts = OpenPMDTimeSeries(path)
        F, m = ts.get_field(iteration=iteration, field=field, coord=coord, theta=None)
        assert m.axes in [
            {0: "x", 1: "y", 2: "z"},
            {0: "z", 1: "y", 2: "x"},
            {0: "x", 1: "y", 2: "t"},
            {0: "t", 1: "y", 2: "x"},
        ]

        if m.axes in [{0: "z", 1: "y", 2: "x"}, {0: "t", 1: "y", 2: "x"}]:
            F = F.swapaxes(0, 2)

        if "z" in m.axes.values():
            dt = m.dz / c
            t = (m.z - m.z[0]) / c
        else:
            dt = m.dt
            t = m.t
        axes = {"x": m.x, "y": m.y, "t": t}

        # If array does not contain the envelope but the electric field,
        # extract the envelope with a Hilbert transform
        if envelope == False:
            # Assumes z is last dimension!
            h = hilbert(F)

            if "z" in m.axes.values():
                # Flip to get complex envelope in t assuming z = -c*t
                h = np.flip(h, axis=2)

            # Get central wavelength from array
            phase = np.unwrap(np.angle(h))
            omg0_h = (
                -np.average(
                    np.diff(phase),
                    weights=0.5 * (np.abs(h[:, :, 1:]) + np.abs(h[:, :, :-1])),
                )
                / dt
            )
            wavelength_loc = 2 * np.pi * c / omg0_h
            if wavelength is not None:
                wavelength_loc = wavelength
            array = h * np.exp(1j * omg0_h * t)
        else:
            if wavelength is None:
                s = io.Series(path + "/" + prefix + "_%T.h5", io.Access.read_only)
                it = s.iterations[iteration]
                omg0 = it.meshes["laserEnvelope"].get_attribute("angularFrequency")
                wavelength_loc = 2 * np.pi * c / omg0
            else:
                wavelength_loc = wavelength
            array = F

        axes_order = ["x", "y", "t"]
        super().__init__(
            wavelength=wavelength_loc,
            pol=pol,
            array=array,
            axes=axes,
            axes_order=axes_order,
        )
