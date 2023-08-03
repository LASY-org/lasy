import numpy as np
from scipy.signal import hilbert
from scipy.constants import c
from skimage.restoration import unwrap_phase
import openpmd_api as io
from openpmd_viewer import OpenPMDTimeSeries
from .from_array_profile import FromArrayProfile
from lasy.utils.laser_utils import get_frequency


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


    theta : float or None, optional
        Only used if the openPMD input is in thetaMode geometry.
        Directly passed to openpmd_viewer.OpenPMDTimeSeries.get_field.
        The angle of the plane of observation, with respect to the x axis
        If `theta` is not None, then this function returns a 2D array
        corresponding to the plane of observation given by `theta`;
        otherwise it returns a full 3D Cartesian array.

    phase_unwrap_1d : boolean (optional)
        Whether the phase unwrapping is done in 1D.
        This is not recommended, as the unwrapping will not be accurate,
        but it might be the only practical solution when dim is 'xyt'.
        If None, it is set to False for dim = 'rt' and True for dim = 'xyt'.
    """

    def __init__(
        self, path, iteration, pol, field, coord=None, envelope=False,
            prefix=None, theta=None, phase_unwrap_1d=None
    ):
        ts = OpenPMDTimeSeries(path)
        F, m = ts.get_field(iteration=iteration, field=field, coord=coord, theta=theta)

        if theta is None: # Envelope obtained from the full 3D array

            assert m.axes in [
                {0: "x", 1: "y", 2: "z"},
                {0: "z", 1: "y", 2: "x"},
                {0: "x", 1: "y", 2: "t"},
                {0: "t", 1: "y", 2: "x"}
            ]

            dim = 'xyt'

            if m.axes in [{0: "z", 1: "y", 2: "x"}, {0: "t", 1: "y", 2: "x"}]:
                F = F.swapaxes(0, 2)

            if "z" in m.axes.values():
                t = (m.z - m.z[0]) / c
            else:
                t = m.t
            axes = {"x": m.x, "y": m.y, "t": t}

            if phase_unwrap_1d is None:
                phase_unwrap_1d = True

            # If array does not contain the envelope but the electric field,
            # extract the envelope with a Hilbert transform
            if envelope == False:
                # Assumes z is last dimension!
                h = hilbert(F)

                if "z" in m.axes.values():
                    # Flip to get complex envelope in t assuming z = -c*t
                    h = np.flip(h, axis=-1)

                # Get central wavelength from array
                axes_list = [axes[i] for i in axes.keys()]
                omg_h, omg0_h = get_frequency(h, axes_list, dim, is_envelope=False,
                    is_hilbert=True, phase_unwrap_1d=phase_unwrap_1d)
                wavelength = 2 * np.pi * c / omg0_h
                array = h * np.exp(1j * omg0_h * t)
            else:
                s = io.Series(path + "/" + prefix + "_%T.h5", io.Access.read_only)
                it = s.iterations[iteration]
                omg0 = it.meshes["laserEnvelope"].get_attribute("angularFrequency")
                wavelength = 2 * np.pi * c / omg0
                array = F

            axes_order = ["x", "y", "t"]

        else: # Envelope assumes axial symmetry processing RZ data
            assert m.axes in [
                {0: "r", 1: "z"},
                {0: "z", 1: "r"},
                {0: "r", 1: "t"},
                {0: "t", 1: "r"}
            ]

            dim = 'rt'

            if m.axes in [{0: "z", 1: "r"}, {0: "t", 1: "r"}]:
                F = F.swapaxes(0, 1)

            if "z" in m.axes.values():
                t = (m.z - m.z[0]) / c
            else:
                t = m.t
            r = m.r[m.r.size//2:]
            axes = {"r": r, "t": t}

            F = F[F.shape[0]//2:,:]

            if phase_unwrap_1d is None:
                phase_unwrap_1d = False
            # If array does not contain the envelope but the electric field,
            # extract the envelope with a Hilbert transform
            if envelope == False:
                # Assumes z is last dimension!
                h = hilbert(F)

                if "z" in m.axes.values():
                    # Flip to get complex envelope in t assuming z = -c*t
                    h = np.flip(h, axis=-1)

                # Get central wavelength from array
                axes_list = [axes[i] for i in axes.keys()]
                omg_h, omg0_h = get_frequency(h, axes_list, dim=dim,
                    is_envelope=False, is_hilbert=True,
                    phase_unwrap_1d=phase_unwrap_1d)
                wavelength = 2 * np.pi * c / omg0_h
                array = h * np.exp(1j * omg0_h * t)
            else:
                s = io.Series(path + "/" + prefix + "_%T.h5", io.Access.read_only)
                it = s.iterations[iteration]
                omg0 = it.meshes["laserEnvelope"].get_attribute("angularFrequency")
                wavelength = 2 * np.pi * c / omg0
                array = F

            axes_order = ["r", "t"]

        print("Wavelength used in the definition of the envelope (nm):", wavelength*1.e9)

        super().__init__(
            wavelength=wavelength,
            pol=pol,
            array=array,
            dim=dim,
            axes=axes,
            axes_order=axes_order,
        )
