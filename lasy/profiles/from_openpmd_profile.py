import numpy as np
from scipy.constants import c
import openpmd_api as io
from openpmd_viewer import OpenPMDTimeSeries
from .from_array_profile import FromArrayProfile
from lasy.utils.laser_utils import field_to_envelope
from lasy.utils.openpmd_input import reorder_array, create_grid


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

    verbose : boolean (optional)
        Whether to print extended information.
    """

    def __init__(
        self,
        path,
        iteration,
        pol,
        field,
        coord=None,
        prefix=None,
        theta=None,
        phase_unwrap_1d=None,
        verbose=False,
    ):
        ts = OpenPMDTimeSeries(path)
        F, m = ts.get_field(iteration=iteration, field=field, coord=coord, theta=theta)

        if theta is None:  # Envelope obtained from the full 3D array
            dim = "xyt"
            if phase_unwrap_1d is None:
                phase_unwrap_1d = True
            axes_order = ["x", "y", "t"]

        else:  # Envelope assumes axial symmetry processing RZ data
            dim = "rt"
            if phase_unwrap_1d is None:
                phase_unwrap_1d = False
            axes_order = ["r", "t"]

        F, axes = reorder_array(F, m, dim)

        # If array does not contain the envelope but the electric field,
        # extract the envelope with a Hilbert transform
        if not np.iscomplexobj(F):
            grid = create_grid(F, axes, dim)
            grid, omg0 = field_to_envelope(grid, dim, phase_unwrap_1d)
            array = grid.field[0]
        else:
            s = io.Series(path + "/" + prefix + "_%T.h5", io.Access.read_only)
            it = s.iterations[iteration]
            omg0 = it.meshes["laserEnvelope"].get_attribute("angularFrequency")
            array = F

        wavelength = 2 * np.pi * c / omg0
        if verbose:
            print(
                "Wavelength used in the definition of the envelope (nm):",
                wavelength * 1.0e9,
            )

        super().__init__(
            wavelength=wavelength,
            pol=pol,
            array=array,
            dim=dim,
            axes=axes,
            axes_order=axes_order,
        )
