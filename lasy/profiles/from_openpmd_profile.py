import numpy as np
import openpmd_api as io
from scipy.constants import c

from lasy.utils.laser_utils import (
    create_grid,
    field_to_envelope,
    vector_potential_to_field,
)
from lasy.utils.openpmd_helper import convert_modes, extract_array, isolate_polarization

from .from_array_profile import FromArrayProfile


class FromOpenPMDProfile(FromArrayProfile):
    r"""
    Profile defined from an openPMD file.

    Upon initialization, read from an openPMD profile, build interpolation objects on the array data and use them to create function evaluate.

    Parameters
    ----------
    file_name : string
        Name of openPMD file, including path, to read the laser field or envelope from.

    envelope_name : string (optional)
        The name of the envelope field (this is not prescribed by the openPMD standard for the envelope).
        If specified, an envelope field is expected from the openPMD file. Otherwise, a full electric field is assumed.
        In the case of a full field, The transverse electric field (Ex & Ey or Er and Etheta) is read, and the polarization is measured from this.

    verbose : bool (optional)
        If true, print some intermediate steps.
    """

    def __init__(self, file_name, envelope_name=None, verbose=False):
        series = io.Series(file_name, io.Access.read_only)
        iterations = np.array(series.iterations)
        i = series.iterations[iterations[-1]]
        is_envelope = envelope_name is not None
        if is_envelope:
            if verbose:
                print("Read envelope")
            m = i.meshes[envelope_name]
            geometry = m.get_attribute("geometry")
            dim = "xyt" if geometry == "cartesian" else "rt"
            omg0 = m.get_attribute("angularFrequency")
            try:
                envelopeField = m.get_attribute("envelopeField")
                pol = m.get_attribute("polarization")
            except Exception:
                envelopeField = "normalized_vector_potential"
                pol = (1, 0)
                print(
                    "WARNING: 'envelopeField' and/or 'polarization' attributes must be specified according to the standard but are currently missing for mesh record "
                    + envelope_name
                    + ", see https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_LaserEnvelope.md. Assumed 'normalized_vector_potential' and (1,0), respectively."
                )
            axes_order, axes, array = extract_array(m, series)
            arrays = convert_modes([array], geometry, is_envelope, verbose)
            array = arrays[0]
            if envelopeField == "normalized_vector_potential":
                if verbose:
                    print("Convert from vector potential to electric field")
                grid = create_grid(array, axes, dim)
                array = vector_potential_to_field(grid, omg0)
        else:
            geometry = i.meshes["E"].get_attribute("geometry")
            if geometry == "cartesian":
                field_list = ["E", "E"]
                coord_list = ["x", "y"]
            else:  # thetaMode
                field_list = ["E", "E"]
                coord_list = ["r", "t"]
            array_list = []
            for count, field in enumerate(field_list):
                # Read the data
                m = i.meshes[field]
                component = coord_list[count]
                axes_order, axes, array = extract_array(m, series, component)
                array_list.append(array)
            # Convert from Er & Etheta at openPMD mode decomposition
            #           to Ex & Ey at LASY mode decomposition.
            array_list = convert_modes(array_list, geometry, is_envelope, verbose)
            dim = "xyt" if geometry == "cartesian" else "rt"
            # Detect whether Ex (index 0) or Ey (index 1) is strongest (major)
            if np.max(np.abs(array_list[0])) >= np.max(np.abs(array_list[1])):
                imajor = 0
            else:
                imajor = 1
            iminor = 1 - imajor
            # Convert major field to envelope, and measure frequency
            grid_major = create_grid(array_list[imajor], axes, dim, is_envelope=False)
            omg0 = field_to_envelope(grid_major, dim)
            # Convert other field to envelope, for the same frequency
            grid_minor = create_grid(array_list[iminor], axes, dim, is_envelope=False)
            field_to_envelope(grid_minor, dim, omg0)
            # Measure polarization state from Ex and Ey
            env_array_list = [None, None]
            env_array_list[imajor] = grid_major.get_temporal_field()
            env_array_list[iminor] = grid_minor.get_temporal_field()
            array, pol = isolate_polarization(env_array_list, dim)
        wavelength = 2 * np.pi * c / omg0

        super().__init__(
            wavelength=wavelength,
            pol=pol,
            array=array,
            dim=dim,
            axes=axes,
            axes_order=axes_order,
        )
