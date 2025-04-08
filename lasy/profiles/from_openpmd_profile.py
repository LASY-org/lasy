import numpy as np
import openpmd_api as io
from scipy.constants import c

from lasy.utils.laser_utils import (
    create_grid,
    field_to_envelope,
    vector_potential_to_field,
)
from lasy.utils.openpmd_helper import convert_modes, extract_array

from .from_array_profile import FromArrayProfile


class FromOpenPMDProfile(FromArrayProfile):
    r"""
    Profile defined from an openPMD file.

    Upon initialization, read from an openPMD profile, build interpolation objects on the array data and use them to create function evaluate.

    Parameters
    ----------
    filename : string
        Name of openPMD file to read the envelope from, including path.

    is_envelope : bool
        Whether the openPMD file represents a laser envelope.
        Otherwise, electric field is assumed, and its envelope is extracted.

    field_name : string (optional)
        Required if is_envelope is True.
        The name of the envelope field (this is not prescribed by the openPMD standard for the envelope).

    verbose : bool (optional)
        If true, print some more intermediate steps.
    """

    def __init__(self, filename, is_envelope, field_name=None, verbose=False):
        series = io.Series(filename, io.Access.read_only)
        iterations = np.array(series.iterations)
        i = series.iterations[iterations[-1]]
        if is_envelope:
            if verbose:
                print("Read envelope")
            assert field_name is not None, (
                "field_name must be specified for an envelope"
            )
            m = i.meshes[field_name]
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
                    + field_name
                    + ", see https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_LaserEnvelope.md. Assumed 'normalized_vector_potential' and (1,0), respectively."
                )
            axes_order, axes, array = extract_array(m, series)
            array = convert_modes([array], geometry, is_envelope, verbose)
            if envelopeField == "normalized_vector_potential":
                if verbose:
                    print("Convert from vector potential to electric field")
                grid = create_grid(array, axes, dim)
                array = vector_potential_to_field(grid, omg0)
        else:
            geometry = i.meshes["E"].get_attribute("geometry")
            if geometry == "cartesian":
                field_list = ["E"]
                coord_list = ["x"]
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
            array = convert_modes(array_list, geometry, is_envelope, verbose)
            dim = "xyt" if geometry == "cartesian" else "rt"
            grid = create_grid(array, axes, dim, is_envelope=False)
            omg0 = field_to_envelope(grid, dim)
            array = grid.get_temporal_field()
            pol = (1, 0)
        wavelength = 2 * np.pi * c / omg0

        super().__init__(
            wavelength=wavelength,
            pol=pol,
            array=array,
            dim=dim,
            axes=axes,
            axes_order=axes_order,
        )
