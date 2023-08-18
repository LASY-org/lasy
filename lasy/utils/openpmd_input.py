import numpy as np
from .laser_utils import dummy_z_to_t

def refactor_array(array, md, dim):
    """Refactor an openPMD array to the lasy representation.

    Parameters
    ----------
    array : ndarray
        The field array to be reordered.
    md : FieldMetaInformation
        The openPMD metadata of the field.
    dim : {'xyt, 'rt'}
        The dimensionality of the array.

    Returns
    -------
    array : ndarray
        The reordered array.
    axes : dict
        A dictionary with the lasy axes information for the array.
    """
    if dim == "xyt":
        array, axes = reorder_array_xyt(array, md)
    else:
        array, axes = reorder_array_rt(array, md)

    if "z" in axes.keys:
        # Data uses z representation, need to convert to
        # t representation used inside lasy
        array, axes = convert_z_to_t(array, axes, dim, dummy=True)

    return array, axes

def reorder_array_xyt(array, md):
    """Reorder an openPMD array to the lasy representation in `xyt` geometry.

    Parameters
    ----------
    array : ndarray
        The field array to be reordered.
    md : FieldMetaInformation
        The openPMD metadata of the field.

    Returns
    -------
    array : ndarray
        The reordered array.
    axes : dict
        A dictionary with the lasy axes information for the array.
    """
    assert md.axes in [
        {0: "x", 1: "y", 2: "z"},
        {0: "z", 1: "y", 2: "x"},
        {0: "x", 1: "y", 2: "t"},
        {0: "t", 1: "y", 2: "x"},
    ]

    if md.axes == {0: "z", 1: "y", 2: "x"}:
        array = array.swapaxes(0, 2)
        axes = {"x": md.x, "y": md.y, "z": md.z}
    elif md.axes == {0: "t", 1: "y", 2: "x"}:
        array = array.swapaxes(0, 2)
        axes = {"x": md.x, "y": md.y, "t": md.t }

    return array, axes


def reorder_array_rt(array, md):
    """Reorder an openPMD array to the lasy representation in `rt` geometry.

    Parameters
    ----------
    array : ndarray
        The field array to be reordered.
    md : FieldMetaInformation
        The openPMD metadata of the field.

    Returns
    -------
    array : ndarray
        The reordered array.
    axes : dict
        A dictionary with the lasy axes information for the array.
    """
    assert md.axes in [
        {0: "r", 1: "z"},
        {0: "z", 1: "r"},
        {0: "r", 1: "t"},
        {0: "t", 1: "r"},
    ]

    if md.axes == {0: "z", 1: "r"}:
        array = array.swapaxes(0, 1)
        axes = {"r": md.r[md.r.size // 2 :], "z": md.z}
    if md.axes == {0: "t", 1: "r"}:
        array = array.swapaxes(0, 1)
        axes = {"r": md.r[md.r.size // 2 :], "t": md.t}

    array = 0.5 * (
        array[array.shape[0] // 2 :, :]
        + np.flip(array[: array.shape[0] // 2, :], axis=0)
    )

    return array, axes
