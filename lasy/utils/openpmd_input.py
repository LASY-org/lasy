import numpy as np
from scipy.constants import c


def reorder_array(array, md, dim):
    """Reorder an openPMD array to the lasy representation.

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
        return reorder_array_xyt(array, md)
    else:
        return reorder_array_rt(array, md)


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

    if md.axes in [{0: "z", 1: "y", 2: "x"}, {0: "t", 1: "y", 2: "x"}]:
        array = array.swapaxes(0, 2)

    if "z" in md.axes.values():
        t = (md.z - md.z[0]) / c
        # Flip to get complex envelope in t assuming z = -c*t
        array = np.flip(array, axis=-1)
    else:
        t = md.t
    axes = {"x": md.x, "y": md.y, "t": t}
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

    if md.axes in [{0: "z", 1: "r"}, {0: "t", 1: "r"}]:
        array = array.swapaxes(0, 1)

    if "z" in md.axes.values():
        t = (md.z - md.z[0]) / c
        # Flip to get complex envelope in t assuming z = -c*t
        array = np.flip(array, axis=-1)
    else:
        t = md.t
    r = md.r[md.r.size // 2 :]
    axes = {"r": r, "t": t}

    array = 0.5 * (
        array[array.shape[0] // 2 :, :]
        + np.flip(array[: array.shape[0] // 2, :], axis=0)
    )
    return array, axes
