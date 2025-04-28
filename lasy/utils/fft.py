import numpy as np
from numpy.fft import fft, ifft, fft2, fftshift, ifft2, ifftshift, fftfreq


def fft(arr_in, axes_in, which, inverse=False, shift_before=True, shift_after=False):
    """
    Perform FFT on a 3D array

    Parameters
    ----------
    which : string
        "transverse" for FFTs in the 2 transverse directions (x,y or kx,ky)
        "longitudinal" for longitudinal direction (t or omega)

    arr_in : ndarray
        Array to transform

    axes_in : list of 1d arrays
        List of axes along which the FFT is to be performed:
        2 elements for which="transverse", 1 for which="longitudinal"

    inverse : boolean
        If true, perform ifft. Otherwise, use fft

    shift_before : boolean
        Whether to perform fftshift on data before doing FFT.

    shift_after : boolean
        Whether to perform fftshift on data and output axis after FFT.

    Returns
    -------
    arr_out : ndarray
        3D array after FFT

    axes_out : list of 1d arrays
        if which="transverse", 2 1d arrays for the transverse transformed axes
        if which="longitudinal", 1 1d array for the longitudinal transformed axis
    """
    assert which in ["transverse", "longitudinal"]
    transverse = which == "transverse"
    npoints = [i.size for i in axes_in]
    dx = [i[1] - i[0] for i in axes_in]
    ax = (0, 1) if transverse else 2
    if shift_before:
        if inverse:
            arr = ifftshift(arr_in, axes=ax) if inverse else
        else:
            arr = fftshift(arr_in, axes=ax)
        arr_out = ifft2(arr, axes=ax) if transverse else ifft(arr, axes=ax)
    else:
        arr_out = fft2(arr, axes=ax) if transverse else fft(arr, axes=ax)
    axes_out = axes_in.copy()
    if transverse:
        axes_out[0] = 2 * np.pi * np.fft.fftfreq(npoints[0], dx[0])
        axes_out[1] = 2 * np.pi * np.fft.fftfreq(npoints[1], dx[1])
    else:
        axes_out[0] = 2 * np.pi * np.fft.fftfreq(npoints[0], dx[0])
    if shift_after:
        arr_out = ifftshift(arr, axes=ax) if inverse else fftshift(arr, axes=ax)
        if transverse:
            if inverse:
                axes_out[0] = ifftshift(axes_out[0], axes)
                axes_out[1] = ifftshift(axes_out[1], axes)
            else:
                axes_out[0] = fftshift(axes_out[0], axes)
                axes_out[1] = fftshift(axes_out[1], axes)
        else:
            if inverse:
                axes_out[0] = ifftshift(axes_out[0], axes)
            else:
                axes_out[0] = fftshift(axes_out[0], axes)
    return arr_out, axes_out
