import numpy as np


def fft(which, arr_in, axes_in, from_domain):
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

    from_domain : string
        "real" of the FFT is done from real domain (x,y) or (t) to frequency domain (kx, ky) or (omega)
        "frequency" for the opposite way

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
    # Checks, read parameters and set defaults
    assert which in ["transverse", "longitudinal"]
    assert from_domain in ["real", "frequency"]
    transverse = which == "transverse"
    ax = (0, 1) if transverse else 2

    # Here we set our conventions:
    # - From real space to frequency space, we use ifft and
    #   an fftshift is performed on the input data.
    # - From frequency space to real space, we use fft and
    #   an fftshift is performed on the output data.
    if from_domain == "real":
        shift_before = True
        shift_after = False
        inverse = True
    else:
        shift_before = False
        shift_after = True
        inverse = False

    # Build output axes data
    axes_out = axes_in.copy()
    npoints = [i.size for i in axes_in]
    dx = [i[1] - i[0] for i in axes_in]
    if transverse:
        # List of 2 elements for 2 transverse directions, (x, y) or (kx, ky)
        axes_out[0] = np.fft.fftfreq(npoints[0], dx[0])
        axes_out[1] = np.fft.fftfreq(npoints[1], dx[1])
        if from_domain == "real":
            axes_out[0] *= 2 * np.pi
            axes_out[1] *= 2 * np.pi
    else:
        # list of 1 element for longitudinal direction, t or omega
        axes_out[0] = 2 * np.pi * np.fft.fftfreq(npoints[0], dx[0])
        if from_domain == "real":
            axes_out[0] *= 2 * np.pi

    # Perform fftshift of input data if required. Then transform.
    if shift_before:
        if inverse:
            arr = np.fft.ifftshift(arr_in, axes=ax)
        else:
            arr = np.fft.fftshift(arr_in, axes=ax)
        arr_out = (
            np.fft.ifft2(arr, axes=ax) if transverse else np.fft.ifft(arr, axes=ax)
        )
    else:
        arr_out = np.fft.fft2(arr, axes=ax) if transverse else np.fft.fft(arr, axes=ax)
    if shift_after:
        arr_out = (
            np.fft.ifftshift(arr, axes=ax) if inverse else np.fft.fftshift(arr, axes=ax)
        )
        if transverse:
            if inverse:
                axes_out[0] = np.fft.ifftshift(axes_out[0], axes)
                axes_out[1] = np.fft.ifftshift(axes_out[1], axes)
            else:
                axes_out[0] = np.fft.fftshift(axes_out[0], axes)
                axes_out[1] = np.fft.fftshift(axes_out[1], axes)
        else:
            if inverse:
                axes_out[0] = np.fft.ifftshift(axes_out[0], axes)
            else:
                axes_out[0] = np.fft.fftshift(axes_out[0], axes)
    return arr_out, axes_out
