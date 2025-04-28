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

    if transverse and min(axes_in[0].size, axes_in[1].size) <= 1:
        print("fft of size 1: do nothing")
        return arr_in, axes_in
    if not transverse and axes_in.size <= 1:
        print("fft of size 1: do nothing")
        return arr_in, axes_in
    # Build output axes data
    npoints = [i.size for i in axes_in]
    if transverse:
        # List of 2 elements for 2 transverse directions, (x, y) or (kx, ky)
        axes_out = [
            np.fft.fftfreq(npoints[0], axes_in[0][1] - axes_in[0][0]),
            np.fft.fftfreq(npoints[1], axes_in[1][1] - axes_in[1][0]),
        ]
        if from_domain == "real":
            axes_out[0] *= 2 * np.pi
            axes_out[1] *= 2 * np.pi
    else:
        # 1d array for longitudinal direction, t or omega
        axes_out = np.fft.fftfreq(npoints[0], axes_in[1] - axes_in[0])
        if from_domain == "real":
            axes_out *= 2 * np.pi
    if shift_after:
        if transverse:
            if inverse:
                axes_out[0] = np.fft.ifftshift(axes_out[0], axes=ax)
                axes_out[1] = np.fft.ifftshift(axes_out[1], axes=ax)
            else:
                axes_out[0] = np.fft.fftshift(axes_out[0], axes=ax)
                axes_out[1] = np.fft.fftshift(axes_out[1], axes=ax)
        else:
            if inverse:
                axes_out = np.fft.ifftshift(axes_out, axes=ax)
            else:
                axes_out = np.fft.fftshift(axes_out, axes=ax)

    # Perform fftshift of input data if required. Then transform.
    arr = np.copy(arr_in)
    if shift_before:
        if inverse:
            arr = np.fft.ifftshift(arr_in, axes=ax)
        else:
            arr = np.fft.fftshift(arr_in, axes=ax)

    # Do the FFT
    if inverse:
        arr_out = (
            np.fft.ifft2(arr, axes=ax) if transverse else np.fft.ifft(arr, axis=-1)
        )
    else:
        arr_out = np.fft.fft2(arr, axes=ax) if transverse else np.fft.fft(arr, axis=-1)

    # shift after?
    if shift_after:
        if inverse:
            arr_out = np.fft.ifftshift(arr, axes=ax)
        else:
            arr_out = np.fft.fftshift(arr, axes=-1)
    return arr_out, axes_out
