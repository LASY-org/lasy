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
    # ax = (0, 1) if transverse else -1

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

    if which == "transverse":
        # Exit if only 1 element
        if min(axes_in[0].size, axes_in[1].size) < 2:
            print("fft of size 1: do nothing")
            return arr_in, axes_in
        dx = axes_in[0][1] - axes_in[0][0]
        dy = axes_in[1][1] - axes_in[1][0]

        # Set right FFT functions
        if inverse:
            xfft = np.fft.ifft2
            xfftshift = np.fft.ifftshift
        else:
            xfft = np.fft.fft2
            xfftshift = np.fft.fftshift

        # Build output axes data
        axes_out = [
            np.fft.fftfreq(axes_in[0].size, dx),
            np.fft.fftfreq(axes_in[1].size, dy),
        ]
        if from_domain == "real":
            axes_out[0] *= 2 * np.pi
            axes_out[1] *= 2 * np.pi

        # Do the FFT
        arr = np.copy(arr_in)
        if shift_before:
            arr = xfftshift(arr, axes=(0, 1))
        arr_out = xfft(arr, axes=(0, 1))

        # Shift after FFT
        if shift_after:
            axes_out[0] = xfftshift(axes_out[0])
            axes_out[1] = xfftshift(axes_out[1])
            arr_out = xfftshift(arr_out, axes=(0, 1))

    else:  # "longitudinal"
        # Exit if only 1 element
        if axes_in.size <= 1:
            print("fft of size 1: do nothing")
            return arr_in, axes_in
        d = axes_in[1] - axes_in[0]

        # Set right FFT functions
        if inverse:
            xfft = np.fft.ifft
            xfftshift = np.fft.ifftshift
        else:
            xfft = np.fft.fft
            xfftshift = np.fft.fftshift

        # Build output axes data
        axes_out = np.fft.fftfreq(axes_in.size, d)
        if from_domain == "real":
            axes_out *= 2 * np.pi

        # Do the FFT
        arr = np.copy(arr_in)
        if shift_before:
            arr = xfftshift(arr, axes=-1)
        arr_out = xfft(arr, axis=-1)

        # Shift after FFT
        if shift_after:
            axes_out = xfftshift(axes_out)
            arr_out = xfftshift(arr_out, axes=-1)

    return arr_out, axes_out
