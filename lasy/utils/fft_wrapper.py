import numpy as np


def fft(which, arr_in, axes_in, from_domain, verbose=0):
    """
    Perform FFT on a 3D array.

    We use the following conventions:
     - From physical space (x&y or t) to frequency space (kx&ky or omega), the FFT done is data_freq = ifft(ifftshift(data_phys))
     - From frequency space (kx&ky or omega) to physical space (x&y or t), the FFT done is data_phys = fftshift(fft(data_freq))

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

    verbose : integer, optional (default 0)
        Verbosity level. If >0 print some warning statements.

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

    # Set conventions:
    # - From real to frequency, use ifft & fftshift on input data.
    # - From frequency to real, use fft & fftshift on output data.
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
            if verbose > 0:
                print("fft of size 1: do nothing")
            return arr_in, axes_in

        # Set right FFT functions
        if inverse:
            xfft = np.fft.ifft2
            xfftshift = np.fft.ifftshift
        else:
            xfft = np.fft.fft2
            xfftshift = np.fft.fftshift

        # Do the FFT
        arr = np.copy(arr_in)
        if shift_before:
            arr = xfftshift(arr, axes=(0, 1))
        arr_out = xfft(arr, axes=(0, 1))

        # Shift after FFT
        if shift_after:
            arr_out = xfftshift(arr_out, axes=(0, 1))

    else:  # which == "longitudinal"
        # Exit if only 1 element
        if axes_in.size <= 1:
            if verbose > 0:
                print("fft of size 1: do nothing")
            return arr_in, axes_in

        # Set right FFT functions
        if inverse:
            xfft = np.fft.ifft
            xfftshift = np.fft.ifftshift
        else:
            xfft = np.fft.fft
            xfftshift = np.fft.fftshift

        # Do the FFT
        arr = np.copy(arr_in)
        if shift_before:
            arr = xfftshift(arr, axes=-1)
        arr_out = xfft(arr, axis=-1)

        # Shift after FFT
        if shift_after:
            arr_out = xfftshift(arr_out, axes=-1)

    axes_out = frequency_axis(which, axes_in, from_domain)

    return arr_out, axes_out


def frequency_axis(which, axes_in, from_domain):
    """
    Perform FFT on a 3D array.

    We use the following conventions:
     - From physical space (x&y or t) to frequency space (kx&ky or omega), the FFT done is data_freq = ifft(ifftshift(data_phys))
     - From frequency space (kx&ky or omega) to physical space (x&y or t), the FFT done is data_phys = fftshift(fft(data_freq))

    Parameters
    ----------
    which : string
        "transverse" for FFTs in the 2 transverse directions (x,y or kx,ky)
        "longitudinal" for longitudinal direction (t or omega)

    axes_in : list of 1d arrays
        List of axes along which the FFT is to be performed:
        2 elements for which="transverse", 1 for which="longitudinal"

    from_domain : string
        "real" of the FFT is done from real domain (x,y) or (t) to frequency domain (kx, ky) or (omega)
        "frequency" for the opposite way

    Returns
    -------
    axes_out : list of 1d arrays
        if which="transverse", 2 1d arrays for the transverse transformed axes
        if which="longitudinal", 1 1d array for the longitudinal transformed axis
    """
    # Checks, read parameters and set defaults
    assert which in ["transverse", "longitudinal"]
    assert from_domain in ["real", "frequency"]

    # Set conventions:
    # - From real to frequency, use ifft & fftshift on input data.
    # - From frequency to real, use fft & fftshift on output data.
    if from_domain == "real":
        shift_after = False
        inverse = True
    else:
        shift_after = True
        inverse = False

    if which == "transverse":
        # Exit if only 1 element
        if min(axes_in[0].size, axes_in[1].size) < 2:
            return axes_in
        dx = axes_in[0][1] - axes_in[0][0]
        dy = axes_in[1][1] - axes_in[1][0]

        # Set right FFT functions
        if inverse:
            xfftshift = np.fft.ifftshift
        else:
            xfftshift = np.fft.fftshift

        # Build output axes data
        axes_out = [
            np.fft.fftfreq(axes_in[0].size, dx),
            np.fft.fftfreq(axes_in[1].size, dy),
        ]
        if from_domain == "real":
            axes_out[0] *= 2 * np.pi
            axes_out[1] *= 2 * np.pi

        # Shift after FFT
        if shift_after:
            axes_out[0] = xfftshift(axes_out[0])
            axes_out[1] = xfftshift(axes_out[1])

    else:  # which == "longitudinal"
        # Exit if only 1 element
        if axes_in.size <= 1:
            return axes_in
        d = axes_in[1] - axes_in[0]

        # Set right FFT functions
        if inverse:
            xfftshift = np.fft.ifftshift
        else:
            xfftshift = np.fft.fftshift

        # Build output axes data
        axes_out = np.fft.fftfreq(axes_in.size, d)
        if from_domain == "real":
            axes_out *= 2 * np.pi

        # Shift after FFT
        if shift_after:
            axes_out = xfftshift(axes_out)

    return axes_out
