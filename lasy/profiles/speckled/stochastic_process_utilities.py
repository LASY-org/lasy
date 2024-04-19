import numpy as np


def gen_gaussian_time_series(t_num, dt, fwhm, rms_mean):
    """Generate a discrete time series that has gaussian power spectrum.

    Credit Han Wen, possibly NRL

    Parameters
    ----------
    t_num: number of grid points in time
    fwhm: full width half maximum of the power spectrum
    rms_mean: root-mean-square average of the spectrum

    Returns
    -------
    temporal_amplitude: a time series array of complex numbers with shape [t_num]
    """
    if fwhm == 0.0:
        temporal_amplitude = np.zeros(t_num, dtype=np.complex128)
    else:
        omega = np.fft.fftshift(np.fft.fftfreq(t_num, d=dt))
        psd = np.exp(-np.log(2) * 0.5 * np.square(omega / fwhm * 2 * np.pi))
        spectral_amplitude = np.array(psd) * (
            np.random.normal(size=t_num) + 1j * np.random.normal(size=t_num)
        )
        temporal_amplitude = np.fft.ifftshift(
            np.fft.fft(np.fft.fftshift(spectral_amplitude))
        )
        temporal_amplitude *= rms_mean / np.sqrt(
            np.mean(np.square(np.abs(temporal_amplitude)))
        )
    return temporal_amplitude
