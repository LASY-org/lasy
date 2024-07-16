from lasy.backend import xp
from scipy.constants import c

from .longitudinal_profile import LongitudinalProfile


class LongitudinalProfileFromData(LongitudinalProfile):
    """
    Class for longitudinal laser profile created using data.

    The data used can either come from an experimental measurement
    or from the output of another code. This data is then used to
    define the longitudinal profile of the laser pulse.

    The data may be supplied in either the spectral or temporal
    domain. The data should be passed as a structure (defined
    below). If spectral data is passed, it will be converted to
    the temporal domain.

    Parameters
    ----------
    data : structure
        The data structure comprises several items indexed by
        a series of keys

        datatype : string
            The domain in which the data has been passed. Options
            are 'spectral' and 'temporal'

        axis : ndarrays of floats
            The horizontal axis of the pulse duration measurement.
            The array must be monotonously increasing.
            When datatype is 'spectral' axis is wavelength in meters.
            When datatype is 'temporal' axis is time in seconds.

        intensity : ndarrays of floats
            The vertical axis of the pulse duration measurement.
            Spectral (resp. temporal) intensity when datatype is 'spectral' (resp 'temporal').

        phase : ndarray of floats
            If provided, this phase will be added to the pulse.
            When datatype is 'spectral' phase is spectral phase.
            When datatype is 'temporal' phase is temporal phase.

        dt : float
            Only required when datatype is 'spectral'. In this
            case this defines the user requested resolution in
            the conversion from the spectral to the temporal
            domain.

        wavelength : float
            Only required when datatype is 'temporal'. Then,
            this is the central wavelength of the pulse

    lo, hi : floats (seconds)
        Lower and higher ends of the required domain of the data.
        The data imported will be cut to this range prior to
        being incorporated into the ``lasy`` pulse.
    """

    def __init__(self, data, lo, hi):
        if data["datatype"] == "spectral":
            # First find central frequency
            wavelength = data["axis"]
            assert xp.all(
                xp.diff(wavelength) > 0
            ), 'data["axis"] must be in monotonously increasing order.'
            spectral_intensity = data["intensity"]
            if data.get("phase") is None:
                spectral_phase = xp.zeros_like(wavelength)
            else:
                spectral_phase = data["phase"]
            dt = data["dt"]
            cwl = xp.sum(spectral_intensity * wavelength) / xp.sum(spectral_intensity)
            cfreq = c / cwl
            # Determine required sampling frequency for desired dt
            sample_freq = 1 / dt
            # Determine number of points in temporal domain. This is the number of
            # points required to maintain the input spectral resolution while spanning
            # enough spectrum to achieve the desired temporal resolution.
            indx = xp.argmin(xp.abs(wavelength - cwl))
            dfreq = xp.abs(c / wavelength[indx] - c / wavelength[indx + 1])
            N = int(sample_freq / dfreq)
            freq = xp.linspace(cfreq - sample_freq / 2, cfreq + sample_freq / 2, N)
            # interpolate the spectrum onto this new array
            freq_intensity = xp.interp(
                freq, c / wavelength[::-1], spectral_intensity[::-1], left=0, right=0
            )
            freq_phase = xp.interp(
                freq, c / wavelength[::-1], spectral_phase[::-1], left=0, right=0
            )

            freq_amplitude = xp.sqrt(freq_intensity)

            # Inverse Fourier Transform to the time domain
            t_amplitude = (
                xp.fft.fftshift(
                    xp.fft.ifft(
                        xp.fft.ifftshift(freq_amplitude * xp.exp(-1j * freq_phase))
                    )
                )
                / dt
            )
            time = xp.linspace(-dt * N / 2, dt * N / 2 - dt, N)

            # Extract intensity and phase
            temporal_intensity = xp.abs(t_amplitude) ** 2
            temporal_intensity /= xp.max(temporal_intensity)
            temporal_phase = xp.unwrap(-xp.angle(t_amplitude))
            temporal_phase -= temporal_phase[xp.argmin(xp.abs(time))]

        elif data["datatype"] == "temporal":
            time = data["axis"]
            temporal_intensity = data["intensity"]
            if data.get("phase") is None:
                temporal_phase = xp.zeros_like(time)
            else:
                temporal_phase = data["phase"]
            cwl = data["wavelength"]

        else:
            raise Exception("datatype must be 'spectral' or 'temporal'")

        super().__init__(cwl)

        # Finally crop the temporal domain to the physical domain
        # of interest

        tIndLo = xp.argmin(xp.abs(time - lo))
        tIndHi = xp.argmin(xp.abs(time - hi))

        self.time = time[tIndLo:tIndHi]
        self.temporal_intensity = temporal_intensity[tIndLo:tIndHi]
        self.temporal_phase = temporal_phase[tIndLo:tIndHi]

    def evaluate(self, t):
        """
        Return the longitudinal field envelope.

        Parameters
        ----------
        t : ndarray of floats
            Define points on which to evaluate the envelope

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the longitudinal envelope at the
            specified points. This array has the same shape as the array t.
        """
        intensity = xp.interp(t, self.time, self.temporal_intensity)
        phase = xp.interp(t, self.time, self.temporal_phase)

        envelope = xp.sqrt(intensity) * xp.exp(-1j * phase)

        return envelope
