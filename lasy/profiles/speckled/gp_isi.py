import numpy as np
from .speckle_profile import SpeckleProfile
from .stochastic_process_utilities import gen_gaussian_time_series


class GP_ISI_Profile(SpeckleProfile):
    r"""Generate a speckled laser profile with smoothing inspired by Induced Spatial Incoherence (ISI).

    This is a smoothing technique with temporal stochastic variation in the beamlet phases and amplitudes
    to simulate the random phase differences and amplitudes the beamlets pick up when passing through ISI echelons.
    In this case, :math:`\phi_{ml}(t)` and :math:`A_{ml}(t)` are chosen randomly.
    Practically, this is done by drawing the complex amplitudes :math:\tilde A_{ml}(t)`
    from a stochastic process with Guassian power spectral density having mean of 1 and FWHM of twice the laser bandwidth.

    Parameters
    ----------
    relative_laser_bandwidth : float
        Bandwidth :math:`\Delta_\nu` of the incoming laser pulse, relative to the central frequency.

    """

    def __init__(
        self,
        wavelength,
        pol,
        laser_energy,
        focal_length,
        beam_aperture,
        n_beamlets,
        relative_laser_bandwidth,
        do_include_transverse_envelope=True,
        long_profile=None,
    ):
        super().__init__(
            wavelength,
            pol,
            laser_energy,
            focal_length,
            beam_aperture,
            n_beamlets,
            do_include_transverse_envelope,
            long_profile,
        )
        self.laser_bandwidth = relative_laser_bandwidth
        self.dt_update = 1 / self.laser_bandwidth / 50
        return

    def init_gaussian_time_series(
        self,
        series_time,
    ):
        r"""Initialize a time series sampled from a Gaussian process.

        At every time specified by the input `series_time`, calculate the random phases and/or amplitudes.

        * This function returns a time series with random phase offsets in x and y at each time.
            The phase offsets are real-valued and centered around the user supplied ``phase_modulation_amplitude``
            :math:`\delta_{x},\delta_{y}`, with distribution FWHM ``phase_modulation_frequency``.

        Parameters
        ----------
        series_time: array of times at which to sample from Gaussian process

        Returns
        -------
        array-like, the supplied `series_time`
        array-like, either with 2 random numbers at every time
        """
        complex_amp = np.stack(
            [
                np.stack(
                    [
                        gen_gaussian_time_series(
                            series_time.size,
                            self.dt_update,
                            2 * self.laser_bandwidth,
                            1,
                        )
                        for _i in range(self.n_beamlets[1])
                    ]
                )
                for _j in range(self.n_beamlets[0])
            ]
        )
        return complex_amp

    def setup_for_evaluation(self, t_norm):
        """Create or update data used in evaluation."""
        self.x_y_dephasing = np.random.standard_normal(2) * np.pi
        self.phase_plate = np.random.uniform(
            -np.pi, np.pi, size=self.n_beamlets[0] * self.n_beamlets[1]
        ).reshape(self.n_beamlets)

        t_max = t_norm[-1]
        series_time = np.arange(0, t_max + self.dt_update, self.dt_update)

        self.time_series = self.init_gaussian_time_series(series_time)
        return

    def beamlets_complex_amplitude(
        self,
        t_now,
    ):
        """Calculate complex amplitude of the beamlets in the near-field, before propagating to the focal plane.

        Parameters
        ----------
        t_now: float, time at which to evaluate complex amplitude

        Returns
        -------
        array of complex numbers giving beamlet amplitude and phases in the near-field
        """
        return self.time_series[:, :, int(round(t_now / self.dt_update))]
