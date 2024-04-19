import numpy as np
from .speckle_profile import SpeckleProfile
from .stochastic_process_utilities import gen_gaussian_time_series


class GP_RPM_SSD_Profile(SpeckleProfile):
    r"""Generate a speckled laser profile with smoothing by a random phase modulated (RPM) spectral dispersion (SSD).

    This provides a version of smoothing by spectral dispersion (SSD) where the phases are randomly modulated.
    Here the amplitude of the beamlets is always :math:`A_{ml}(t)=1`.
    There are two contributions to the phase :math:`\phi_{ml}` of each beamlet:

    ..math::

        \phi_{ml}(t) = \phi_{PP,ml} + \phi_{SSD,ml}.

    The phase plate part :math:`\phi_{PP,ml}` is the initial phase delay from the randomly sized phase plate sections,
    drawn from uniform distribution on the interval :math:`[0,2\pi]`.
    The phases :math:`\phi_{SSD,ml}(t)` are drawn from a stochastic process
    with Gaussian power spectrum with means :math:`\delta_x,\delta_y` given by the `phase_modulation_amplitude` argument
    and FWHM given by the modulation frequencies :math:`\omega_x,\omega_y`.
    The modulation frequencies :math:`\omega_x,\omega_y` are determined by the
    laser bandwidth and modulation amplitudes according to the relation

        .. math::

            \omega_x = \frac{\Delta_\nu r_x }{2\delta_x},
            \omega_y = \frac{\Delta_\nu r_y }{2\delta_y},

    where :math:`\Delta_\nu` is the resulting relative bandwidth of the laser pulse
    and :math:`r_x, r_y` are additional rotation factors supplied by the user
    in the `transverse_bandwidth_distribution` parameter that determine
    how much of the modulation is in x and how much is in y. [Michel, Eqn. 9.69]

    Parameters
    ----------

    relative_laser_bandwidth : float
        Bandwidth :math:`\Delta_\nu` of the laser pulse, relative to central frequency.
        Only used if ``temporal_smoothing_type`` is ``'FM SSD'``, ``'GP RPM SSD'`` or ``'GP ISI'``.

    phase_modulation_amplitude :list of 2 floats
        Amplitudes :math:`\delta_{x},\delta_{y}` of phase modulation in each transverse direction.
        Only used if ``temporal_smoothing_type`` is ``'FM SSD'`` or ``'GP RPM SSD'``.

    number_color_cycles : list of 2 floats
        Number of color cycles :math:`N_{cc,x},N_{cc,y}` of SSD spectrum to include in modulation
        Only used if ``temporal_smoothing_type`` is ``'FM SSD'`` or ``'GP RPM SSD'``.

    transverse_bandwidth_distribution: list of 2 floats
        Determines how much SSD is distributed in the :math:`x` and :math:`y` directions.
        if `transverse_bandwidth_distribution=[a,b]`, then the SSD frequency modulation is :math:`a/\sqrt{a^2+b^2}` in :math:`x` and :math:`b/\sqrt{a^2+b^2}` in :math:`y`.
        Only used if ``temporal_smoothing_type`` is ``'FM SSD'`` or ``'GP RPM SSD'``.
    """

    def __init__(
        self,
        *speckle_args,
        relative_laser_bandwidth,
        phase_modulation_amplitude,
        number_color_cycles,
        transverse_bandwidth_distribution,
    ):
        super().__init__(*speckle_args)
        self.laser_bandwidth = relative_laser_bandwidth
        # the amplitude of phase along each direction
        self.phase_modulation_amplitude = phase_modulation_amplitude
        # number of color cycles
        self.number_color_cycles = number_color_cycles
        # bandwidth distributed with respect to the two transverse direction
        self.transverse_bandwidth_distribution = transverse_bandwidth_distribution
        normalization = np.sqrt(
            self.transverse_bandwidth_distribution[0] ** 2
            + self.transverse_bandwidth_distribution[1] ** 2
        )
        frac = [
            self.transverse_bandwidth_distribution[0] / normalization,
            self.transverse_bandwidth_distribution[1] / normalization,
        ]
        self.phase_modulation_frequency = [
            self.laser_bandwidth * sf * 0.5 / pma
            for sf, pma in zip(frac, self.phase_modulation_amplitude)
        ]
        self.time_delay = (
            (
                self.number_color_cycles[0] / self.phase_modulation_frequency[0]
                if self.phase_modulation_frequency[0] > 0
                else 0
            ),
            (
                self.number_color_cycles[1] / self.phase_modulation_frequency[1]
                if self.phase_modulation_frequency[1] > 0
                else 0
            ),
        )
        self.dt_update = 1 / self.laser_bandwidth / 50
        return

    def init_gaussian_time_series(
        self,
        series_time,
    ):
        r"""Initialize a time series sampled from a Gaussian process with the correct power spectral density.

        At every time specified by the input `series_time`, calculate the random phases and/or amplitudes.

        This function returns a time series with random phase offsets in x and y at each time.
        The phase offsets are real-valued and centered around the user supplied ``phase_modulation_amplitude``
        :math:`\delta_{x},\delta_{y}`.
        The time series has Gaussian power spectral density, or autocorrelation, with a FWHM ``phase_modulation_frequency``.

        Parameters
        ----------
        series_time: array of times at which to sample from Gaussian process
        time_delay: only required for "SSD" type smoothing
        phase_modulation_frequency: only required for "SSD" type smoothing

        Returns
        -------
        array-like, the supplied `series_time` with some padding at the end for "SSD" smoothing
        array-like, 2 random numbers at every time
        """
        pm_phase0 = gen_gaussian_time_series(
            series_time.size + int(np.sum(self.time_delay) / self.dt_update) + 2,
            self.dt_update,
            2 * np.pi * self.phase_modulation_frequency[0],
            self.phase_modulation_amplitude[0],
        )
        pm_phase1 = gen_gaussian_time_series(
            series_time.size + int(np.sum(self.time_delay) / self.dt_update) + 2,
            self.dt_update,
            2 * np.pi * self.phase_modulation_frequency[1],
            self.phase_modulation_amplitude[1],
        )
        time_interp = np.arange(
            start=0,
            stop=series_time[-1] + np.sum(self.time_delay) + 3 * self.dt_update,
            step=self.dt_update,
        )[: pm_phase0.size]
        return (
            time_interp,
            [
                (np.real(pm_phase0) + np.imag(pm_phase0)) / np.sqrt(2),
                (np.real(pm_phase1) + np.imag(pm_phase1)) / np.sqrt(2),
            ],
        )

    def setup_for_evaluation(self, t_norm):
        self.x_y_dephasing = np.random.standard_normal(2) * np.pi
        self.phase_plate = np.random.uniform(
            -np.pi, np.pi, size=self.n_beamlets[0] * self.n_beamlets[1]
        ).reshape(self.n_beamlets)

        t_max = t_norm[-1]
        series_time = np.arange(0, t_max + self.dt_update, self.dt_update)

        self.series_time, self.time_series = self.init_gaussian_time_series(series_time)
        return

    def beamlets_complex_amplitude(
        self,
        t_now,
    ):
        """Calculate complex amplitude of the beamlets in the near-field, before propagating to the focal plane.

        Parameters
        ----------

        Returns
        -------
        array of complex numbers giving beamlet amplitude and phases in the near-field
        """
        phase_t = np.interp(
            t_now + self.X_lens_index_matrix * self.time_delay[0] / self.n_beamlets[0],
            self.series_time,
            self.time_series[0],
        ) + np.interp(
            t_now + self.Y_lens_index_matrix * self.time_delay[1] / self.n_beamlets[1],
            self.series_time,
            self.time_series[1],
        )
        return np.exp(1j * (self.phase_plate + phase_t))
