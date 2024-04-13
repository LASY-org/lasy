import numpy as np
from .speckle_profile import SpeckleProfile

class FMSSDProfile(SpeckleProfile):
    """Generate a speckled laser profile with smoothing by spectral dispersion (SSD).

    This has temporal smoothing.

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
    def __init__(self, *speckle_args, 
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
        self.transverse_bandwidth_distribution = (
            transverse_bandwidth_distribution
        )
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
                self.number_color_cycles[0]
                / self.phase_modulation_frequency[0]
                if self.phase_modulation_frequency[0] > 0
                else 0
            ),
            (
                self.number_color_cycles[1]
                / self.phase_modulation_frequency[1]
                if self.phase_modulation_frequency[1] > 0
                else 0
            ),
        )
        self.x_y_dephasing = np.random.standard_normal(2) * np.pi
        
    def beamlets_complex_amplitude(
        self, t_now,
    ):
        """Calculate complex amplitude of the beamlets in the near-field, before propagating to the focal plane.

        Parameters
        ----------

        Returns
        -------
        array of complex numbers giving beamlet amplitude and phases in the near-field
        """
        phase_plate = np.random.uniform(
            -np.pi, np.pi, size=self.n_beamlets[0] * self.n_beamlets[1]
        ).reshape(self.n_beamlets)

        phase_t = self.phase_modulation_amplitude[0] * np.sin(
            self.x_y_dephasing[0]
            + 2 * np.pi * self.phase_modulation_frequency[0]
            * (
                t_now - self.X_lens_matrix * self.time_delay[0] / self.n_beamlets[0]
            )
        ) + self.phase_modulation_amplitude[1] * np.sin(
            self.x_y_dephasing[1]
            + 2 * np.pi * self.phase_modulation_frequency[1]
            * (
                t_now - self.Y_lens_matrix * self.time_delay[1] / self.n_beamlets[1]
            )
        )
        return np.exp(1j * (phase_plate + phase_t))
    