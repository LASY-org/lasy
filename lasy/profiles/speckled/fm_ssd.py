import numpy as np
from .speckle_profile import SpeckleProfile


class FM_SSD_Profile(SpeckleProfile):
    r"""Generate a speckled laser profile with smoothing by frequency modulated (FM) spectral dispersion (SSD).

    In frequency-modulated smoothing by spectral dispersion, or FM-SSD, the amplitude of the beamlets is always :math:`A_{ml}(t)=1`.
    There are two contributions to the phase :math:`\phi_{ml}` of each beamlet:

    .. math::

        \phi_{ml}(t)=\phi_{PP,ml}+\phi_{SSD,ml}.

    The phase plate part :math:`\phi_{PP,ml}` is the initial phase delay from the randomly sized phase plate sections,
    drawn from uniform distribution on the interval :math:`[0,2\pi]`.
    The temporal smoothing is from the SSD term:

    .. math::

        \begin{aligned}
        \phi_{SSD,ml}(t)&=\delta_{x} \sin\left(\omega_{x} t + 2\pi\frac{mN_{cc,x}}{N_{bx}}\right)\\
        &+\delta_{y} \sin\left(\omega_{y} t + 2\pi\frac{lN_{cc,y}}{N_{by}}\right).
        \end{aligned}

    The modulation frequencies :math:`\omega_x,\omega_y` are determined by the
    laser bandwidth and modulation amplitudes according to the relation

    .. math::

        \omega_x = \frac{\Delta_\nu r_x }{2\delta_x},
        \omega_y = \frac{\Delta_\nu r_y }{2\delta_y},

    where :math:`\Delta_\nu` is the relative bandwidth of the laser pulse
    and :math:`r_x, r_y` are additional rotation factors supplied by the user
    in the `transverse_bandwidth_distribution` parameter that determine
    how much of the modulation is in x and how much is in y. [Michel, Eqn. 9.69]

    Parameters
    ----------
    relative_laser_bandwidth : float
        Resulting bandwidth :math:`\Delta_\nu` of the laser pulse, relative to central frequency, due to the frequency modulation.

    phase_modulation_amplitude :list of 2 floats
        Amplitudes :math:`\delta_{x},\delta_{y}` of phase modulation in each transverse direction.

    number_color_cycles : list of 2 floats
        Number of color cycles :math:`N_{cc,x},N_{cc,y}` of SSD spectrum to include in modulation

    transverse_bandwidth_distribution: list of 2 floats
        Determines how much SSD is distributed in the :math:`x` and :math:`y` directions.
        if `transverse_bandwidth_distribution=[a,b]`, then the SSD frequency modulation is :math:`r_x=a/\sqrt{a^2+b^2}` in :math:`x` and :math:`r_y=b/\sqrt{a^2+b^2}` in :math:`y`.
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
        self.x_y_dephasing = np.random.standard_normal(2) * np.pi
        self.phase_plate = np.random.uniform(
            -np.pi, np.pi, size=self.n_beamlets[0] * self.n_beamlets[1]
        ).reshape(self.n_beamlets)

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
        phase_t = self.phase_modulation_amplitude[0] * np.sin(
            self.x_y_dephasing[0]
            + 2
            * np.pi
            * self.phase_modulation_frequency[0]
            * (t_now - self.X_lens_matrix * self.time_delay[0] / self.n_beamlets[0])
        ) + self.phase_modulation_amplitude[1] * np.sin(
            self.x_y_dephasing[1]
            + 2
            * np.pi
            * self.phase_modulation_frequency[1]
            * (t_now - self.Y_lens_matrix * self.time_delay[1] / self.n_beamlets[1])
        )
        return np.exp(1j * (self.phase_plate + phase_t))
