import numpy as np
from scipy.constants import c

from .profile import Profile


class SpeckleProfile(Profile):
    r"""
    Derived class for the profile of a speckled laser pulse.

    Speckled lasers are used to mitigate laser-plasma interactions in fusion and ion acceleration contexts.
    More on the subject can be found in chapter 9 of `Introduction to Laser-Plasma Interactions <https://link.springer.com/book/10.1007/978-3-031-23424-8>`__. 
    A speckled laser beam is a laser that is deliberately divided transversely into several beamlets in the near-field.
    This is done with a near-field phase plate divided into 
    The phase plate provides a different phase to each beamlet, which then propagate incoherently and combine in the far field.
    This profile admits several options for calculating the amplitudes and phases of the beamlets:
    
    * Continuous phase plates (CPP)
    * CPP + Smoothing by spectral dispersion (SSD)
    * CPP + a generalization of SSD that has temporal stochastic variation in the beamlet phases
    * Induced spatial incoherence (ISI), which has temporal stochastic variation in the beamlet phases and amplitudes

    More precisely, the electric field corresponds to:

    .. math::

        \begin{aligned}
        E_u(\boldsymbol{x}_\perp,t) &= Re\left[ E_0 
        \sum_{j=1}^{N_{bx}\times N_{by}} A_j 
        {\rm sinc}\left(\frac{\pi D_xx}{\lambda_0 f}\right)
        {\rm sinc}\left(\frac{\pi D_yy}{\lambda_0 f}\right) 
        \right.
        \\
        & \left. \times\exp\left(i\boldsymbol{k}_{\perp,j}\cdot\boldsymbol{x}_\perp 
        + i\phi_{{\rm RPP/CPP},j}+i\psi_{{\rm SSD/ISI},j}(t)\right) \times p_u
        \right]
        \end{aligned}

    where :math:`u` is either :math:`x` or :math:`y`, :math:`p_u` is
    the polarization vector, :math:`Re` represent the real part, and
    :math:`\boldsymbol{x}_\perp=(x,y)` is the transverse coordinate (orthogonal
    to the propagation direction).
    Several quantities are computed internally to the code depending on the
    method of smoothing chosen, including the beamlet amplitude :math:`A_j`,
    the beamlet wavenumber at focus :math:`k_{\perp,j}`,
    the phase contribution :math:`\phi_{{\rm RPP/CPP},j}` from the phase plate,
    and the phase contribution :math:`\psi_{{\rm SSD/ISI},j}(t)` from the smoothing.
    The other parameters in this formula are defined below.

    This is an adapation of work by `Han Wen <https://github.com/Wen-Han/LasersSmoothing2d>`__ to LASY.


    Notes
    -----
    This assumes a flat-top rectangular laser and so a rectangular arrangement of beamlets in the near-field.

    Parameters
    ----------
    wavelength : float (in meter)
        The main laser wavelength :math:`\lambda_0` of the laser, which
        defines :math:`\omega_0` in the above formula, according to
        :math:`\omega_0 = 2\pi c/\lambda_0`.

    pol : list of 2 complex numbers (dimensionless)
        Polarization vector. It corresponds to :math:`p_u` in the above
        formula ; :math:`p_x` is the first element of the list and
        :math:`p_y` is the second element of the list. Using complex
        numbers enables elliptical polarizations.

    focal_length : float (in meter)
        Focal length of lens :math:`f` just after the RPP/CPP.

    beam_aperture : list of 2 floats (in meters)
        Width :math:`D_x,D_y` of the rectangular beam in the near-field, i.e., size of the illuminated region of the RPP/CPP.

    n_beamlets : list of integers
        Number of RPP/CPP elements :math:`N_{bx},N_{by}` in each direction, in the near field.

    temporal_smoothing_type : string
        Which method for beamlet production and evolution is used.
        Can be ``'FM SSD'``, ``'GS RPM SSD'``, or ``'GS ISI'``

        - ``'FM SSD'``: frequency modulated (FM) Smoothing by Spectral Dispersion (SSD)
        - ``'GP RPM SSD'``: Gaussian process (GP) Random Phase Modulated (RPM) SSD

        An idealized form of SSD where each beamlet has random phase
        determined by sampling from a Gaussian stochastic process.

        - ``'GP ISI'``: GP Induced spatial incoherence (ISI)

        An idealized form of ISI where each beamlet has random phase and amplitude
        sampled from a Gaussian stochastic process.

    relative_laser_bandwidth : float
        Bandwidth of laser pulse, relative to central frequency.

    ssd_phase_modulation_amplitude : 2-tuple of floats
        Amplitude of phase modulation in each transverse direction.
        Only used if `temporal_smoothing_type` is `FM SSD`.

    ssd_number_color_cycles : list of 2 floats
        Number of color cycles of SSD spectrum to include in modulation
        Only used if `temporal_smoothing_type` is `FM SSD`.

    ssd_transverse_bandwidth_distribution: list of 2 floats
        Determines how much SSD is distributed in the `x` and `y` directions.
        if `ssd_transverse_bandwidth_distribution=[a,b]`, then the SSD frequency modulation is `a/sqrt(a^2+b^2)` in `x` and `b/sqrt(a^2+b^2)` in `y`.
        Only used if `temporal_smoothing_type` is `FM SSD`.

    do_include_transverse_envelope : boolean, (optional, default False)
        Whether to include the transverse sinc envelope or not.
        I.e. whether it is assumed to be close enough to the laser axis to neglect the transverse field decay.
    """

    def __init__(
        self,
        wavelength,
        pol,
        focal_length,
        beam_aperture,
        n_beamlets,
        temporal_smoothing_type,
        relative_laser_bandwidth,
        ssd_phase_modulation_amplitude=None,
        ssd_number_color_cycles=None,
        ssd_transverse_bandwidth_distribution=None,
        do_include_transverse_envelope=False,
    ):
        super().__init__(wavelength, pol)
        self.wavelength = wavelength
        self.focal_length = focal_length
        self.beam_aperture = np.array(beam_aperture, dtype="float")
        self.n_beamlets = np.array(n_beamlets, dtype="int")
        self.temporal_smoothing_type = temporal_smoothing_type
        self.laser_bandwidth = relative_laser_bandwidth

        # time interval to update the speckle pattern, roughly update 50 times every bandwidth cycle
        self.dt_update = 1 / self.laser_bandwidth / 50
        self.do_include_transverse_envelope = do_include_transverse_envelope

        # ======================== SSD parameters ========================= #
        # Only support single FM for now
        # the amplitude of phase along each direction
        self.ssd_phase_modulation_amplitude = ssd_phase_modulation_amplitude
        # number of color cycles
        self.ssd_number_color_cycles = ssd_number_color_cycles
        # bandwidth distributed with respect to the two transverse direction
        self.ssd_transverse_bandwidth_distribution = (
            ssd_transverse_bandwidth_distribution
        )

        # ================== Sanity checks on user inputs ===================== #
        self.supported_bandwidth = "FM SSD", "GP RPM SSD", "GP ISI"
        assert (
            temporal_smoothing_type.upper() in self.supported_bandwidth
        ), "Only support one of the following: " + ", ".join(self.supported_bandwidth)
        assert relative_laser_bandwidth > 0, "laser_bandwidth must be greater than 0"
        for q in (
            n_beamlets,
        ):
            assert np.size(q) == 2, "has to be a size 2 array"
        if "SSD" in self.temporal_smoothing_type.upper():
            assert ssd_number_color_cycles is not None, "must supply `ssd_number_color_cycles` to use SSD"
            assert ssd_transverse_bandwidth_distribution is not None, "must supply `ssd_transverse_bandwidth_distribution` to use SSD"
            assert ssd_phase_modulation_amplitude is not None, "must supply `ssd_phase_modulation_amplitude` to use SSD"
            for q in (
                ssd_number_color_cycles,
                ssd_transverse_bandwidth_distribution,
                ssd_phase_modulation_amplitude,
            ):
                assert np.size(q) == 2, "has to be a size 2 array"
                assert q[0] > 0 or q[1] > 0, "cannot be all zeros"

    def evaluate(self, x, y, t):
        """
        Return the envelope field of the laser.

        Parameters
        ----------
        x, y, t: ndarrays of floats
            Define points on which to evaluate the envelope
            These arrays need to all have the same shape.

        Returns
        -------
        envelope: ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y, t
        """
        # ======================== General parameters ==================== #
        t_norm = t[0, 0, :] * c / self.wavelength
        t_max = t_norm[-1]

        # # ================== Calculate auxiliary variables ================== #
        if "SSD" in self.temporal_smoothing_type.upper():
            phase_plate = np.random.uniform(
                -np.pi, np.pi, size=self.n_beamlets[0] * self.n_beamlets[1]
            ).reshape(self.n_beamlets)
        elif "ISI" in self.temporal_smoothing_type.upper():
            phase_plate = np.zeros(self.n_beamlets)  # ISI does not require phase plates
        else:
            raise NotImplementedError

        x_lens_list = np.linspace(
            -0.5 * (self.n_beamlets[0] - 1),
            0.5 * (self.n_beamlets[0] - 1),
            num=self.n_beamlets[0],
        )
        y_lens_list = np.linspace(
            -0.5 * (self.n_beamlets[1] - 1),
            0.5 * (self.n_beamlets[1] - 1),
            num=self.n_beamlets[1],
        )
        Y_lens_matrix, X_lens_matrix = np.meshgrid(y_lens_list, x_lens_list)
        Y_lens_index_matrix, X_lens_index_matrix = np.meshgrid(
            np.arange(self.n_beamlets[1], dtype=float),
            np.arange(self.n_beamlets[0], dtype=float),
        )
        phase_plate_phase_modulation = np.random.standard_normal(2) * np.pi
        stochastic_process_time = np.arange(0, t_max + self.dt_update, self.dt_update)

        if "SSD" in self.temporal_smoothing_type.upper():
            ssd_normalization = np.sqrt(
                self.ssd_transverse_bandwidth_distribution[0] ** 2
                + self.ssd_transverse_bandwidth_distribution[1] ** 2
            )
            ssd_frac = [
                self.ssd_transverse_bandwidth_distribution[0] / ssd_normalization,
                self.ssd_transverse_bandwidth_distribution[1] / ssd_normalization,
            ]
            ssd_phase_modulation_frequency = [
                self.laser_bandwidth * sf * 0.5 / pma
                for sf, pma in zip(ssd_frac, self.ssd_phase_modulation_amplitude)
            ]
            ssd_time_delay = (
                self.ssd_number_color_cycles[0] / ssd_phase_modulation_frequency[0]
                if ssd_phase_modulation_frequency[0] > 0
                else 0,
                self.ssd_number_color_cycles[1] / ssd_phase_modulation_frequency[1]
                if ssd_phase_modulation_frequency[1] > 0
                else 0,
            )

        # ======================= Initialization ========================= #
        def gen_gaussian_time_series(t_num, fwhm, rms_mean):
            """Generate a discrete time series that has gaussian power spectrum.

            :param t_num: number of grid points in time
            :param fwhm: full width half maximum of the power spectrum
            :param rms_mean: root-mean-square average of the spectrum
            :return: a time series array of complex numbers with shape [t_num]
            """
            if fwhm == 0.0:
                return np.zeros((2, t_num))
            else:
                omega = np.fft.fftshift(np.fft.fftfreq(t_num, d=self.dt_update))
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

        def init_gaussian_time_series():
            if "SSD" in self.temporal_smoothing_type.upper():
                pm_phase0 = gen_gaussian_time_series(
                    stochastic_process_time.size + int(np.sum(ssd_time_delay) / self.dt_update) + 2,
                    2 * np.pi * ssd_phase_modulation_frequency[0],
                    self.ssd_phase_modulation_amplitude[0],
                )
                pm_phase1 = gen_gaussian_time_series(
                    stochastic_process_time.size + int(np.sum(ssd_time_delay) / self.dt_update) + 2,
                    2 * np.pi * ssd_phase_modulation_frequency[1],
                    self.ssd_phase_modulation_amplitude[1],
                )
                time_interp = np.arange(
                    start=0,
                    stop=stochastic_process_time[-1] + np.sum(ssd_time_delay) + 3 * self.dt_update,
                    step=self.dt_update,
                )[: pm_phase0.size]
                return (
                    time_interp,
                    [
                        (np.real(pm_phase0) + np.imag(pm_phase0)) / np.sqrt(2),
                        (np.real(pm_phase1) + np.imag(pm_phase1)) / np.sqrt(2),
                    ],
                )
            elif "ISI" in self.temporal_smoothing_type.upper():
                complex_amp = np.stack(
                    [
                        np.stack(
                            [
                                gen_gaussian_time_series(
                                    stochastic_process_time.size,
                                    2 * self.laser_bandwidth,
                                    1,
                                )
                                for _i in range(self.n_beamlets[1])
                            ]
                        )
                        for _j in range(self.n_beamlets[0])
                    ]
                )
                return stochastic_process_time, complex_amp

        if "GP" in self.temporal_smoothing_type.upper():
            time_ext, timeSeries = init_gaussian_time_series()
        else:
            time_ext, timeSeries = stochastic_process_time, None

        def beamlets_complex_amplitude(t_now, temporal_smoothing_type="FM SSD"):
            if temporal_smoothing_type.upper() == "FM SSD":
                phase_t = self.ssd_phase_modulation_amplitude[0] * np.sin(
                    phase_plate_phase_modulation[0]
                    + 2 * np.pi
                    * ssd_phase_modulation_frequency[0]
                    * (t_now - X_lens_matrix * ssd_time_delay[0] / self.n_beamlets[0])
                ) + self.ssd_phase_modulation_amplitude[1] * np.sin(
                    phase_plate_phase_modulation[1]
                    + 2 * np.pi
                    * ssd_phase_modulation_frequency[1]
                    * (t_now - Y_lens_matrix * ssd_time_delay[1] / self.n_beamlets[1])
                )
                return np.exp(1j * phase_t)
            elif temporal_smoothing_type.upper() == "GP RPM SSD":
                phase_t = np.interp(
                    t_now + X_lens_index_matrix * ssd_time_delay[0] / self.n_beamlets[0],
                    time_ext,
                    timeSeries[0],
                ) + np.interp(
                    t_now + Y_lens_index_matrix * ssd_time_delay[1] / self.n_beamlets[1],
                    time_ext,
                    timeSeries[1],
                )
                return np.exp(1j * phase_t)
            elif temporal_smoothing_type.upper() == "GP ISI":
                return timeSeries[:, :, int(round(t_now / self.dt_update))]
            else:
                raise NotImplementedError

        exp_phase_plate = np.exp(1j * phase_plate)
        lambda_fnum = self.wavelength * self.focal_length / self.beam_aperture
        X_focus_matrix = x[:, :, 0] / lambda_fnum[0]
        Y_focus_matrix = y[:, :, 0] / lambda_fnum[1]
        x_focus_list = X_focus_matrix[:, 0]
        y_focus_list = Y_focus_matrix[0, :]
        x_phase_matrix = np.exp(
            -2 * np.pi * 1j
            / self.n_beamlets[0]
            * np.einsum("i,j", x_lens_list, x_focus_list)
        )
        y_phase_matrix = np.exp(
            -2 * np.pi * 1j
            / self.n_beamlets[1]
            * np.einsum("i,j", y_lens_list, y_focus_list)
        )

        def generate_speckle_pattern(tnow):
            bca = beamlets_complex_amplitude(
                tnow, temporal_smoothing_type=self.temporal_smoothing_type
            )
            speckle_amp = np.einsum(
                "jk,jl->kl",
                np.einsum("ij,ik->jk", bca * exp_phase_plate, x_phase_matrix),
                y_phase_matrix,
            )
            if self.do_include_transverse_envelope:
                speckle_amp = (
                    np.sinc(X_focus_matrix / self.n_beamlets[0])
                    * np.sinc(Y_focus_matrix / self.n_beamlets[1])
                    * speckle_amp
                )
            return speckle_amp

        envelope = np.zeros(x.shape, dtype=complex)
        for i, t_i in enumerate(t_norm):
            envelope[:, :, i] = generate_speckle_pattern(t_i)
        return envelope
