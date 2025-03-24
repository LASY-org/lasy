import numpy as np
from scipy.constants import c

from .profile import Profile


def gen_gaussian_time_series(t_num, dt, fwhm, rms_mean):
    """Generate a discrete time series that has gaussian power spectrum.

    Parameters
    ----------
    t_num : int
        The number of grid points in time

    fwhm : float
        The full width half maximum of the power spectrum

    rms_mean : scalar
        The root-mean-square average of the spectrum

    Returns
    -------
    temporal_amplitude : 1darray
        A time series array of complex numbers with shape [t_num]
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


class SpeckleProfile(Profile):
    r"""
    Class for the profile of a speckled laser pulse.

    Speckled lasers are used to mitigate laser-plasma interactions in fusion and ion acceleration contexts.
    More on the subject can be found in chapter 9 of `P. Michel, Introduction to Laser-Plasma Interactions <https://link.springer.com/book/10.1007/978-3-031-23424-8>`__.
    A speckled laser beam is a laser that is deliberately divided transversely into :math:`N_{bx}\times N_{by}` beamlets in the near-field.
    The phase plate provides a different phase to each beamlet, with index :math:`ml`, which then propagate to the far field and combine incoherently.

    The electric field in the focal plane, as a function of time :math:`t` and the coordinates
    :math:`\boldsymbol{x}_\perp=(x,y)` transverse to the direction of propagation, is:

    .. math::

        \begin{aligned}
        E_u(\boldsymbol{x}_\perp,t) &= Re\left[ E_0
        {\rm sinc}\left(\frac{\pi x}{\Delta x}\right)
        {\rm sinc}\left(\frac{\pi y}{\Delta y}\right)\times p_u
        \right.
        \\
        & \times\sum_{m,l=1}^{N_{bx}, N_{by}} A_{ml}
        \exp\left(i\boldsymbol{k}_{\perp ml}\cdot\boldsymbol{x}_\perp
        + i\phi_{{\rm RPP/CPP},ml}+i\psi_{{\rm SSD},ml}(t)\right)
        \Bigg]
        \end{aligned}

    where :math:`u` is either :math:`x` or :math:`y`, :math:`p_u` is
    the polarization vector, and :math:`Re` represent the real part [Michel, Eqns. 9.11, 87, 94].
    Several quantities are computed internally to the code depending on the
    method of smoothing chosen, including the beamlet amplitude :math:`A_{ml}`,
    the beamlet wavenumber :math:`k_{\perp ml}`,
    the relative phase contribution :math:`\phi_{{\rm RPP/CPP},ml}` of beamlet :math:`ml` induced by the phase plate,
    and the phase contribution to beamlet :math:`ml`, :math:`\psi_{{\rm SSD},ml}(t)`, from the temporal smoothing.
    The beam widths are :math:`\Delta x=\frac{\lambda_0fN_{bx}}{D_{x}}`,
    :math:`\Delta y=\frac{\lambda_0fN_{by}}{D_{y}}`.
    The other parameters in these formulas are defined below.

    This profile admits several options for calculating the amplitudes and phases of the beamlets:

    * Random phase plates (RPP), ``'RPP'``:
        Here the phase plate contribution to the phase of beamlet :math:`ml` is :math:`\phi_{{\rm RPP},ml}\in\{0,\pi\}` (drawn randomly for each :math:`m,l`), :math:`\psi_{{\rm SSD},ml}(t)=0`, and :math:`A_{ml}=1`
    * Continuous phase plates (CPP), ``'CPP'``:
        :math:`\phi_{{\rm CPP},ml}\in[0,\pi]` (drawn randomly with uniform probability between :math:`0` and :math:`\pi`, for each :math:`m,l`), :math:`\psi_{{\rm SSD},ml}(t)=0`, and :math:`A_{ml}=1`
    * CPP + Smoothing by spectral dispersion (SSD), ``'FM SSD'``:
        :math:`\phi_{{\rm CPP},ml}\in[0,\pi]` (drawn randomly with uniform probability between :math:`0` and :math:`\pi`, for each :math:`m,l`),

        .. math::

            \begin{aligned}
            \psi_{{\rm SSD},ml}(t)&=\delta_{x} \sin\left(\omega_{x} t + 2\pi\frac{mN_{cc,x}}{N_{bx}}\right)\\
            &+\delta_{y} \sin\left(\omega_{y} t + 2\pi\frac{lN_{cc,y}}{N_{by}}\right),
            \end{aligned}

        and :math:`A_{ml}=1`.  The modulation frequencies :math:`\omega_x,\omega_y` are determined by the
        laser bandwidth and modulation amplitudes by the relation

        .. math::

            \omega_x = \frac{\Delta_\nu r }{2\delta_x}

        where :math:`\Delta_\nu` is the relative bandwidth of the laser pulse and :math:`r` is an additional rotation factor
        supplied by the user that determines how much of the modulation is in x and how much is in y. [Michel, Eqn. 9.69]

    * Gaussian Process Randomly phase-modulated SSD, ``'GP RPM SSD'``:
        CPP + a generalization of SSD that has temporal stochastic variation in the beamlet phases; that is, :math:`\phi_{{\rm CPP},ml}\in[0,\pi]`, :math:`\psi_{{\rm SSD},ml}(t)` is sampled from a Gaussian stochastic process, and :math:`A_{ml}=1`
    * Induced spatial incoherence (ISI), ``'GP ISI'``:
        a smoothing technique with temporal stochastic variation in the beamlet phases and amplitudes; that is, :math:`\phi_{{\rm CPP},ml}=0`, :math:`\psi_{{\rm SSD},ml}(t)=0`, and the :math:`A_{ml}` are complex amplitudes sampled from a Gaussian stochastic process to simulate the random phase differences and amplitudes the beamlets pick up when passing through the ISI echelons

    This is an adapation of work by `Han Wen <https://github.com/Wen-Han/LasersSmoothing2d>`__ to LASY.


    Notes
    -----
    This assumes a flat-top rectangular laser and so a rectangular arrangement of beamlets in the near-field.

    Parameters
    ----------
    wavelength : float (in meters)
        The main laser wavelength :math:`\lambda_0` of the laser, which
        defines :math:`\omega_0` in the above formula, according to
        :math:`\omega_0 = 2\pi c/\lambda_0`.

    pol : list of 2 complex numbers (dimensionless)
        Polarization vector. It corresponds to :math:`p_u` in the above
        formula ; :math:`p_x` is the first element of the list and
        :math:`p_y` is the second element of the list. Using complex
        numbers enables elliptical polarizations.

    laser_energy : float (in Joules)
        The total energy of the laser pulse. The amplitude of the laser
        field (:math:`E_0` in the above formula) is automatically
        calculated so that the pulse has the prescribed energy.

    focal_length : float (in meters)
        Focal length of lens :math:`f` just after the RPP/CPP.

    beam_aperture : list of 2 floats (in meters)
        Widths :math:`D_x,D_y` of the rectangular beam in the near-field, i.e., size of the illuminated region of the RPP/CPP.

    n_beamlets : list of 2 integers
        Number of RPP/CPP elements :math:`N_{bx},N_{by}` in each direction, in the near field.

    temporal_smoothing_type : string
        Which method for beamlet production and evolution is used.
        Can be ``'RPP'``, ``'CPP'``, ``'FM SSD'``, ``'GP RPM SSD'``, or ``'GP ISI'``.
        (See the above bullet list for an explanation of each of these methods.)

    relative_laser_bandwidth : float
        Bandwidth :math:`\Delta_\nu` of the laser pulse, relative to central frequency.
        Only used if ``temporal_smoothing_type`` is ``'FM SSD'``, ``'GP RPM SSD'`` or ``'GP ISI'``.

    ssd_phase_modulation_amplitude :list of 2 floats
        Amplitudes :math:`\delta_{x},\delta_{y}` of phase modulation in each transverse direction.
        Only used if ``temporal_smoothing_type`` is ``'FM SSD'`` or ``'GP RPM SSD'``.

    ssd_number_color_cycles : list of 2 floats
        Number of color cycles :math:`N_{cc,x},N_{cc,y}` of SSD spectrum to include in modulation
        Only used if ``temporal_smoothing_type`` is ``'FM SSD'`` or ``'GP RPM SSD'``.

    ssd_transverse_bandwidth_distribution: list of 2 floats
        Determines how much SSD is distributed in the :math:`x` and :math:`y` directions.
        if `ssd_transverse_bandwidth_distribution=[a,b]`, then the SSD frequency modulation is :math:`a/\sqrt{a^2+b^2}` in :math:`x` and :math:`b/\sqrt{a^2+b^2}` in :math:`y`.
        Only used if ``temporal_smoothing_type`` is ``'FM SSD'`` or ``'GP RPM SSD'``.

    do_include_transverse_envelope : boolean, (optional, default False)
        Whether to include the transverse sinc envelope or not.
        I.e. whether it is assumed to be close enough to the laser axis to neglect the transverse field decay.
    """

    supported_smoothing = "RPP", "CPP", "FM SSD", "GP RPM SSD", "GP ISI"

    def __init__(
        self,
        wavelength,
        pol,
        laser_energy,
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
        self.laser_energy = laser_energy
        self.focal_length = focal_length
        self.beam_aperture = np.array(beam_aperture, dtype="float")
        self.n_beamlets = np.array(n_beamlets, dtype="int")
        self.temporal_smoothing_type = temporal_smoothing_type
        self.laser_bandwidth = relative_laser_bandwidth

        # time interval to update the speckle pattern, roughly update 50 times every bandwidth cycle
        self.dt_update = 1 / self.laser_bandwidth / 50
        self.do_include_transverse_envelope = do_include_transverse_envelope

        self.x_lens_list = np.linspace(
            -0.5 * (self.n_beamlets[0] - 1),
            0.5 * (self.n_beamlets[0] - 1),
            num=self.n_beamlets[0],
        )
        self.y_lens_list = np.linspace(
            -0.5 * (self.n_beamlets[1] - 1),
            0.5 * (self.n_beamlets[1] - 1),
            num=self.n_beamlets[1],
        )
        self.Y_lens_matrix, self.X_lens_matrix = np.meshgrid(
            self.y_lens_list, self.x_lens_list
        )
        self.Y_lens_index_matrix, self.X_lens_index_matrix = np.meshgrid(
            np.arange(self.n_beamlets[1], dtype=float),
            np.arange(self.n_beamlets[0], dtype=float),
        )

        if "SSD" in self.temporal_smoothing_type.upper():
            # Initialize SSD parameters
            # the amplitude of phase along each direction
            self.ssd_phase_modulation_amplitude = ssd_phase_modulation_amplitude
            # number of color cycles
            self.ssd_number_color_cycles = ssd_number_color_cycles
            # bandwidth distributed with respect to the two transverse direction
            self.ssd_transverse_bandwidth_distribution = (
                ssd_transverse_bandwidth_distribution
            )
            ssd_normalization = np.sqrt(
                self.ssd_transverse_bandwidth_distribution[0] ** 2
                + self.ssd_transverse_bandwidth_distribution[1] ** 2
            )
            ssd_frac = [
                self.ssd_transverse_bandwidth_distribution[0] / ssd_normalization,
                self.ssd_transverse_bandwidth_distribution[1] / ssd_normalization,
            ]
            self.ssd_phase_modulation_frequency = [
                self.laser_bandwidth * sf * 0.5 / pma
                for sf, pma in zip(ssd_frac, self.ssd_phase_modulation_amplitude)
            ]
            self.ssd_time_delay = (
                (
                    self.ssd_number_color_cycles[0]
                    / self.ssd_phase_modulation_frequency[0]
                    if self.ssd_phase_modulation_frequency[0] > 0
                    else 0
                ),
                (
                    self.ssd_number_color_cycles[1]
                    / self.ssd_phase_modulation_frequency[1]
                    if self.ssd_phase_modulation_frequency[1] > 0
                    else 0
                ),
            )

        # Check user input
        assert temporal_smoothing_type.upper() in SpeckleProfile.supported_smoothing, (
            "Only support one of the following: "
            + ", ".join(SpeckleProfile.supported_smoothing)
        )
        assert relative_laser_bandwidth > 0, "laser_bandwidth must be greater than 0"
        assert np.size(n_beamlets) == 2, "has to be a size 2 array"
        if "SSD" in self.temporal_smoothing_type.upper():
            assert ssd_number_color_cycles is not None, (
                "must supply `ssd_number_color_cycles` to use SSD"
            )
            assert ssd_transverse_bandwidth_distribution is not None, (
                "must supply `ssd_transverse_bandwidth_distribution` to use SSD"
            )
            assert ssd_phase_modulation_amplitude is not None, (
                "must supply `ssd_phase_modulation_amplitude` to use SSD"
            )
            for q in (
                ssd_number_color_cycles,
                ssd_transverse_bandwidth_distribution,
                ssd_phase_modulation_amplitude,
            ):
                assert np.size(q) == 2, "has to be a size 2 array"
                assert q[0] > 0 or q[1] > 0, "cannot be all zeros"

    def init_gaussian_time_series(
        self,
        series_time,
    ):
        r"""Initialize a time series sampled from a Gaussian process.

        At every time specified by the input `series_time`, calculate the random phase and/or amplitudes as determined by the smoothing type.

        * If the smoothing type is "SSD", then this function returns a time series with random phase offsets in x and y at each time.
            The phase offsets are real-valued and centered around the user supplied ``ssd_phase_modulation_amplitude``
            :math:`\delta_{x},\delta_{y}`, with distribution FWHM ``ssd_phase_modulation_frequency``.

        If the smoothing type is "ISI", this function returns a time series with complex numbers defining beamlet phase and amplitude.
            Each beamlet has a complex number whose magnitude is the amplitude of the beamlet and
            whose phase gives the beamlet phase offset.
            The complex numbers have mean 1 and FWHM given by twice the laser bandwidth

        Parameters
        ----------
        series_time : 1d array
            Array of times at which to sample from Gaussian process

        ssd_time_delay : scalar
            Only required for "SSD" type smoothing

        ssd_phase_modulation_frequency : scalar
            Only required for "SSD" type smoothing

        Returns
        -------
        array-like, either the supplied `series_time` if "ISI" smoothing or `series_time` with some padding at the end for "SSD" smoothing
        array-like, either with 2 (for "SSD" smoothing) or `n_beamlets[0] x n_beamlets[1]` ("ISI" smoothing) random numbers at every time
        """
        if "SSD" in self.temporal_smoothing_type.upper():
            pm_phase0 = gen_gaussian_time_series(
                series_time.size
                + int(np.sum(self.ssd_time_delay) / self.dt_update)
                + 2,
                self.dt_update,
                2 * np.pi * self.ssd_phase_modulation_frequency[0],
                self.ssd_phase_modulation_amplitude[0],
            )
            pm_phase1 = gen_gaussian_time_series(
                series_time.size
                + int(np.sum(self.ssd_time_delay) / self.dt_update)
                + 2,
                self.dt_update,
                2 * np.pi * self.ssd_phase_modulation_frequency[1],
                self.ssd_phase_modulation_amplitude[1],
            )
            time_interp = np.arange(
                start=0,
                stop=series_time[-1] + np.sum(self.ssd_time_delay) + 3 * self.dt_update,
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
            return series_time, complex_amp
        else:
            return series_time, None

    def beamlets_complex_amplitude(
        self, t_now, series_time, time_series, temporal_smoothing_type="FM SSD"
    ):
        """Calculate complex amplitude of the beamlets in the near-field, before propagating to the focal plane.

        If the temporal smoothing type is "RPP" or "CPP", this returns a matrix of ones, giving no modification to the amplitude.
        If the temporal smoothing type is "FM SSD", this returns the complex phases as calculated in, for example, Introduction to Laser-Plasma Interactions eqn. 9.87.
        If the temporal smoothing type is "GP RPM FM ", this returns complex phases modeled as random variables.
        If the temporal smoothing type is "ISI", this returns an array of random complex numbers that gives both amplitude and phase of the beamlets.

        Parameters
        ----------
        t_now : scalar
            Time at which to calculate the complex amplitude of the beamlets

        series_time : 1d array
            Array of times at which the stochastic process was sampled to generate the time series

        time_series : 1d array
            Array of random phase and/or amplitudes as determined by the smoothing type

        temporal_smoothing_type : string
            What type of temporal smoothing to perform.

        Returns
        -------
        array of complex numbers giving beamlet amplitude and phases in the near-field
        """
        if any(
            rpp_type in temporal_smoothing_type.upper() for rpp_type in ["RPP", "CPP"]
        ):
            return np.ones_like(self.X_lens_matrix)
        if temporal_smoothing_type.upper() == "FM SSD":
            phase_t = self.ssd_phase_modulation_amplitude[0] * np.sin(
                self.ssd_x_y_dephasing[0]
                + 2
                * np.pi
                * self.ssd_phase_modulation_frequency[0]
                * (
                    t_now
                    - self.X_lens_matrix * self.ssd_time_delay[0] / self.n_beamlets[0]
                )
            ) + self.ssd_phase_modulation_amplitude[1] * np.sin(
                self.ssd_x_y_dephasing[1]
                + 2
                * np.pi
                * self.ssd_phase_modulation_frequency[1]
                * (
                    t_now
                    - self.Y_lens_matrix * self.ssd_time_delay[1] / self.n_beamlets[1]
                )
            )
            return np.exp(1j * phase_t)
        elif temporal_smoothing_type.upper() == "GP RPM SSD":
            phase_t = np.interp(
                t_now
                + self.X_lens_index_matrix
                * self.ssd_time_delay[0]
                / self.n_beamlets[0],
                series_time,
                time_series[0],
            ) + np.interp(
                t_now
                + self.Y_lens_index_matrix
                * self.ssd_time_delay[1]
                / self.n_beamlets[1],
                series_time,
                time_series[1],
            )
            return np.exp(1j * phase_t)
        elif temporal_smoothing_type.upper() == "GP ISI":
            return time_series[:, :, int(round(t_now / self.dt_update))]
        else:
            raise NotImplementedError

    def generate_speckle_pattern(
        self, t_now, exp_phase_plate, x, y, series_time, time_series
    ):
        """Calculate the speckle pattern in the focal plane.

        Calculates the complex envelope defining the laser pulse in the focal plane at time `t=t_now`.
        This function first gets the beamlet complex amplitudes and phases with the function `beamlets_complex_amplitude`
        then propagates the the beamlets to the focal plane.

        Parameters
        ----------
        t_now : Scalar
            Time at which to calculate the speckle pattern

        exp_phase_plate : 2d array of complex numbers
            The RPP / CPP phase contributions to the beamlets

        x : 3d array
            x-positions in focal plane

        y : 3d array
            y-positions in focal plane

        series_time : 1d array
            Times at which the stochastic process was sampled to generate the time series

        time_series : array
            Random phase and/or amplitudes as determined by the smoothing type

        Returns
        -------
        speckle_amp : 2D array of complex numbers defining the laser envelope at focus at time `t_now`
        """
        lambda_fnum = self.lambda0 * self.focal_length / self.beam_aperture
        X_focus_matrix = x[:, :, 0] / lambda_fnum[0]
        Y_focus_matrix = y[:, :, 0] / lambda_fnum[1]
        x_focus_list = X_focus_matrix[:, 0]
        y_focus_list = Y_focus_matrix[0, :]
        x_phase_focus_matrix = np.exp(
            -2
            * np.pi
            * 1j
            / self.n_beamlets[0]
            * self.x_lens_list[:, np.newaxis]
            * x_focus_list[np.newaxis, :]
        )
        y_phase_focus_matrix = np.exp(
            -2
            * np.pi
            * 1j
            / self.n_beamlets[1]
            * self.y_lens_list[:, np.newaxis]
            * y_focus_list[np.newaxis, :]
        )

        bca = self.beamlets_complex_amplitude(
            t_now,
            series_time=series_time,
            time_series=time_series,
            temporal_smoothing_type=self.temporal_smoothing_type,
        )
        speckle_amp = np.einsum(
            "jk,jl->kl",
            np.einsum("ij,ik->jk", bca * exp_phase_plate, x_phase_focus_matrix),
            y_phase_focus_matrix,
        )
        if self.do_include_transverse_envelope:
            speckle_amp = (
                np.sinc(X_focus_matrix / self.n_beamlets[0])
                * np.sinc(Y_focus_matrix / self.n_beamlets[1])
                * speckle_amp
            )
        return speckle_amp

    def evaluate(self, x, y, t):
        """
        Return the envelope field of the laser.

        Parameters
        ----------
        x, y, t : ndarrays of floats
            Define points on which to evaluate the envelope
            These arrays need to all have the same shape.

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y, t
        """
        # General parameters
        t_norm = t[0, 0, :] * c / self.lambda0
        t_max = t_norm[-1]

        # Calculate auxiliary parameters
        if "RPP" == self.temporal_smoothing_type.upper():
            phase_plate = np.random.choice([0, np.pi], self.n_beamlets)
        elif any(
            cpp_smoothing_type in self.temporal_smoothing_type.upper()
            for cpp_smoothing_type in ["CPP", "SSD"]
        ):
            phase_plate = np.random.uniform(
                -np.pi, np.pi, size=self.n_beamlets[0] * self.n_beamlets[1]
            ).reshape(self.n_beamlets)
        elif "ISI" in self.temporal_smoothing_type.upper():
            phase_plate = np.zeros(self.n_beamlets)  # ISI does not require phase plates
        else:
            raise NotImplementedError
        exp_phase_plate = np.exp(1j * phase_plate)
        if self.temporal_smoothing_type.upper() == "FM SSD":
            self.ssd_x_y_dephasing = np.random.standard_normal(2) * np.pi

        series_time = np.arange(0, t_max + self.dt_update, self.dt_update)

        if "GP" in self.temporal_smoothing_type.upper():
            new_series_time, time_series = self.init_gaussian_time_series(series_time)
        else:
            new_series_time, time_series = series_time, None

        envelope = np.zeros(x.shape, dtype=complex)
        for i, t_i in enumerate(t_norm):
            envelope[:, :, i] = self.generate_speckle_pattern(
                t_i,
                exp_phase_plate=exp_phase_plate,
                x=x,
                y=y,
                series_time=new_series_time,
                time_series=time_series,
            )
        return envelope
