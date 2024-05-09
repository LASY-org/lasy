import numpy as np
from scipy.constants import c
from ..profile import Profile


class SpeckleProfile(Profile):
    r"""Profile of a speckled laser pulse.

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
        & \times\sum_{m,l=1}^{N_{bx}, N_{by}} A_{ml}(t)
        \exp\left(i\boldsymbol{k}_{\perp ml}\cdot\boldsymbol{x}_\perp
        + i\phi_{ml}(t)\right)
        \Bigg]
        \end{aligned}

    where :math:`u` is either :math:`x` or :math:`y`, :math:`p_u` is
    the polarization vector, and :math:`Re` represent the real part [Michel, Eqns. 9.11, 87, 94].
    Several quantities are computed internally to the code depending on the
    method of smoothing chosen, including the beamlet amplitude :math:`A_{ml}(t)`,
    the beamlet wavenumber :math:`k_{\perp ml}`,
    the relative phase contribution :math:`\phi_{ml}(t)` of beamlet :math:`ml` induced by the phase plate and temporal smoothing.
    The beam widths are :math:`\Delta x=\frac{\lambda_0fN_{bx}}{D_{x}}`,
    :math:`\Delta y=\frac{\lambda_0fN_{by}}{D_{y}}`.
    The other parameters in these formulas are defined below.

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

    do_include_transverse_envelope : boolean (optional, default: False)
        Whether to include the transverse sinc envelope or not.
        I.e. whether it is assumed to be close enough to the laser axis to neglect the transverse field decay.

    long_profile : Lasy Longitudinal laser object (optional, default: None).
        If this is not None, the longitudinal profile is applied individually to the beamlets in the near-field.

    Notes
    -----
    This is an adaptation of work by `Han Wen <https://github.com/Wen-Han/LasersSmoothing2d>`__ to LASY.

    This assumes a flat-top rectangular laser and so a rectangular arrangement of beamlets in the near-field.
    The longitudinal profile is currently applied to the beamlets
    individually in the near-field before they are propagated to the focal plane.
    """

    def __init__(
        self,
        wavelength,
        pol,
        laser_energy,
        focal_length,
        beam_aperture,
        n_beamlets,
        do_include_transverse_envelope,
        long_profile,
    ):
        super().__init__(wavelength, pol)
        self.laser_energy = laser_energy
        self.focal_length = focal_length
        self.beam_aperture = np.array(beam_aperture, dtype="float")
        self.n_beamlets = np.array(n_beamlets, dtype="int")
        self.do_include_transverse_envelope = do_include_transverse_envelope
        self.long_profile = long_profile

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
        return

    def beamlets_complex_amplitude(
        self,
        t_now,
    ):
        """Calculate complex amplitude of the beamlets in the near-field, before propagating to the focal plane.

        This function can be overwritten to define custom speckled laser objects.

        Parameters
        ----------
        t_now: float, time at which to evaluate complex amplitude

        Returns
        -------
        array of complex numbers giving beamlet amplitude and phases in the near-field
        """
        return np.ones_like(self.X_lens_matrix)

    def setup_for_evaluation(self, t_norm):
        """Create or update data used in evaluation."""
        pass

    def generate_speckle_pattern(self, t_now, x, y):
        """Calculate the speckle pattern in the focal plane.

        Calculates the complex envelope defining the laser pulse in the focal plane at time `t=t_now`.
        This function first gets the beamlet complex amplitudes and phases with the function `beamlets_complex_amplitude`
        then propagates the the beamlets to the focal plane.

        Parameters
        ----------
        t_now: float, time at which to calculate the speckle pattern
        exp_phase_plate: 2d array of complex numbers giving the RPP / CPP phase contributions to the beamlets
        x: 3d array of x-positions in focal plane
        y: 3d array of y-positions in focal plane
        series_time: 1d array of times at which the stochastic process was sampled to generate the time series
        time_series: array of random phase and/or amplitudes as determined by the smoothing type

        Returns
        -------
        speckle_amp: 2D array of complex numbers defining the laser envelope at focus at time `t_now`
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
        bca = self.beamlets_complex_amplitude(t_now)
        if self.long_profile is not None:
            # have to unnormalize t_now to evaluate in longitudinal profile
            bca = bca * self.long_profile.evaluate(t_now / c * self.lambda0)
        speckle_amp = np.einsum(
            "jk,jl->kl",
            np.einsum("ij,ik->jk", bca, x_phase_focus_matrix),
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
        x, y, t: ndarrays of floats
            Define points on which to evaluate the envelope
            These arrays need to all have the same shape.

        Returns
        -------
        envelope: ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y, t
        """
        # General parameters
        t_norm = t[0, 0, :] * c / self.lambda0
        self.setup_for_evaluation(t_norm)

        envelope = np.zeros(x.shape, dtype=complex)
        for i, t_i in enumerate(t_norm):
            envelope[:, :, i] = self.generate_speckle_pattern(
                t_i,
                x=x,
                y=y,
            )
        return envelope
