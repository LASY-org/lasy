import numpy as np

from .longitudinal_profile import LongitudinalProfile


class SuperGaussianLongitudinalProfile(LongitudinalProfile):
    r"""
    Class for the analytic profile of a longitudinally-super-Gaussian laser pulse.

    More precisely, the longitudinal envelope
    (to be used in the :class:`.CombinedLongitudinalTransverseProfile` class)
    corresponds to:

    .. math::

        \mathcal{L}(t) = \exp\left( -\left(\frac{(t-t_{peak})^2}{\tau^2}\right)^{n/2}
                        + i\omega_0t_{peak} \right)

    Parameters
    ----------
    tau : float (in second)
        The duration of the laser pulse, i.e. :math:`\tau` in the above
        formula. Note that :math:`\tau = \tau_{FWHM}/(\sqrt{2}\log(2)^{1/n})`,
        where :math:`\tau_{FWHM}` is the Full-Width-Half-Maximum duration
        of the intensity distribution of the pulse.

    t_peak : float (in second)
        The time at which the laser envelope reaches its maximum amplitude,
        i.e. :math:`t_{peak}` in the above formula.

    cep_phase : float (in radian), optional
        The Carrier Enveloppe Phase (CEP), i.e. :math:`\phi_{cep}`
        in the above formula (i.e. the phase of the laser
        oscillation, at the time where the laser envelope is maximum)

    n_order : float (in meter)
        The shape parameter of the super-gaussian function, i.e. :math:`n` in the above formula.
        If :math:`n=2` the super-Gaussian becomes a standard Gaussian function.
        If :math:`n=1` the super-Gaussian becomes a Laplace function.
    """

    def __init__(self, wavelength, tau, t_peak, n_order, cep_phase=0):
        super().__init__(wavelength)
        self.tau = tau
        self.t_peak = t_peak
        self.n_order = n_order
        self.cep_phase = cep_phase

    def evaluate(self, t):
        """
        Return the longitudinal envelope.

        Parameters
        ----------
        t : ndarrays of floats
            Define points on which to evaluate the envelope

        Returns
        -------
        envelope : ndarray of complex numbers
            Contains the value of the longitudinal envelope at the
            specified points. This array has the same shape as the array t.
        """
        envelope = np.exp(
            -np.power(((t - self.t_peak) ** 2) / self.tau**2, self.n_order / 2)
            + 1.0j * (self.cep_phase + self.omega0 * self.t_peak)
        )

        return envelope
