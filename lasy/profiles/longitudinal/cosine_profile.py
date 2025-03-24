import numpy as np

from .longitudinal_profile import LongitudinalProfile


class CosineLongitudinalProfile(LongitudinalProfile):
    r"""
    Class for the analytic longitudinal truncated cosine profile of a laser pulse.

    More precisely, the longitudinal envelope
    (to be used in the :class:`.CombinedLongitudinalTransverseProfile` class)
    corresponds to:

    .. math::

        \mathcal{L}(t) = \cos\left({ \frac{\pi}{2} \frac{t-t_{peak}}{\tau_{fwhm}} }\right)
            \theta\left({ \frac{t-t_{peak}}{\tau_{fwhm}} + 1 }\right)
            \theta\left({ 1 - \frac{t-t_{peak}}{\tau_{fwhm}}} \right)
            \exp\left({ + i (\phi_{cep} + \omega_0 t_{peak} ) }\right)


    Parameters
    ----------
    wavelength : float (in meter)
        The main laser wavelength :math:`\lambda_0` of the laser.

    tau_fwhm : float (in second)
        The Full-Width-Half-Maximum duration of the intensity distribution of the pulse,
        i.e. :math:`\tau_{fwhm}` in the above formula.

    t_peak : float (in second)
        The time at which the laser envelope reaches its maximum amplitude,
        i.e. :math:`t_{peak}` in the above formula.

    cep_phase : float (in radian), optional
        The Carrier Enveloppe Phase (CEP)
        (i.e. the phase of the laser oscillation, at the time where the
        laser envelope is maximum,  :math:`\phi_{cep}` in the above formula).
    """

    def __init__(self, wavelength, tau_fwhm, t_peak, cep_phase=0):
        super().__init__(wavelength)
        self.tau_fwhm = tau_fwhm
        self.t_peak = t_peak
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
        tn = (t - self.t_peak) / self.tau_fwhm

        envelope = (
            np.cos(0.5 * np.pi * tn)
            * (tn > -1)
            * (tn < 1)
            * np.exp(+1.0j * (self.cep_phase + self.omega0 * self.t_peak))
        )

        return envelope
