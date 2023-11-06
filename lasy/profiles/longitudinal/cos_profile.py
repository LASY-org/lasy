import numpy as np

from .longitudinal_profile import LongitudinalProfile


class CosLongitudinalProfile(LongitudinalProfile):
    r"""
    Derived class for the analytic longitudinal truncated cosinus profile  profile of a laser pulse.

    More precisely, the longitudinal envelope
    (to be used in the :class:CombinedLongitudinalTransverseProfile class)
    corresponds to:

    .. math::

        \mathcal{L}(t) = \cos\left({ \frac{\pi}{2} \frac{t-t_peak}{tau_fwhm} }\right)
            \theta( \frac{t-t_peak}{tau_fwhm} + 1 ) \theta( 1 - \frac{t-t_peak}{tau_fwhm} )
            \exp\left({ + i\omega_0t_{peak} }\right)

    Parameters
    ----------
    tau_fwhm : float (in second)
        The Full-Width-Half-Maximum duration of the intensity distribution of the pulse.

    t_peak : float (in second)
        The time at which the laser envelope reaches its maximum amplitude,
        i.e. :math:`t_{peak}` in the above formula.

    cep_phase : float (in radian), optional
        The Carrier Enveloppe Phase (CEP) :math:`\phi_{cep}`
        (i.e. the phase of the laser oscillation, at the time where the
        laser envelope is maximum)
    """

    def __init__(self, wavelength, tau, t_peak, cep_phase=0):
        super().__init__(wavelength)
        self.tau = tau
        self.t_peak = t_peak
        self.cep_phase = cep_phase

    def evaluate(self, t):
        """
        Return the longitudinal envelope.

        Parameters
        ----------
        t: ndarrays of floats
            Define points on which to evaluate the envelope

        Returns
        -------
        envelope: ndarray of complex numbers
            Contains the value of the longitudinal envelope at the
            specified points. This array has the same shape as the array t.
        """

        tn = (t - t_peak)/self.tau

        envelope = np.cos(0.5*np.pi*tn)*np.theta(tn + 1)*np.theta(1 - tn)*np.exp(
            + 1.0j * (self.cep_phase + self.omega0 * self.t_peak))

        return envelope
