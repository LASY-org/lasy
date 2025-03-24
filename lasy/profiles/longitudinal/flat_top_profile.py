import numpy as np

from .longitudinal_profile import LongitudinalProfile


class FlatTopLongitudinalProfile(LongitudinalProfile):
    r"""
    Derived class for the analytic longitudinal flat-top profile of a laser pulse,
    i.e., a longitudinal profile that rises smoothly from zero to one (either linearly or as a squared cosine),
    stays constant for some time and then goes down smoothly to zero.

    More precisely, the longitudinal envelope
    (to be used in the :class:`.CombinedLongitudinalTransverseProfile` class)
    corresponds to:

    .. math::

        \mathcal{L}(t) = \left({
                                \mathcal{R}(t, t_1, t_2)\theta(t - t_1)\theta(t_2 - t) +
                                \theta(t - t_2)\theta(t_3 - t) +
                                \mathcal{R}(t, t_4, t_3)\theta(t - t_3)\theta(t_4 - t)
                         }\right)
                         \exp\left({ + i \left({ \phi_{cep} + \omega_0 \frac{t_3 + t_2}{2} }\right) }\right)

    Where:

    :math:`t_1 = t_{start}`

    :math:`t_2 = t_{start} + t_{rise}`

    :math:`t_3 = t_{start} + t_{rise} + t_{flat}`

    :math:`t_4 = t_{start} + t_{rise} + t_{flat} + t_{down}`

    :math:`\mathcal{R}(t, t_1, t_2) = (t-t_1)/(t_2 - t_1)` if 'linear' rise type is selected
    or
    :math:`\mathcal{R}(t, t_1, t_2) = \cos^2(\pi/2 (t-t_1)/(t_1-t_2))` if 'cos2' rise type is selected

    Parameters
    ----------
    wavelength : float (in meter)
        The main laser wavelength :math:`\lambda_0` of the laser.

    t_start : float (in seconds)
        The starting time of the pulse,
        i.e. :math:`\t_{start}` in the above formulae.

    t_rise : float (in seconds)
        The rise time of the pulse envelope,
        i.e. :math:`\t_{rise}` in the above formulae.

    t_flat : float (in seconds)
        The duration of the flat part of the pulse envelope,
        i.e. :math:`t_{flat}` in the above formula.

    t_down : float (in seconds)
        The duration of the decreasing part of the pulse envelope,
        i.e. :math:`t_{down}` in the above formula.

    cep_phase : float (in radian), optional
        The Carrier Enveloppe Phase (CEP)
        (i.e. the phase of the laser oscillation, at the time where the
        laser envelope is maximum,  :math:`\phi_{cep}` in the above formula).

    rise_type : string, optional
        If equal to `linear`, :math:`\mathcal{R} is a linear function.
        If equal to `cos2`, :math:`\mathcal{R} is a math:`\cos^2` function.
    """

    def __init__(
        self,
        wavelength,
        t_start,
        t_rise,
        t_flat,
        t_down,
        cep_phase=0,
        rise_type="linear",
    ):
        super().__init__(wavelength)
        self.t_start = t_start
        self.t_rise = t_rise
        self.t_flat = t_flat
        self.t_down = t_down
        self.cep_phase = cep_phase
        self.rise_type = rise_type

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
        t1 = self.t_start
        t2 = t1 + self.t_rise
        t3 = t2 + self.t_flat
        t4 = t3 + self.t_down
        print(t1, t2, t3, t4)
        tcep = 0.5 * (t3 + t2)

        if self.rise_type == "linear":
            envelope = (
                (t >= t1) * (t < t2) * (t - t1) / (t2 - t1)
                + (t >= t2) * (t < t3)
                + (t >= t3) * (t < t4) * (t - t4) / (t3 - t4)
            ) * np.exp(+1.0j * (self.cep_phase + self.omega0 * tcep))
            return envelope
        elif self.rise_type == "cos2":
            envelope = (
                (t >= t1) * (t < t2) * np.cos(0.5 * np.pi * (t - t2) / (t2 - t1)) ** 2
                + (t >= t2) * (t < t3)
                + (t >= t3) * (t < t4) * np.cos(0.5 * np.pi * (t - t3) / (t3 - t4)) ** 2
            ) * np.exp(+1.0j * (self.cep_phase + self.omega0 * tcep))
            return envelope
        else:
            raise Exception("rise type must be either 'linear' or 'cos2'")

        return envelope
