from scipy.constants import c

from lasy.backend import xp

from .optical_element import OpticalElement


class Axicon(OpticalElement):
    r"""
    Class for an axicon.

    This object can represent a reflective, refractive or diffractive axicon,
    where the chromatic effects of the refractive axicon are assumed to be negligible.

    For a reflective or refractive axicon, the amplitude multiplier corresponds to:

    .. math::

        T(\boldsymbol{x}_\perp,\omega) = \exp(-i (\omega/c) \sqrt{x^2+y^2} \tan(\gamma/2))

    where :math:`\boldsymbol{x}_\perp` is the transverse coordinate (orthogonal
    to the propagation direction). The other parameters in this formula
    are defined below.

    For the diffractive axicon, the amplitude multiplier corresponds to:

    .. math::
        T(\boldsymbol{x}_\perp,\omega) = \exp\!\left(-i\,\frac{\omega}{\omega_0}\,
        \phi_0(\boldsymbol{x}_\perp)\right)

    A diffractive axicon is a thin phase element with a fixed physical groove depth.
    At the design frequency :math:`\omega_0` (equivalently, a design wavelength), the
    groove depth is chosen so that the phase matches the ideal (continuous) axicon
    phase, wrapped into :math:`(-\pi,\pi]` since only phase modulo :math:`2\pi` can be
    physically imprinted.

    The sawtooth (blazed-grating) phase profile of the physical element is:

    .. math::
        \phi_0(\boldsymbol{x}_\perp) = \operatorname{wrap}_{2\pi}\!\left[
        2\,(\omega_0/c)\,\sqrt{x^2+y^2}\,\tan(\gamma/2)\right]

    where :math:`\operatorname{wrap}_{2\pi}[\phi] = \phi - 2\pi\left\lfloor
    \phi/(2\pi) + 1/2\right\rfloor`, and has a constant spatial period
    :math:`\Lambda = \pi c/(\omega_0 \tan(\gamma/2))`.
    :math:`\boldsymbol{x}_\perp` is the transverse coordinate
    (orthogonal to the propagation direction). The other parameters in this
    formula are defined below.

    Parameters
    ----------
    gamma : float (in radians)
        The angle that the outcoming rays (coming from the axicon) would make
        with the optical axis, if the incoming rays (impinging on the axicon)
        are parallel to the optical axis.
    axicon_type : str (optional, default=``R``)
        The axicon type to use, either ``R`` or ``D``
    lambda0 : float (in m, optional, default=0.8e-6)
        If using axicon_type=``D``, the design (angular) wavelength for which
        the diffractive axicon's physical groove profile was fabricated should
        be specified.
    """

    def __init__(self, gamma, axicon_type="R", lambda0=0.8e-6):
        self.gamma = gamma
        self.axicon_type = axicon_type
        self.omega0 = 2.0 * xp.pi * c / (lambda0)
        assert self.axicon_type in ("R", "D"), (
            f"axicon type must be 'R' or 'D', got {self.axicon_type!r}"
        )

    def amplitude_multiplier(self, x, y, omega):
        """
        Return the amplitude multiplier.

        Parameters
        ----------
        x, y, omega : ndarrays of floats
            Define points on which to evaluate the multiplier.
            These arrays need to all have the same shape.

        Returns
        -------
        multiplier : ndarray of complex numbers
            Contains the value of the multiplier at the specified points.
            This array has the same shape as the array omega.
        """
        if self.axicon_type == "R":
            return xp.exp(
                -2j * (omega / c) * xp.sqrt(x**2 + y**2) * xp.tan(0.5 * self.gamma)
            )
        elif self.axicon_type == "D":
            unwrapped = (
                2 * (self.omega0 / c) * xp.sqrt(x**2 + y**2) * xp.tan(0.5 * self.gamma)
            )
            wrapped = xp.mod(unwrapped + xp.pi, 2 * xp.pi) - xp.pi
            return xp.exp(-1j * (omega / self.omega0) * wrapped)
