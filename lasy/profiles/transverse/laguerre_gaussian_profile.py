import numpy as np
from scipy.special import genlaguerre

from .transverse_profile import TransverseProfile


class LaguerreGaussianTransverseProfile(TransverseProfile):
    """
    Derived class for an analytic profile of a high-order Gaussian
    laser pulse expressed in the Laguerre-Gaussian formalism.
    """

    def __init__(self, w0, p, m):
        """
        Defines a Laguerre-Gaussian transverse envelope

        More precisely, the transverse envelope
        (to be used in the :class:CombinedLongitudinalTransverseLaser class)
        corresponds to:

        .. math::

            \\mathcal{T}(x, y) = r^{|m|}e^{-im\\theta} \\,
            L_p^{|m|}\\left( \\frac{2 r^2 }{w_0^2}\\right )\\,
            \\exp\\left( -\\frac{r^2}{w_0^2} \\right)

        where :math:`x = r \\cos{\\theta}`,
        :math:`y = r \\sin{\\theta}`, :math:`L_p^{|m|}` is the
        Generalised Laguerre polynomial of radial order :math:`p` and
        azimuthal order :math:`|m|`

        Parameters
        ----------
        w0: float (in meter)
            The waist of the laser pulse, i.e. :math:`w_0` in the above formula.
        p: int (dimensionless)
            The radial order of Generalized Laguerre polynomial
        m: int (dimensionless)
            Defines the phase rotation, i.e. :math:`m` in the above formula.
        """
        super().__init__()
        self.w0 = w0
        self.p = p
        self.m = m

    def evaluate(self, x, y):
        """
        Returns the transverse envelope.

        Parameters
        ----------
        x, y: ndarrays of floats
            Define points on which to evaluate the envelope
            These arrays need to all have the same shape.

        Returns
        -------
        envelope: ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y
        """
        # complex_position corresponds to r e^{+/-i\theta}
        if self.m > 0:
            complex_position = x - 1j * y
        else:
            complex_position = x + 1j * y
        radius = abs(complex_position)
        scaled_rad_squared = (radius**2) / self.w0**2
        envelope = (
            complex_position ** abs(self.m)
            * genlaguerre(self.p, abs(self.m))(2 * scaled_rad_squared)
            * np.exp(-scaled_rad_squared)
        )

        return envelope

    def __add__(self, other):
        """Overload the + operations for laser profiles."""
        return SummedTransverseProfile(self, other)

    def __mul__(self, other):
        """Overload the * operations for laser profiles."""
        return ScaledTransverseProfile(self, other)

    def __rmul__(self, other):
        """Overload the * operations for laser profiles."""
        return ScaledTransverseProfile(self, other)


class SummedTransverseProfile(LaguerreGaussianTransverseProfile):
    """Class for a transverse profile that is the sum of several profiles."""

    def __init__(self, *profiles):
        """
        Initialize the transverse profile.

        Parameters
        ----------
        *profiles: list of LaguerreGaussianTransverseProfile objects
            The profiles to be summed.
        """
        # Store the input profiles
        self.profiles = profiles
        # Get the waist values from each profile
        self.w0 = [p.w0 for p in self.profiles]
        # Check if all waist values are equal
        if all([w0 == self.w0[0] for w0 in self.w0]):
            # If all waist values are equal, store the first one as the waist
            self.w0 = self.w0[0]
        else:
            # If all waist values are not equal, raise a ValueError
            raise ValueError("All profiles must have the same waist.")
        # Unset p and m of the input profiles
        self.p = None
        self.m = None
        # Check if all profiles are instances of LaguerreGaussianTransverseProfile
        if not all(
            [isinstance(p, LaguerreGaussianTransverseProfile) for p in self.profiles]
        ):
            # If not, raise a ValueError
            raise ValueError("All profiles must be LaguerreGaussian objects.")

    def _evaluate(self, x, y):
        """Return the sum of the profiles."""
        return sum([p.evaluate(x, y) for p in self.profiles])


class ScaledTransverseProfile(LaguerreGaussianTransverseProfile):
    """Class for a transverse profile that is scaled by a factor."""

    def __init__(self, profile, factor):
        """
        Initialize the transverse profile.

        Parameters
        ----------
        profile: LaguerreGaussianTransverseProfile object
            The profile to be scaled.
        factor: float
            The scaling factor.
        """
        self.w0 = profile.w0
        self.p = profile.p
        self.m = profile.m
        self.profile = profile
        self.factor = factor
        if not isinstance(self.factor, (int, float)):
            raise ValueError("The scaling factor must be a float.")

    def _evaluate(self, x, y):
        """Return the scaled profile."""
        return self.factor * self.profile.evaluate(x, y)
