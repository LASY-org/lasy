import numpy as np
from scipy.special.orthogonal import hermite

from .transverse_profile import TransverseProfile


class HermiteGaussianTransverseProfile(TransverseProfile):
    """
    Derived class for an analytic profile of a high-order Gaussian
    laser pulse expressed in the Hermite-Gaussian formalism.
    """

    def __init__(self, w0, n_x, n_y):
        """
        Defines a Hermite-Gaussian transverse envelope

        More precisely, the transverse envelope
        (to be used in the :class:CombinedLongitudinalTransverseLaser class)
        corresponds to:

        .. math::
            \\mathcal{T}(x, y) =
            H_{n_x}\\left ( \\frac{\\sqrt{2} x}{w_0}\\right )\\,
            H_{n_y}\\left ( \\frac{\\sqrt{2} y}{w_0}\\right )\\,
            \\exp\\left( -\\frac{x^2+y^2}{w_0^2} \\right)

        where  :math:`H_{n}` is the Hermite polynomial of order :math:`n`.

        Parameters
        ----------
        w0: float (in meter)
            The waist of the laser pulse, i.e. :math:`w_0` in the above formula.
        n_x: int (dimensionless)
            The order of hermite polynomial in the x direction
        n_y: int (dimensionless)
            The order of hermite polynomial in the y direction
        """
        super().__init__()
        self.w0 = w0
        self.n_x = n_x
        self.n_y = n_y

    def _evaluate(self, x, y):
        """
        Returns the transverse envelope

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
        envelope = (
            hermite(self.n_x)(np.sqrt(2) * x / self.w0)
            * hermite(self.n_y)(np.sqrt(2) * y / self.w0)
            * np.exp(-(x**2 + y**2) / self.w0**2)
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


class SummedTransverseProfile(HermiteGaussianTransverseProfile):
    """Class for a transverse profile that is the sum of several profiles."""

    def __init__(self, *profiles):
        """
        Initialize the transverse profile.

        Parameters
        ----------
        *profiles: list of HermiteGaussianTransverseProfile objects
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
        self.n_x = None
        self.n_y = None
        # Check if all profiles are instances of HermiteGaussianTransverseProfile
        if not all(
            [isinstance(p, HermiteGaussianTransverseProfile) for p in self.profiles]
        ):
            # If not, raise a ValueError
            raise ValueError(
                "All profiles must be HermiteGaussianTransverseProfile objects."
            )

    def _evaluate(self, x, y):
        """Return the sum of the profiles."""
        return sum([p.evaluate(x, y) for p in self.profiles])


class ScaledTransverseProfile(HermiteGaussianTransverseProfile):
    """Class for a transverse profile that is scaled by a factor."""

    def __init__(self, profile, factor):
        """
        Initialize the transverse profile.

        Parameters
        ----------
        profile: HermiteGaussianTransverseProfile object
            The profile to be scaled.
        factor: float
            The scaling factor.
        """
        self.w0 = profile.w0
        self.n_x = profile.n_x
        self.n_y = profile.n_y
        self.profile = profile
        self.factor = factor
        if not isinstance(self.factor, (int, float)):
            raise ValueError("The scaling factor must be a float.")

    def _evaluate(self, x, y):
        """Return the scaled profile."""
        return self.factor * self.profile.evaluate(x, y)
