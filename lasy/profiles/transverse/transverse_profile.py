import numpy as np


class TransverseProfile(object):
    """
    Base class for all transverse profiles.

    Any new transverse profile should inherit from this class, and define its own
    `evaluate` method, using the same signature as the method below.
    """

    def __init__(self):
        """Initialize the transverse profile."""
        pass

    def evaluate(self, x, y):
        """
        Return the transverse envelope.

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
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        return np.zeros(x.shape, dtype="complex128")

    def __add__( self, other ):
        """Overload the + operations for laser profiles."""
        return( SummedTransverseProfile( self, other ) )

    def __mul__(self, other):
        """Overload the * operations for laser profiles."""
        return( ScaledTransverseProfile( self, other ) )
    
    def __rmul__(self, other):
        """Overload the * operations for laser profiles."""
        return( ScaledTransverseProfile( self, other ) )


class SummedTransverseProfile(TransverseProfile):
    """Class for a transverse profile that is the sum of several profiles."""

    def __init__(self, *profiles):
        """
        Initialize the transverse profile.

        Parameters
        ----------
        *profiles: list of TransverseProfile objects
            The profiles to be summed.
        """
        super().__init__()
        self.profiles = profiles
    
    def evaluate(self, x, y):
        """Return the sum of the profiles."""
        return sum([p.evaluate(x, y) for p in self.profiles])

class ScaledTransverseProfile(TransverseProfile):
    """Class for a transverse profile that is scaled by a factor."""

    def __init__(self, profile, factor):
        """
        Initialize the transverse profile.

        Parameters
        ----------
        profile: TransverseProfile object
            The profile to be scaled.
        factor: float
            The scaling factor.
        """
        super().__init__()
        self.profile = profile
        self.factor = factor
        if not isinstance(self.factor, (int, float)):
            raise ValueError("The scaling factor must be a float.")

    def evaluate(self, x, y):
        """Return the scaled profile."""
        return self.factor * self.profile.evaluate(x, y)