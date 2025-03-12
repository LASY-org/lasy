import numpy as np


class TransverseProfile(object):
    """
    Base class for all transverse profiles.

    Any new transverse profile should inherit from this class, and define its own
    `evaluate` method, using the same signature as the method below.
    For most cases, use derived classes instead of this base class.

    Common operators (addition and multiplication by a scalar) are provided as part of this base class.
    For such operations, the user is responsible for handling the complex phase and weights of summed profiles.
    In particular, summing between different types of profiles is not recommended.
    This feature is tailored for linear combination of profiles of the same type, in particular analytical profiles like Laguerre-gauss or Hermite-Gauss.
    """

    def __init__(self):
        # Initialise x and y spatial offsets as placeholders
        self.x_offset = 0
        self.y_offset = 0
        self.is_plane_wave = False

    def _evaluate(self, x, y):
        """
        Return the transverse envelope.

        Parameters
        ----------
        x, y : ndarrays of floats
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

    def __add__(self, other):
        """Return the sum of two transverse profiles."""
        return SummedTransverseProfile(self, other)

    def __mul__(self, factor):
        """Return the scaled transverse profile."""
        return ScaledTransverseProfile(self, factor)

    def __rmul__(self, factor):
        """Return the scaled transverse profile."""
        return ScaledTransverseProfile(self, factor)

    def __update_is_plane_wave__(self, value):
        """Update state of is_plane_wave variable."""
        self.is_plane_wave = value

    def evaluate(self, x, y):
        """
        Return the transverse envelope modified by any spatial offsets.

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
        return self._evaluate(x + self.x_offset, y + self.y_offset)

    def set_offset(self, x_offset, y_offset):
        """
        Populate the x and y spatial offsets of the profile.

        The profile will be shifted by these according to
        x+x_offset and y+y_offset prior to execution of
        _evaluate.

        Parameters
        ----------
        x_offset, y_offset: floats (m)
            Define spatial offsets to the beam. That is, how much
            to shift the beam by transversely
        """
        self.x_offset = x_offset
        self.y_offset = y_offset

        return self


class SummedTransverseProfile(TransverseProfile):
    """
    Base class for transverse profiles that are the sum of several other transverse profiles.

    Transverse Profile class that represents the sum of multiple transverse profiles.

    Parameters
    ----------
    transverse_profiles: list of TransverseProfile objects
        List of transverse profiles to be summed.
    """

    def __init__(self, *transverse_profiles):
        """Initialize the summed profile."""
        TransverseProfile.__init__(self)
        # Check that all transverse_profiles are TransverseProfile objects
        assert all([isinstance(tp, TransverseProfile) for tp in transverse_profiles]), (
            "All summands must be Profile objects."
        )
        self.transverse_profiles = transverse_profiles

    def evaluate(self, x, y):
        """Return the envelope field of the summed profile."""
        # Sum the fields of each profile
        return sum([tp.evaluate(x, y) for tp in self.transverse_profiles])


class ScaledTransverseProfile(TransverseProfile):
    """
    Base class for transverse profiles that are scaled by a factor.

    Transverse Profile class that represents scaled transverse profiles.

    Parameters
    ----------
    transverse_profile: TrasnverseProfile object
        Trasnverse profile to be scaled.
    factor: int or float
        Factor by which to scale the profile.
    """

    def __init__(self, transverse_profile, factor):
        """Initialize the summed profile."""
        TransverseProfile.__init__(self)
        # Check that the factor is a number
        assert isinstance(factor, (int, float, complex)), "The factor must be a number."
        # Check that the profile is a Profile object
        assert isinstance(transverse_profile, TransverseProfile), (
            "The profile must be a TransverseProfile object."
        )
        self.transverse_profile = transverse_profile
        self.factor = factor

    def evaluate(self, x, y):
        """Return the envelope field of the scaled transverse profile."""
        # Sum the fields of each profile
        return self.transverse_profile.evaluate(x, y) * self.factor
