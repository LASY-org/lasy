import numpy as np
from scipy.constants import c


class Profile(object):
    """
    Base class for all laser profiles.

    Any new laser profile should inherit from this class, and define its own
    `evaluate` method, using the same signature as the method below.

    Parameters
    ----------
    wavelength : scalar
        Central wavelength for which the laser pulse envelope is defined.

    pol : list of 2 complex numbers
        Polarization vector that multiplies array_in to get the Ex and Ey fields.
        The envelope of each component of the electric field is given by:
        - Ex_env = array_in*pol(0)
        - Ey_env = array_in*pol(1)
        Standard polarizations can be obtained from:
        - Linear polarization in x: pol = (1,0)
        - Linear polarization in y: pol = (0,1)
        - Circular polarization: pol = (1,j)/sqrt(2) (j is the imaginary number)
        The polarization vector is normalized to have a unitary magnitude.

    """

    def __init__(self, wavelength, pol):
        assert len(pol) == 2
        norm_pol = np.sqrt(np.abs(pol[0]) ** 2 + np.abs(pol[1]) ** 2)
        self.pol = np.array([pol[0] / norm_pol, pol[1] / norm_pol])
        self.lambda0 = wavelength
        self.omega0 = 2 * np.pi * c / self.lambda0

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
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        return np.zeros_like(x, dtype="complex128")

    def __add__(self, other):
        """Return the sum of two profiles."""
        return SummedProfile(self, other)

    def __mul__(self, factor):
        """Return the scaled profile."""
        return ScaledProfile(self, factor)

    def __rmul__(self, factor):
        """Return the scaled profile."""
        return ScaledProfile(self, factor)


class SummedProfile(Profile):
    """
    Base class for profiles that are the sum of several other profiles.

    Profile class that represents the sum of multiple profiles.

    Parameters
    ----------
    profiles: list of Profile objects
        List of profiles to be summed.
    """

    def __init__(self, *profiles):
        """Initialize the summed profile."""
        # Check that all profiles are Profile objects
        assert all(
            [isinstance(p, Profile) for p in profiles]
        ), "All summands must be Profile objects."
        self.profiles = profiles
        # Get the wavelength values from each profile
        lambda0s = [p.lambda0 for p in self.profiles]
        pols = [p.pol for p in self.profiles]
        # Check that all wavelengths are the same
        assert np.allclose(
            lambda0s, lambda0s[0]
        ), "Added profiles must have the same wavelength."
        lambda0 = profiles[0].lambda0
        # Check that all polarizations are the same
        assert np.allclose(
            pols, pols[0]
        ), "Added profiles must have the same polarization."
        pol = profiles[0].pol
        # Initialize the parent class
        super().__init__(lambda0, pol)

    def evaluate(self, x, y, t):
        """Return the envelope field of the summed profile."""
        # Sum the fields of each profile
        return sum([p.evaluate(x, y, t) for p in self.profiles])


class ScaledProfile(Profile):
    """
    Base class for profiles that are scaled by a factor.

    Profile class that represents scaled profiles.

    Parameters
    ----------
    profiles: Profile object
        Profile to be scaled.
    factor: int or float
        Factor by which to scale the profile.
    """

    def __init__(self, profile, factor):
        """Initialize the summed profile."""
        # Check that the factor is a number
        assert isinstance(factor, (int, float)), "The factor must be a number."
        # Check that the profile is a Profile object
        assert isinstance(profile, Profile), "The profile must be a Profile object."
        self.profile = profile
        self.factor = factor
        # Get the wavelength and polarization from the profile
        lambda0 = profile.lambda0
        pol = profile.pol
        # Initialize the parent class
        super().__init__(lambda0, pol)

    def evaluate(self, x, y, t):
        """Return the envelope field of the scaled profile."""
        # Sum the fields of each profile
        return self.profile.evaluate(x, y, t) * self.factor
