import numpy as np
import scipy.constants as scc

class LongitudinalProfile(object):
    """
    Base class for longitudinal profiles of laser pulses.

    Any new longitudinal profile should inherit from this class, and define
    its own `evaluate` method, using the same signature as the method below.
    """
    def __init__( self, wavelength ):
        """
        Initialize the longitudinal profile
        """
        self.lambda0 = wavelength
        self.omega0 = 2*scc.pi*scc.c/self.lambda0

    def evaluate( self, t ):
        """
        Returns the longitudinal envelope

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
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        return np.zeros( t.shape, dtype='complex128' )
