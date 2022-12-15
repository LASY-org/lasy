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
        Initialize the propagation direction of the laser.
        """
        self.lambda0 = wavelength
        self.omega0 = 2*scc.pi*scc.c/self.lambda0

    def evaluate( self, t ):
        """
        Fills the envelope field of the laser

        Parameters
        -----------
        TODO
        """
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        return np.zeros( t.shape, dtype='complex128' )
