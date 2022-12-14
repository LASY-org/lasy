import numpy as np

class TransverseProfile(object):
    """
    Base class for all transverse profiles.

    Any new transverse profile should inherit from this class, and define its own
    `evaluate` method, using the same signature as the method below.
    """
    def __init__( self ):
        """
        Initialize the transverse profile.
        """
        pass

    def evaluate( self, x, y ):
        """
        Fills the envelope field of the laser

        Parameters
        -----------
        TODO
        """
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        return np.zeros( x.shape, dtype='complex128' )
