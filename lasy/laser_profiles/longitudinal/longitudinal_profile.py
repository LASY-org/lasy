import numpy as np
import scipy.constants as scc

class LongitudinalProfile(object):
    """
    Base class for longitudinal profiles of laser pulses.

    Any new longitudinal profile should inherit from this class, and define its own
    `evaluate` method, using the same signature as the method below.
    """
    def __init__( self, wavelength ):
        """
        Initialize the propagation direction of the laser.
        (Each subclass should call this method at initialization.)

        Parameters:
        -----------
        wavelength: scalar
            Central wavelength for which the laser pulse envelope is defined.
        """
        self.lambda0 = wavelength
        self.omega0 = 2*scc.pi*scc.c/self.lambda0

    def evaluate( self, envelope, axis ):
        """
        Fills the envelope field of the laser
        Usage: evaluate(envelope, t)

        Parameters
        -----------
        envelope: ndarrays (V/m)
            Contains the values of the envelope field, to be filled

        axis: Time coordinates at which the envelope should be evaluated.
        """
        # The base class only defines dummy fields
        # (This should be replaced by any class that inherits from this one.)
        pass
