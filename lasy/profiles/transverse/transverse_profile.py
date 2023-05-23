import numpy as np


class TransverseProfile(object):
    """
    Base class for all transverse profiles.

    Any new transverse profile should inherit from this class, and define its own
    `evaluate` method, using the same signature as the method below.
    """

    def __init__(self):
        # Initialise x and y spatial offsets as placeholders
        self.x_offset = 0
        self.y_offset = 0

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

    def evaluate(self, x, y):
        """
        Return the transverse envelope modified by any spatial offsets.

        This is the public facing evaluate method, it calls the _evaluate function of the derived class.

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
