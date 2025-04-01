import numpy as np

from .transverse_profile import TransverseProfile


class PlaneWaveProfile(TransverseProfile):
    r"""
    Class representing a plane wave.

    Specifically, the transverse profile will be represented by a constant value.

    Note that using this profile will force the number of points in each transverse
    dimension to 1. Additionally, it will force the area of the transverse plane to
    1 m^2.

    """

    def __init__(self):
        super().__init__()
        self.__update_is_plane_wave__(True)

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
        envelope : ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y
        """
        return np.ones_like(x**2 + y**2)
