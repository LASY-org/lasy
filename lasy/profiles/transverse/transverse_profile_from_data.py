import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .transverse_profile import TransverseProfile


class TransverseProfileFromData(TransverseProfile):
    """
    Derived class for transverse laser profile created using
    data from an experimental measurement or from the output
    of another code.
    """

    def __init__(self, intensity_data, lo, hi):
        """
        Uses user supplied data to define the transverse profile
        of the laser pulse.

        The data must be supplied as a 2D numpy array of intensity
        values (for example an imported cameran image from an
        experimental measurement).

        In the case of experimental measurements, this data
        should already have some undergone some preprocessing
        such as background subtraction and noise removal.

        The beam will be imported and automatically centered unless
        otherwise specified.

        Parameters:
        -----------
        intensity_data: 2Darray of floats
            The 2D transverse intensity profile of the laser pulse.

        lo, hi : list of scalars (in meters)
            Lower and higher end of the physical domain of the data.
            One element per direction (in this case 2)
        """
        super().__init__()

        intensity_data = intensity_data.astype("float64")

        n_y, n_x = np.shape(intensity_data)

        dx = (hi[0] - lo[0]) / n_x
        dy = (hi[1] - lo[1]) / n_y

        x_data = np.linspace(lo[0], hi[0], n_x)
        y_data = np.linspace(lo[1], hi[1], n_y)

        # Normalise the profile such that its squared integeral == 1
        intensity_data /= np.sum(intensity_data) * dx * dy

        # Note here we use the square root of intensity to get the 'field'
        self.field_interp = RegularGridInterpolator(
            (y_data, x_data),
            np.sqrt(intensity_data),
            bounds_error=False,
            fill_value=0.0,
        )

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
        envelope: ndarray of floats
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y
        """

        envelope = self.field_interp((y, x))

        return envelope
