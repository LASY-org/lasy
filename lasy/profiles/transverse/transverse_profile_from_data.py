import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .transverse_profile import TransverseProfile


class TransverseProfileFromData(TransverseProfile):
    """
    Derived class for transverse laser profile created using
    data from an experimental measurement or from the output
    of another code. 
    """

    def __init__(self,intensity_data,lo,hi,center_beam=True):
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

        center_beam: bool
            If True (default), the beam will be centered based on its
            center of mass.

        """

        n_y, n_x = np.shape(intensity_data)
        
        dx = (hi[0] - lo[0])/n_x
        dy = (hi[1] - lo[1])/n_y

        x_data = np.linspace(lo[0],hi[0],n_x)
        y_data = np.linspace(lo[1],hi[1],n_y)
        
        assert dx == dy, "Data elements are not square"


        if center_beam:
            # find the beam center
            img_tot = np.sum(intensity_data)
            x0 = np.sum(np.dot(intensity_data, x_data)) / img_tot
            y0 = np.sum(np.dot(intensity_data.T, y_data)) / img_tot

            x_data -= x0
            y_data -= y0

            self.beam_shift_x = x0
            self.beam_shift_y = y0
        else:
            self.beam_shift_x = 0
            self.beam_shift_y = 0

        self.field_interp = RegularGridInterpolator((y_data,x_data),
                                np.sqrt(intensity_data), bounds_error=False,
                                fill_value=0)
                                
        self.center_beam = center_beam




    def evaluate(self, x, y):
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

        envelope = self.field_interp((y,x))

        return envelope

