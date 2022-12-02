import numpy as np

class Grid:
    """
    Contains data for fields (in position or Fourier space), including metadata
    """

    def __init__(self, box, ncomps=1, array_in=None):
        """

        Parameters
        ----------
        box : Box
            Object containing metadata for the grid array

        ncomps : int
            Number of components for the array.
            Currently not used. We could:
            - Store a list of arrays
            - Add 1 dimension np.zeros(box.npoints+(ncomps,))
        """
        self.box = box
        self.ncomps = ncomps
        if array_in is not None:
            assert(box.npoints == array_in.shape)
            self.field = array_in.astype('complex128')
        else:
            self.field = np.zeros(box.npoints, dtype='complex128')
