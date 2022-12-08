import numpy as np

class Grid:
    """
    Contains data for fields (in position or Fourier space), including metadata
    """

    def __init__(self, box, ncomps=1, array=None):
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
        if array is not None:
            assert(array.ndim == 3 if self.box.dim == 'xyt' else 2)
            if self.box.dim == 'xyt':
                assert(box.npoints == array.shape)
            else:
                assert(box.npoints == array.shape[1:])
            self.field = array
        else:
            if self.box.dim == 'xyt':
                self.field = np.zeros(box.npoints, dtype='complex128')
            elif self.box.dim == 'rt':
                # Supports only 1 azimuthal mode for now
                self.field = np.zeros((1, box.npoints[0], box.npoints[1]),
                                      dtype='complex128')
