import numpy as np

class Grid:
    """
    Contains data for fields (in position or Fourier space), including metadata
    """

    def __init__(self, box):
        """

        Parameters
        ----------
        box : Box
            Object containing metadata for the grid array
        """
        self.box = box
        if self.box.dim == 'xyt':
            self.field = np.zeros(box.npoints, dtype='complex128')
        elif self.box.dim == 'rt':
            # Azimuthal modes are arranged in the following order:
            # 0, 1, 2, ..., n_azimuthal_modes-1, -n_azimuthal_modes+1, ..., -1
            ncomp = 2*self.box.n_azimuthal_modes-1
            self.field = np.zeros((ncomp, box.npoints[0], box.npoints[1]),
                                    dtype='complex128')
