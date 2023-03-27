import numpy as np


class Grid:
    """
    Contains data for fields (in position or Fourier space), including metadata

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - 'xyt': The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - 'rt' : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    box : Box
        Object containing metadata for the grid array
    """

    def __init__(self, dim, box):
        self.box = box
        if dim == "xyt":
            self.field = np.zeros(box.npoints, dtype="complex128")
        elif dim == "rt":
            # Azimuthal modes are arranged in the following order:
            # 0, 1, 2, ..., n_azimuthal_modes-1, -n_azimuthal_modes+1, ..., -1
            ncomp = 2 * self.box.n_azimuthal_modes - 1
            self.field = np.zeros(
                (ncomp, box.npoints[0], box.npoints[1]), dtype="complex128"
            )
