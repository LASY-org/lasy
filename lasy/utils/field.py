import numpy as np


class Field:
    def __init__(self, dim, lo, hi, npoints, n_azimuthal_modes):
        # Metadata
        ndims = 2 if dim == "rt" else 3
        assert dim in ["rt", "xyt"]
        assert len(lo) == ndims
        assert len(hi) == ndims

        self.lo = list(lo)
        self.hi = list(hi)
        self.npoints = npoints
        self.axes = []
        self.dx = []
        for i in range(ndims):
            self.axes.append(np.linspace(lo[i], hi[i], npoints[i]))
            self.dx.append(self.axes[i][1] - self.axes[i][0])

        if dim == "rt":
            self.n_azimuthal_modes = n_azimuthal_modes
            self.azimuthal_modes = np.r_[
                np.arange(n_azimuthal_modes), np.arange(-n_azimuthal_modes + 1, 0, 1)
            ]

        # Data
        if dim == "xyt":
            self.array = np.zeros(self.npoints, dtype="complex128")
        elif dim == "rt":
            # Azimuthal modes are arranged in the following order:
            # 0, 1, 2, ..., n_azimuthal_modes-1, -n_azimuthal_modes+1, ..., -1
            ncomp = 2 * self.n_azimuthal_modes - 1
            self.array = np.zeros(
                (ncomp, self.npoints[0], self.npoints[1]), dtype="complex128"
            )
