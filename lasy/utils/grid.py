import numpy as np

time_axis_indx = -1


class Grid:
    """
    Store the envelope in temporal and spectral space and corresponding metadata.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    lo, hi : list of scalars
        Lower and higher end of the physical domain.
        One element per direction (2 for ``dim='rt'``, 3 for ``dim='xyt'``)

    npoints : tuple of int
        Number of points in each direction.
        One element per direction (2 for ``dim='rt'``, 3 for ``dim='xyt'``)
        For the moment, the lower end is assumed to be (0,0) in rt and (0,0,0) in xyt

    n_azimuthal_modes : int (optional)
        Only used if ``dim`` is ``'rt'``. The number of azimuthal modes
        used in order to represent the laser field.
    """

    def __init__(self, dim, lo, hi, npoints, n_azimuthal_modes=None):
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
            self.shape = self.npoints
        elif dim == "rt":
            # Azimuthal modes are arranged in the following order:
            # 0, 1, 2, ..., n_azimuthal_modes-1, -n_azimuthal_modes+1, ..., -1
            ncomp = 2 * self.n_azimuthal_modes - 1
            self.shape = (ncomp, self.npoints[0], self.npoints[1])
        self.temporal_field = np.zeros(self.shape, dtype="complex128")
        self.temporal_field_valid = False
        self.spectral_field = np.zeros(self.shape, dtype="complex128")
        self.spectral_field_valid = False

    def set_temporal_field(self, field):
        """
        Set the temporal field.

        Parameters
        ----------
        field : ndarray of complexs
            The temporal field.
        """
        assert field.shape == self.temporal_field.shape
        assert field.dtype == "complex128"
        self.temporal_field[:, :, :] = field
        self.temporal_field_valid = True
        self.spectral_field_valid = False  # Invalidates the spectral field

    def set_spectral_field(self, field):
        """
        Set the spectral field.

        Parameters
        ----------
        field : ndarray of complexs
            The spectral field.
        """
        assert field.shape == self.spectral_field.shape
        assert field.dtype == "complex128"
        self.spectral_field[:, :, :] = field
        self.spectral_field_valid = True
        self.temporal_field_valid = False  # Invalidates the temporal field

    def get_temporal_field(self):
        """
        Return a copy of the temporal field.

        (Modifying the returned object will not modify the original field stored
        in the Grid object ; one must use set_temporal_field to do so.)

        Returns
        -------
        field : ndarray of complexs
            The temporal field.
        """
        # We return a copy, so that the user cannot modify
        # the original field, unless get_temporal_field is called
        if self.temporal_field_valid:
            return self.temporal_field.copy()
        elif self.spectral_field_valid:
            self.spectral2temporal_fft()
            return self.temporal_field.copy()
        else:
            raise ValueError("Both temporal and spectral fields are invalid")

    def get_spectral_field(self):
        """
        Return a copy of the spectral field.

        (Modifying the returned object will not modify the original field stored
        in the Grid object ; one must use set_spectral_field to do so.)

        Returns
        -------
        field : ndarray of complexs
            The spectral field.
        """
        # We return a copy, so that the user cannot modify
        # the original field, unless set_spectral_field is called
        if self.spectral_field_valid:
            return self.spectral_field.copy()
        elif self.temporal_field_valid:
            self.temporal2spectral_fft()
            return self.spectral_field.copy()
        else:
            raise ValueError("Both temporal and spectral fields are invalid")

    def temporal2spectral_fft(self):
        """
        Perform the Fourier transform of field from temporal to spectral space.

        (Only along the time axis, not along the transverse spatial coordinates.)
        """
        assert self.temporal_field_valid
        self.spectral_field = np.fft.ifft(
            self.temporal_field, axis=time_axis_indx, norm="backward"
        )
        self.spectral_field_valid = True

    def spectral2temporal_fft(self):
        """
        Perform the Fourier transform of field from spectral to temporal space.

        (Only along the time axis, not along the transverse spatial coordinates.)
        """
        assert self.spectral_field_valid
        self.temporal_field = np.fft.fft(
            self.spectral_field, axis=time_axis_indx, norm="backward"
        )
        self.temporal_field_valid = True
