import numpy as np

from .fft_wrapper import fft, frequency_axis

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

    is_envelope : bool (optional)
        Whether the field provided uses the (complex) envelope representation, as
        used internally in lasy. If False, field is assumed to represent the
        the full (real) electric field (with fast oscillations).

    is_cw : bool (optional)
        Whether the laser pulse longitudinal profile is a continuous wave laer profile
        or not.

    is_plane_wave : bool (optional)
        Whether the laser pulse transverse profile is a plane wave laer profile or not.

    position : scalar (optional)
        Longitudinal (z) position in a beamline at which the pulse is defined.
    """

    def __init__(
        self,
        dim,
        lo,
        hi,
        npoints,
        n_azimuthal_modes=None,
        is_envelope=True,
        is_cw=False,
        is_plane_wave=False,
        position=0.0,
    ):
        # Metadata
        ndims = 2 if dim == "rt" else 3
        assert dim in ["rt", "xyt"]
        assert len(lo) == ndims
        assert len(hi) == ndims

        lo = list(lo)
        hi = list(hi)
        npoints = list(npoints)

        if is_cw:
            if npoints[-1] != 1:
                print(
                    "CW profile: overwrite npoints to only 1 cell in the longitudinal direction."
                )
            lo[-1] = -0.5
            hi[-1] = 0.5
            npoints[-1] = 1
        if is_plane_wave:
            if npoints[0] != 1:
                print(
                    "Plane wave: overwrite npoints to only 1 cell in the transverse directions."
                )
            if dim == "rt":
                lo[0] = 0.0
                hi[0] = np.sqrt(1 / np.pi)
                npoints[0] = 1
            else:
                lo[0] = -0.5
                hi[0] = 0.5
                npoints[0] = 1
                lo[1] = -0.5
                hi[1] = 0.5
                npoints[1] = 1

        self.npoints = npoints
        self.axes = []
        self.dx = []
        for i in range(ndims):
            self.axes.append(np.linspace(lo[i], hi[i], npoints[i]))
            if len(self.axes[i]) > 1:
                self.dx.append(self.axes[i][1] - self.axes[i][0])
            else:
                self.dx.append(hi[i] - lo[i])

        self.lo = lo
        self.hi = hi

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

        self.set_is_envelope(is_envelope)
        self.temporal_field = np.zeros(self.shape, dtype=self.dtype)
        self.temporal_field_valid = False
        self.spectral_field = np.zeros(self.shape, dtype="complex128")
        self.spectral_field_valid = False
        self.position = position

    def set_is_envelope(self, is_envelope):
        """
        Set is_envelope attribute. Also set dtype accordingly.

        Parameters
        ----------
        is_envelope : boolean
            Whether the grid should represent an envelope (True) or a full electric field (False)
        """
        assert is_envelope in [True, False]
        if is_envelope:
            self.dtype = "complex128"
        else:
            self.dtype = "float64"
        if hasattr(self, "temporal_field"):
            self.temporal_field = self.temporal_field.astype(dtype=self.dtype)
        self.is_envelope = is_envelope

    def set_temporal_field(self, field):
        """
        Set the temporal field.

        Parameters
        ----------
        field : ndarray of complexs
            The temporal field.
        """
        assert field.shape == self.temporal_field.shape
        assert field.dtype == self.dtype
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

        omega : 1d array of real numbers
            The frequency axis consistent with the spectral field.
            This is centered around 0, the central frequency of the envelope
            must be added separately to construct the physical frequency array.
        """
        # We return a copy, so that the user cannot modify
        # the original field, unless set_spectral_field is called
        assert self.is_envelope
        if not hasattr(self, "spectral_axis"):
            self.spectral_axis = frequency_axis("longitudinal", self.axes[-1], "real")
        if self.spectral_field_valid:
            return self.spectral_field.copy(), self.spectral_axis.copy()
        elif self.temporal_field_valid:
            self.temporal2spectral_fft()
            return self.spectral_field.copy(), self.spectral_axis.copy()
        else:
            raise ValueError("Both temporal and spectral fields are invalid")

    def temporal2spectral_fft(self):
        """
        Perform the Fourier transform of field from temporal to spectral space.

        (Only along the time axis, not along the transverse spatial coordinates.)
        """
        assert self.temporal_field_valid

        self.spectral_field, self.spectral_axis = fft(
            "longitudinal", self.temporal_field, self.axes[-1], "real"
        )

        self.spectral_field_valid = True

    def spectral2temporal_fft(self):
        """
        Perform the Fourier transform of field from spectral to temporal space.

        (Only along the time axis, not along the transverse spatial coordinates.)
        """
        assert self.spectral_field_valid

        self.temporal_field, _ = fft(
            "longitudinal", self.spectral_field, self.axes[-1], "frequency"
        )

        self.temporal_field_valid = True
