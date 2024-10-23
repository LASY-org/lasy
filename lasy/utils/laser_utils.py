import numpy as np
from axiprop.containers import ScalarFieldEnvelope
from axiprop.lib import PropagatorFFT2, PropagatorResampling
from scipy.constants import c, e, epsilon_0, m_e
from scipy.interpolate import interp1d
from scipy.signal import hilbert

from .grid import Grid


def compute_laser_energy(dim, grid):
    """
    Compute the total laser energy that corresponds to the current envelope data.

    This is used mainly for normalization purposes.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - 'xyt': The laser pulse is represented on a 3D grid:
                 Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - 'rt' : The laser pulse is represented on a 2D grid:
                 Cylindrical (r) transversely, and temporal (t) longitudinally.

    grid : a Grid object.
        It contains an ndarray (V/m) with
        the value of the envelope field and the associated metadata
        that defines the points at which the laser is defined.

    Returns
    -------
    energy: float (in Joules)
    """
    # This uses the following volume integral:
    # $E_{laser} = \int dV \;\frac{\epsilon_0}{2} | E_{env} |^2$
    # which assumes that we can average over the oscilations at the
    # specified laser wavelength.
    # This probably needs to be generalized for few-cycle laser pulses.

    envelope = grid.get_temporal_field()

    dV = get_grid_cell_volume(grid, dim)

    if dim == "xyt":
        energy = ((dV * epsilon_0) * abs(envelope) ** 2).sum()
    else:  # dim == "rt":
        energy = (
            dV[np.newaxis, :, np.newaxis] * epsilon_0 * abs(envelope[:, :, :]) ** 2
        ).sum()

    if grid.is_envelope:
        energy *= 0.5

    return energy


def normalize_energy(dim, energy, grid):
    """
    Normalize energy of the laser pulse contained in grid.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - 'xyt': The laser pulse is represented on a 3D grid:
                 Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - 'rt' : The laser pulse is represented on a 2D grid:
                 Cylindrical (r) transversely, and temporal (t) longitudinally.

    energy : scalar (J)
        Energy of the laser pulse after normalization.

    grid: a Grid object
        Contains value of the laser envelope and metadata.
    """
    if energy is None:
        return

    current_energy = compute_laser_energy(dim, grid)
    if current_energy == 0.0:
        print("Field is zero everywhere, normalization will be skipped")
    else:
        norm_factor = (energy / current_energy) ** 0.5
        field = grid.get_temporal_field()
        field *= norm_factor
        grid.set_temporal_field(field)


def normalize_peak_field_amplitude(amplitude, grid):
    """
    Normalize energy of the laser pulse contained in grid.

    Parameters
    ----------
    amplitude : scalar (V/m)
        Peak field amplitude of the laser pulse after normalization.

    grid : a Grid object
        Contains value of the laser envelope and metadata.
    """
    if amplitude is not None:
        field = grid.get_temporal_field()
        field_max = np.abs(field).max()
        if field_max == 0.0:
            print("Field is zero everywhere, normalization will be skipped")
        else:
            field *= amplitude / field_max
            grid.set_temporal_field(field)


def normalize_peak_intensity(peak_intensity, grid):
    """
    Normalize energy of the laser pulse contained in grid.

    Parameters
    ----------
    peak_intensity : scalar (W/m^2)
        Peak field amplitude of the laser pulse after normalization.

    grid : a Grid object
        Contains value of the laser envelope and metadata.
    """
    if peak_intensity is not None:
        field = grid.get_temporal_field()
        intensity = np.abs(epsilon_0 * field**2 / 2 * c)
        input_peak_intensity = intensity.max()
        if input_peak_intensity == 0.0:
            print("Field is zero everywhere, normalization will be skipped")
        else:
            grid.set_temporal_field(np.sqrt(peak_intensity / input_peak_intensity))


def get_full_field(laser, theta=0, slice=0, slice_axis="x", Nt=None):
    """
    Reconstruct the laser pulse with carrier frequency on the default grid.

    Parameters
    ----------
    theta : float (rad) (optional)
        Azimuthal angle
    slice : float (optional)
        Normalised position of the slice from -0.5 to 0.5.
    Nt: int (optional)
        Number of time points on which field should be sampled. If is None,
        the orignal time grid is used, otherwise field is interpolated on a
        new grid.

    Returns
    -------
        Et : ndarray (V/m)
            The reconstructed field, with shape (Nr, Nt) (for `rt`)
            or (Nx, Nt) (for `xyt`).
        extent : ndarray (Tmin, Tmax, Xmin, Xmax)
            Physical extent of the reconstructed field.
    """
    omega0 = laser.profile.omega0
    env = laser.grid.get_temporal_field()
    time_axis = laser.grid.axes[-1]

    # If the field is not an envelope, it is a full field, so no
    # reason to recompute the full field.
    assert laser.grid.is_envelope

    if laser.dim == "rt":
        azimuthal_phase = np.exp(-1j * laser.grid.azimuthal_modes * theta)
        env_upper = env * azimuthal_phase[:, None, None]
        env_upper = env_upper.sum(0)
        azimuthal_phase = np.exp(1j * laser.grid.azimuthal_modes * theta)
        env_lower = env * azimuthal_phase[:, None, None]
        env_lower = env_lower.sum(0)
        env = np.vstack((env_lower[::-1][:-1], env_upper))
    elif slice_axis == "x":
        Nx_middle = env.shape[0] // 2 - 1
        Nx_slice = int((1 + slice) * Nx_middle)
        env = env[Nx_slice, :]
    elif slice_axis == "y":
        Ny_middle = env.shape[1] // 2 - 1
        Ny_slice = int((1 + slice) * Ny_middle)
        env = env[:, Ny_slice, :]
    else:
        return None

    if Nt is not None:
        Nr = env.shape[0]
        time_axis_new = np.linspace(laser.grid.lo[-1], laser.grid.hi[-1], Nt)
        env_new = np.zeros((Nr, Nt), dtype=env.dtype)

        for ir in range(Nr):
            interp_fu_abs = interp1d(time_axis, np.abs(env[ir]))
            slice_abs = interp_fu_abs(time_axis_new)
            interp_fu_angl = interp1d(time_axis, np.unwrap(np.angle(env[ir])))
            slice_angl = interp_fu_angl(time_axis_new)
            env_new[ir] = slice_abs * np.exp(1j * slice_angl)

        time_axis = time_axis_new
        env = env_new

    env *= np.exp(-1j * omega0 * time_axis[None, :])
    env = np.real(env)

    if laser.dim == "rt":
        ext = np.array(
            [
                laser.grid.lo[-1],
                laser.grid.hi[-1],
                -laser.grid.hi[0],
                laser.grid.hi[0],
            ]
        )
    else:
        ext = np.array(
            [
                laser.grid.lo[-1],
                laser.grid.hi[-1],
                laser.grid.lo[0],
                laser.grid.hi[0],
            ]
        )

    return env, ext


def get_spectrum(grid, dim, range=None, bins=20, omega0=None, method="sum"):
    r"""
    Get the frequency spectrum of an envelope or electric field.

    The spectrum can be calculated in three different ways, depending on the
    `method` specified by the user:

    Initially, the spectrum is calculated as the Fourier transform of the
    electric field :math:`E(t)`.

    ..math::
        \int E(t) e^{-i \omega t} dt

    neglecting the negative frequencies. If ``method=="raw"``, no further
    processing is done and the returned spectrum is a complex array with the
    same transverse dimensions as the input grid. The units are
    :math:`\mathrm{V / Hz}`.

    For the other methods, the spectral energy density is calculated as

    ..math::
        \frac{\epsilon_0 c}{2\pi} |\int E(t) e^{-i \omega t} dt| ^ 2

    If ``method=="on_axis"``, a 1D real array with on-axis value of the
    equation above is returned. The units are :math:`\mathrm{J / (rad Hz m^2)}`.

    Otherwise, if ``method=="sum"`` (default), the transverse integral of the
    spectral energy density is calculated:

    ..math::
        \frac{\epsilon_0 c}{2\pi} \int |\int E(t) e^{-i \omega t} dt| ^ 2 dx dy

    The units of this array are :math:`\mathrm{J / (rad Hz)}`

    Parameters
    ----------
    grid : a Grid object.
        It contains an ndarray with the field data from which the
        spectrum is computed, and the associated metadata. The last axis must
        be the longitudinal dimension.

    dim : string (optional)
        Dimensionality of the array. Options are:

        - 'xyt': The laser pulse is represented on a 3D grid:
                 Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - 'rt' : The laser pulse is represented on a 2D grid:
                 Cylindrical (r) transversely, and temporal (t) longitudinally.

    range : list of float (optional)
        List of two values indicating the minimum and maximum frequency of the
        spectrum. If provided, only the FFT spectrum within this range
        will be returned using interpolation.

    bins : int (optional)
        Number of bins into which to interpolate the spectrum if a `range`
        is given.

    omega0 : scalar (optional)
        Angular frequency at which the envelope is defined.
        Only used if grid.is_envelope is True.

    method : {'sum', 'on_axis', 'raw'} (optional)
        Determines the type of spectrum that is returned as described above.
        By default 'sum'.

    Returns
    -------
    spectrum : ndarray
        Array with the spectrum (units and array type depend on ``method``).

    omega : ndarray
        Array with the angular frequencies of the spectrum.
    """
    # Get the frequencies of the fft output.
    freq = np.fft.fftfreq(grid.shape[-1], d=(grid.axes[-1][1] - grid.axes[-1][0]))
    omega = 2 * np.pi * freq

    # Get on axis or full field.
    field = grid.get_temporal_field()
    if method == "on_axis":
        if dim == "xyt":
            nx, ny, _ = field.shape
            field = field[nx // 2, ny // 2]
        else:
            field = field[0, 0]

    # Get spectrum.
    if grid.is_envelope:
        # Assume that the FFT of the envelope and the FFT of the complex
        # conjugate of the envelope do not overlap. Then we only need
        # one of them.
        assert omega0 is not None
        spectrum = 0.5 * np.fft.fft(field) * grid.dx[-1]
        omega = omega0 - omega
        # Sort frequency array (and the spectrum accordingly).
        i_sort = np.argsort(omega)
        omega = omega[i_sort]
        spectrum = spectrum[..., i_sort]
        # Keep only positive frequencies.
        i_keep = omega >= 0
        omega = omega[i_keep]
        spectrum = spectrum[..., i_keep]
    else:
        spectrum = np.fft.fft(field) * grid.dx[-1]
        # Keep only positive frequencies.
        i_keep = spectrum.shape[-1] // 2
        omega = omega[:i_keep]
        spectrum = spectrum[..., :i_keep]

    # Convert to spectral energy density (J/(m^2 rad Hz)).
    if method != "raw":
        spectrum = np.abs(spectrum) ** 2 * epsilon_0 * c / np.pi

    # Integrate transversely.
    if method == "sum":
        dV = get_grid_cell_volume(grid, dim)
        dz = grid.dx[-1] * c
        if dim == "xyt":
            spectrum = np.sum(spectrum * dV / dz, axis=(0, 1))
        else:
            spectrum = np.sum(spectrum[0] * dV[:, np.newaxis] / dz, axis=0)

    # If the user specified a frequency range, interpolate into it.
    if method in ["sum", "on_axis"] and range is not None:
        omega_interp = np.linspace(*range, bins)
        spectrum = np.interp(omega_interp, omega, spectrum)
        omega = omega_interp

    return spectrum, omega


def get_frequency(
    grid,
    dim=None,
    is_hilbert=False,
    omega0=None,
    phase_unwrap_nd=False,
    lower_bound=0.2,
    upper_bound=5.0,
):
    """
    Get the local and average frequency of a signal, either electric field or envelope.

    Parameters
    ----------
    grid : a Grid object.
        It contains a ndarrays with the field data from which the
        frequency is computed, and the associated metadata. The last axis must
        be the longitudinal dimension.
        Can be the full electric field or the envelope.

    dim : string (optional)
        Dimensionality of the array.
        Options are:

        - 'xyt': The laser pulse is represented on a 3D grid:
                 Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - 'rt' : The laser pulse is represented on a 2D grid:
                 Cylindrical (r) transversely, and temporal (t) longitudinally.

    is_hilbert : boolean (optional)
        If True, the field argument is assumed to be a Hilbert transform, and
        is used through the computation. Otherwise, the Hilbert transform is
        calculated in the function.

    omega0 : scalar
        Angular frequency at which the envelope is defined.
        Only used if grid.is_envelope is True.

    phase_unwrap_nd : boolean (optional)
        If True, the phase unwrapping is n-dimensional (2- or 3-D depending on dim).
        If False, the phase unwrapping is done in t, treating each transverse cell
        separately. This should be less accurate but faster.
        If set to True, scikit-image must be installed.

    lower_bound : scalar (optional)
        Relative lower bound for the local frequency
        Frequencies lower than lower_bound * central_omega are cut
        to lower_bound * central_omega.

    upper_bound : scalar (optional)
        Relative upper bound for the local frequency
        Frequencies larger than upper_bound * central_omega are cut
        to upper_bound * central_omega.

    Returns
    -------
    omega : nd array of doubles
        local angular frequency.

    central_omega : scalar
        Central angular frequency (averaged omega, weighted by the local
        envelope amplitude).
    """
    field = grid.get_temporal_field()

    # Assumes t is last dimension!
    if grid.is_envelope:
        assert omega0 is not None
        phase = np.unwrap(np.angle(field))
        omega = omega0 + np.gradient(-phase, grid.axes[-1], axis=-1, edge_order=2)
        central_omega = np.average(omega, weights=np.abs(field))
    else:
        assert dim in ["xyt", "rt"]
        if dim == "xyt" and phase_unwrap_nd:
            print("WARNING: using 3D phase unwrapping, this can be expensive")

        h = field if is_hilbert else hilbert_transform(grid)
        h = np.squeeze(field)
        if phase_unwrap_nd:
            try:
                from skimage.restoration import unwrap_phase

                skimage_installed = True
            except ImportError:
                skimage_installed = False
            assert skimage_installed, (
                "scikit-image must be install for nd phase unwrapping.",
                "Please install scikit-image or use phase_unwrap_nd=False.",
            )
            phase = unwrap_phase(np.angle(h))
        else:
            phase = np.unwrap(np.angle(h))
        omega = np.gradient(-phase, grid.axes[-1], axis=-1, edge_order=2)

        if dim == "xyt":
            weights = np.abs(h)
        else:
            r = grid.axes[0].reshape((grid.axes[0].size, 1))
            weights = r * np.abs(h)
        central_omega = np.average(omega, weights=weights)

    # Filter out too small frequencies
    omega = np.maximum(omega, lower_bound * central_omega)
    # Filter out too large frequencies
    omega = np.minimum(omega, upper_bound * central_omega)

    return omega, central_omega


def get_duration(grid, dim):
    """Get duration of the intensity of the envelope, measured as RMS.

    Parameters
    ----------
    grid : Grid
        The grid with the envelope to analyze.
    dim : str
        Dimensionality of the grid.

    Returns
    -------
    float
        RMS duration of the envelope intensity in seconds.
    """
    # Calculate weights of each grid cell (amplitude of the field).
    dV = get_grid_cell_volume(grid, dim)
    field = grid.get_temporal_field()
    if dim == "xyt":
        weights = np.abs(field) ** 2 * dV
    else:  # dim == "rt":
        weights = np.abs(field) ** 2 * dV[np.newaxis, :, np.newaxis]
    # project weights to longitudinal axes
    weights = np.sum(weights, axis=(0, 1))
    return weighted_std(grid.axes[-1], weights)


def field_to_vector_potential(grid, omega0):
    """
    Convert envelope from electric field (V/m) to normalized vector potential.

    Parameters
    ----------
    grid : a Grid object.
        Contains the array of the electric field, to be converted to normalized
        vector potential, with corresponding metadata.
        The last axis must be the longitudinal dimension.

    omega0 : scalar
        Angular frequency at which the envelope is defined.

    Returns
    -------
    Normalized vector potential
    """
    # Here, we neglect the time derivative of the envelope of E, the first RHS
    # term in: E = -dA/dt + 1j * omega0 * A where E and A are the field and
    # vector potential envelopes, respectively
    assert grid.is_envelope
    omega, _ = get_frequency(grid, omega0=omega0)
    return -1j * e * grid.get_temporal_field() / (m_e * omega * c)


def vector_potential_to_field(grid, omega0, direct=True):
    """
    Convert envelope from electric field (V/m) to normalized vector potential.

    Parameters
    ----------
    grid : a Grid object.
        Contains the array of the normalized vector potential, to be
        converted to field, with corresponding metadata.
        The last axis must be the longitudinal dimension.

    omega0 : scalar
        Angular frequency at which the envelope is defined.

    direct : boolean (optional)
        If true, the conversion is done directly with derivative of vector
        potential. Otherwise, this is done using the local frequency.

    Returns
    -------
    Envelope of the electric field (V/m).
    """
    assert grid.is_envelope
    field = grid.get_temporal_field()
    if direct:
        A = (
            -np.gradient(field, grid.axes[-1], axis=-1, edge_order=2)
            + 1j * omega0 * field
        )
        return m_e * c / e * A
    else:
        omega, _ = get_frequency(grid, omega0=omega0)
        return 1j * m_e * omega * c * field / e


def field_to_envelope(grid, dim, phase_unwrap_nd=False):
    """Get the complex envelope of a field by applying a Hilbert transform.

    Parameters
    ----------
    grid : Grid
        The field from which to extract the envelope.

    dim : str
        Dimensions of the field. Possible values are `'xyt'` or `'rt'`.

    phase_unwrap_nd : boolean (optional)
        If True, the phase unwrapping is n-dimensional (2- or 3-D depending on dim).
        If False, the phase unwrapping is done in t, treating each transverse cell
        separately. This should be less accurate but faster.
        If set to True, scikit-image must be installed.

    Returns
    -------
    tuple
        A tuple with the envelope array and the central wavelength.
    """
    assert not grid.is_envelope

    field = grid.get_temporal_field()

    # hilbert transform needs inverted time axis.
    field = hilbert_transform(field)

    # Get central wavelength from array
    omg_h, omg0_h = get_frequency(
        grid,
        dim=dim,
        is_hilbert=True,
        phase_unwrap_nd=phase_unwrap_nd,
    )
    field *= np.exp(1j * omg0_h * grid.axes[-1])
    grid.set_is_envelope(True)
    grid.set_temporal_field(field)

    return grid, omg0_h


def hilbert_transform(field):
    """Make a hilbert transform of the grid field.

    Currently the arrays need to be flipped along t (both the input field and
    its transform) to get the imaginary part (and thus the phase) with the
    correct sign.

    Parameters
    ----------
    grid : Grid
        The lasy grid whose field should be transformed.
    """
    return hilbert(field[:, :, ::-1])[:, :, ::-1]


def get_grid_cell_volume(grid, dim):
    """Get the volume of the grid cells.

    Parameters
    ----------
    grid : Grid
        The grid form which to compute the cell volume
    dim : str
        Dimensionality of the grid.

    Returns
    -------
    float or ndarray
        A float with the cell volume (if dim=='xyt') or a numpy array with the
        radial distribution of cell volumes (if dim=='rt').
    """
    dz = grid.dx[-1] * c
    if dim == "xyt":
        dV = grid.dx[0] * grid.dx[1] * dz
    else:  # dim == "rt":
        r = grid.axes[0]
        dr = grid.dx[0]
        # 1D array that computes the volume of radial cells
        dV = np.pi * ((r + 0.5 * dr) ** 2 - (r - 0.5 * dr) ** 2) * dz
    return dV


def weighted_std(values, weights=None):
    """Calculate the weighted standard deviation of the given values.

    Parameters
    ----------
    values: array
        Contains the values to be analyzed

    weights : array
        Contains the weights of the values to analyze

    Returns
    -------
    A float with the value of the standard deviation
    """
    mean_val = np.average(values, weights=weights)
    std = np.sqrt(np.average((values - mean_val) ** 2, weights=weights))
    return std


def create_grid(array, axes, dim, is_envelope=True):
    """Create a lasy grid from a numpy array.

    Parameters
    ----------
    array : ndarray
        The input field array.
    axes : dict
        Dictionary with the information of the array axes.
    dim : {'xyt, 'rt'}
        The dimensionality of the array.

    Returns
    -------
    grid : Grid
        A lasy grid containing the input array.
    """
    # Create grid.
    if dim == "xyt":
        lo = (axes["x"][0], axes["y"][0], axes["t"][0])
        hi = (axes["x"][-1], axes["y"][-1], axes["t"][-1])
        npoints = (axes["x"].size, axes["y"].size, axes["t"].size)
        grid = Grid(dim, lo, hi, npoints, is_envelope=is_envelope)
        assert np.all(grid.axes[0] == axes["x"])
        assert np.all(grid.axes[1] == axes["y"])
        assert np.allclose(grid.axes[2], axes["t"], rtol=1.0e-14)
        grid.set_temporal_field(array)
    else:  # dim == "rt":
        lo = (axes["r"][0], axes["t"][0])
        hi = (axes["r"][-1], axes["t"][-1])
        npoints = (axes["r"].size, axes["t"].size)
        grid = Grid(dim, lo, hi, npoints, n_azimuthal_modes=1, is_envelope=is_envelope)
        assert np.all(grid.axes[0] == axes["r"])
        assert np.allclose(grid.axes[1], axes["t"], rtol=1.0e-14)
        grid.set_temporal_field(array)
    return grid


def export_to_z(dim, grid, omega0, z_axis=None, z0=0.0, t0=0.0, backend="NP"):
    """
    Export laser pulse to spatial domain from temporal domain (internal LASY representation).

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:
        - 'xyt': The laser pulse is represented on a 3D grid:
                 Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - 'rt' : The laser pulse is represented on a 2D grid:
                 Cylindrical (r) transversely, and temporal (t) longitudinally.

    grid : a Grid object.
        It contains a ndarrays (V/m) with
        the value of the envelope field and the associated metadata
        that defines the points at which the laser is defined.

    omega0 : scalar
        Angular frequency at which the envelope is defined.

    z_axis : 1D ndarray of doubles (optional)
        Spatial `z` axis along which the field should be reconstructed.
        If not provided, `z_axis = c * t_axis` is considered.

    z0 : scalar (optional)
        Position from which the field is produced (emitted).

    t0 : scalar (optional)
        Moment of time at which the field is produced.

    backend : string (optional)
        Backend used by axiprop (see AVAILABLE_BACKENDS in axiprop
        documentation for more information).
    """
    time_axis_indx = -1

    t_axis = grid.axes[time_axis_indx]
    if z_axis is None:
        z_axis = t_axis * c

    FieldAxprp = ScalarFieldEnvelope(omega0 / c, t_axis)

    field = grid.get_temporal_field()

    if dim == "rt":
        # Construct the propagator
        prop = []
        for m in grid.azimuthal_modes:
            prop.append(
                PropagatorResampling(
                    grid.axes[0],
                    FieldAxprp.k_freq,
                    mode=m,
                    backend=backend,
                    verbose=False,
                )
            )

        field_z = np.zeros(
            (field.shape[0], field.shape[1], z_axis.size),
            dtype=field.dtype,
        )

        # Convert the spectral image to the spatial field representation
        for i_m in range(grid.azimuthal_modes.size):
            FieldAxprp.import_field(np.transpose(field[i_m]).copy())

            field_z[i_m] = prop[i_m].t2z(FieldAxprp.Field_ft, z_axis, z0=z0, t0=t0).T

            field_z[i_m] *= np.exp(-1j * (z_axis / c + t0) * omega0)
    else:
        # Construct the propagator
        Nx, Ny, Nt = field.shape
        Lx = grid.hi[0] - grid.lo[0]
        Ly = grid.hi[1] - grid.lo[1]
        prop = PropagatorFFT2(
            (Lx, Nx),
            (Ly, Ny),
            FieldAxprp.k_freq,
            backend=backend,
            verbose=False,
        )
        # Convert the spectral image to the spatial field representation
        FieldAxprp.import_field(np.moveaxis(field, -1, 0).copy())
        field_z = prop.t2z(FieldAxprp.Field_ft, z_axis, z0=z0, t0=t0)
        field_z = np.moveaxis(field_z, 0, -1)
        field_z *= np.exp(-1j * (z_axis / c + t0) * omega0)

    return field_z


def import_from_z(dim, grid, omega0, field_z, z_axis, z0=0.0, t0=0.0, backend="NP"):
    """
    Import laser pulse from spatial domain to temporal domain (internal LASY representation).

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:
        - 'xyt': The laser pulse is represented on a 3D grid:
                 Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - 'rt' : The laser pulse is represented on a 2D grid:
                 Cylindrical (r) transversely, and temporal (t) longitudinally.

    grid : a Grid object.
        It contains an ndarray (V/m) with
        the value of the envelope field and the associated metadata
        that defines the points at which the laser is defined.

    omega0 : scalar
        Angular frequency at which the envelope is defined.

    z_axis : 1D ndarray of doubles
        Spatial `z` axis along which the field should be reconstructed.

    z0 : scalar (optional)
        Position at which the field should be recorded

    t0 : scalar (optional)
        Moment of time at which the field should be recorded

    backend : string (optional)
        Backend used by axiprop (see AVAILABLE_BACKENDS in axiprop
        documentation for more information).
    """
    z_axis_indx = -1
    t_axis = grid.axes[z_axis_indx]
    dz = z_axis[1] - z_axis[0]
    Nz = z_axis.size

    # Transform the field from spatial to wavenumber domain
    field_fft = np.fft.fft(field_z, axis=z_axis_indx, norm="forward")

    # Create the axes for wavenumbers, and for corresponding frequency
    omega = 2 * np.pi * np.fft.fftfreq(Nz, dz / c) + omega0
    k_z = omega / c

    if dim == "rt":
        # Construct the propagator
        prop = []
        for m in grid.azimuthal_modes:
            prop.append(
                PropagatorResampling(
                    grid.axes[0],
                    omega / c,
                    mode=m,
                    backend=backend,
                    verbose=False,
                )
            )

        # Convert the spectral image to the spatial field representation
        field = np.zeros(grid.shape, dtype=np.complex128)
        for i_m in range(grid.azimuthal_modes.size):
            transform_data = np.transpose(field_fft[i_m]).copy()
            transform_data *= np.exp(-1j * z_axis[0] * (k_z[:, None] - omega0 / c))
            field[i_m] = prop[i_m].z2t(transform_data, t_axis, z0=z0, t0=t0).T
            field[i_m] *= np.exp(1j * (z0 / c + t_axis) * omega0)
        grid.set_temporal_field(field)
    else:
        # Construct the propagator
        Nx, Ny, _ = grid.npoints
        Lx = grid.hi[0] - grid.lo[0]
        Ly = grid.hi[1] - grid.lo[1]
        prop = PropagatorFFT2(
            (Lx, Nx),
            (Ly, Ny),
            omega / c,
            backend=backend,
            verbose=False,
        )
        # Convert the spectral image to the spatial field representation
        transform_data = np.moveaxis(field_fft, -1, 0).copy()
        transform_data *= np.exp(-1j * z_axis[0] * (k_z[:, None, None] - omega0 / c))
        field = np.moveaxis(prop.z2t(transform_data, t_axis, z0=z0, t0=t0), 0, -1)
        field *= np.exp(1j * (z0 / c + t_axis) * omega0)
        grid.set_temporal_field(field)

def get_STC(dim, grid, tau, w0, k0):
    r"""
    Calculate the spatio-temporal coupling factors of the laser.
    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:
        - 'xyt': The laser pulse is represented on a 3D grid:
                 Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - 'rt' : The laser pulse is represented on a 2D grid:
                 Cylindrical (r) transversely, and temporal (t) longitudinally.
    grid : a Grid object.
        It contains an ndarray (V/m) with
        the value of the envelope field and the associated metadata
        that defines the points at which the laser is defined.
    tau : scalar
        Duration of the laser pulse in s.
    w0 : scalar
        Waist of laser in m.
    k0 : scalar
        Wavenumber of the field
    Return
    ----------
    STC_fac : dict of floats
        A dictionary of floats corresponding to the STC factors. The keys are:
            Phi2: Group-delayed dispersion in :math:`\Phi^{(2)}=d(\omega_0)/dt`
            phi2: Group-delayed dispersion in :math:`\phi^{(2)}=dt_0/d(\omega)`
            nu: Spatio-chirp in :math:`\nu=d(\omega_0)/dx`
            zeta: Spatio-chirp in :math:`\zeta=dx_0/d(\omega_0)`
            stc_theta_zeta: The direction of the linear spatial chirp on xoy plane\
            in rad (0 is along x)
            beta: Angular dispersion in :math:` \beta = d\theta_0/d\omega`
            pft: Pulse front tilt in :math:` p=dt/dx`
            stc_theta_beta: The direction of the linear angular chirp on xoy plane\
            in rad (0 is along x)
    All those above units and definitions are taken from
    `S. Akturk et al., Optics Express 12, 4399 (2004) <https://doi.org/10.1364/OPEX.12.004399>`__.
    """
    # Initialise the returned dictionary
    STC_fac = {
        "Phi2": 0,
        "phi2": 0,
        "nu": 0,
        "zeta": 0,
        "stc_theta_zeta": 0,
        "beta": 0,
        "pft": 0,
        "stc_theta_beta": 0,
    }
    env = grid.get_temporal_field()
    env_abs = np.abs(env)
    phi_envelop = np.unwrap(np.array(np.arctan2(env.imag, env.real)), axis=2)
    pphi_pt = (np.diff(phi_envelop, axis=2)) / (grid.dx[-1])
    # Calculate goup-delayed dispersion
    pphi_pt2 = (np.diff(pphi_pt, axis=2)) / (grid.dx[-1])
    STC_fac["Phi2"] = np.sum(pphi_pt2 * env_abs[:, :, : env_abs.shape[2] - 2]) / np.sum(
        env_abs[:, :, : env_abs.shape[2] - 2]
    )
    STC_fac["phi2"] = np.max(
        np.roots([4 * STC_fac["Phi2"], -4, tau**4 * STC_fac["Phi2"]])
    )
    # Calculate spatio- and angular dispersion
    if dim == "rt":
        pphi_ptpr = (np.diff(pphi_pt, axis=1)) / grid.dx[0]
        STC_fac["nu"] = np.sum(
            pphi_ptpr * env_abs[:, : env_abs.shape[1] - 1, : env_abs.shape[2] - 1]
        ) / np.sum(env_abs[:, : env_abs.shape[1] - 1, : env_abs.shape[2] - 1])

        # Transfer the unit from nu to zeta
        STC_fac["zeta"] = np.min(
            np.roots([4 * STC_fac["nu"], -4, STC_fac["nu"] * w0**2 * tau**2])
        )
        # No angular dispersion in 2D and the direction of spatio-chirp is certain
    if dim == "xyt":
        pphi_ptpy = (np.diff(pphi_pt, axis=1)) / grid.dx[1]
        pphi_ptpx = (np.diff(pphi_pt, axis=0)) / grid.dx[0]
        # Calculate the STC angle in XOY for spatio coupling
        theta = np.arctan2(
            pphi_ptpy[: env_abs.shape[0] - 1, : env_abs.shape[1] - 1, :],
            pphi_ptpx[: env_abs.shape[0] - 1, : env_abs.shape[1] - 1, :],
        )
        STC_fac["stc_theta_zeta"] = np.sum(
            theta
            * env_abs[
                : env_abs.shape[0] - 1, : env_abs.shape[1] - 1, : env_abs.shape[2] - 1
            ]
        ) / np.sum(
            env_abs[
                : env_abs.shape[0] - 1, : env_abs.shape[1] - 1, : env_abs.shape[2] - 1
            ]
        )
        pphi_ptpr = np.sqrt(
            pphi_ptpy[: env_abs.shape[0] - 1, : env_abs.shape[1] - 1, :] ** 2
            + pphi_ptpx[:: env_abs.shape[0] - 1, : env_abs.shape[1] - 1, :] ** 2
        )
        STC_fac["nu"] = np.sum(
            pphi_ptpr
            * env_abs[
                : env_abs.shape[0] - 1, : env_abs.shape[1] - 1, : env_abs.shape[2] - 1
            ]
        ) / np.sum(
            env_abs[
                : env_abs.shape[0] - 1, : env_abs.shape[1] - 1, : env_abs.shape[2] - 1
            ]
        )
        STC_fac["zeta"] = np.min(
            np.roots([4 * STC_fac["nu"], -4, STC_fac["nu"] * w0**2 * tau**2])
        )
        # calculate angular dispersion and pulse front tilt
        z_centroids = np.sum(grid.axes[2] * env_abs, axis=2) / np.sum(env_abs, axis=2)
        weight = np.mean(env_abs**2, axis=2)
        derivative_x = np.gradient(z_centroids, axis=0) / grid.dx[0]
        derivative_y = np.gradient(z_centroids, axis=1) / grid.dx[1]
        pft_x = np.sum(derivative_x * weight) / np.sum(weight)
        pft_y = np.sum(derivative_y * weight) / np.sum(weight)
        STC_fac["pft"] = np.sqrt((pft_x**2 + pft_y**2))
        STC_fac["stc_theta_beta"] = np.arctan2(pft_y, pft_x)
        STC_fac["beta"] = (
            np.sqrt((pft_x**2 + pft_y**2)) - STC_fac["Phi2"] * STC_fac["nu"]
        ) / k0

    return STC_fac
