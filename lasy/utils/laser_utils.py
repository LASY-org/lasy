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

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
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

    Normalizes by matching laser energy to an energy given as an input.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
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

    Normalizes by matching laser field amplitude to a peak field amplitude given as an input.

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

    Normalizes by matching laser intensity to a peak intensity given as an input.

    Parameters
    ----------
    peak_intensity : scalar (W/m^2)
        Peak intensity of the laser pulse after normalization.

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
            field *= np.sqrt(peak_intensity / input_peak_intensity)
            grid.set_temporal_field(field)


def normalize_peak_fluence(peak_fluence, grid):
    """
    Normalize energy of the laser pulse contained in grid.

    Normalizes by matching laser fluence to a peak fluence given as an input.

    Parameters
    ----------
    peak_fluence : scalar (J/m^2)
        Peak fluence of the laser pulse after normalization.

    grid : a Grid object
        Contains value of the laser envelope and metadata.
    """
    if peak_fluence is not None:
        field = grid.get_temporal_field()
        fluence = get_laser_fluence(grid)
        input_peak_fluence = fluence.max()
        if input_peak_fluence == 0.0:
            print("Field is zero everywhere, normalization will be skipped")
        else:
            field *= np.sqrt(peak_fluence / input_peak_fluence)
            grid.set_temporal_field(field)


def normalize_average_intensity(average_intensity, grid):
    """
    Normalize energy of the laser pulse contained in grid.

    Normalizes by matching average laser intensity to an average intensity given as an input.

    Parameters
    ----------
    average_intensity : scalar (W/m^2)
        Average intensity of the laser pulse envelope after normalization.

    grid : a Grid object
        Contains value of the laser envelope and metadata.
    """
    if average_intensity is not None:
        field = grid.get_temporal_field()
        intensity = np.abs(epsilon_0 * field**2 / 2 * c)
        input_average_intensity = intensity.mean()
        if input_average_intensity == 0.0:
            print("Field is zero everywhere, normalization will be skipped")
        else:
            field *= np.sqrt(average_intensity / input_average_intensity)
            grid.set_temporal_field(field)


def normalize_peak_power(dim, peak_power, grid):
    """
    Normalize energy of the laser pulse contained in grid.

    Normalizes by matching laser power to a peak power given as an input.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    peak_power : scalar (W)
        Peak power of the laser pulse after normalization.

    grid : a Grid object
        Contains value of the laser envelope and metadata.
    """
    if peak_power is not None:
        power = get_laser_power(dim, grid)
        input_peak_power = power.max()
        if input_peak_power == 0.0:
            print("Field is zero everywhere, normalization will be skipped")
        else:
            field = grid.get_temporal_field()
            grid.set_temporal_field(field * np.sqrt(peak_power / input_peak_power))


def get_laser_power(dim, grid):
    r"""
    Calculate the instantaneous power of the laser along the time axis.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    grid : a Grid object.
        It contains an ndarray (V/m) with the value of the envelope field and the associated metadata that defines the points at which the laser is defined.

    Returns
    -------
    power : ndarray (W)
        The instantaneous laser power along the temporal axis.
    """
    field = grid.get_temporal_field()
    intensity = np.abs(epsilon_0 * field**2 / 2 * c)
    dz = grid.dx[-1] * c
    unit_area = get_grid_cell_volume(grid, dim) / dz
    power = intensity.sum(axis=tuple(range(intensity.ndim - 1))) * unit_area

    return power


def get_laser_fluence(grid):
    r"""
    Calculate the fluence of the laser in space.

    Parameters
    ----------
    grid : a Grid object.
        It contains an ndarray (V/m) with the value of the envelope field and the associated metadata that defines the points at which the laser is defined.

    Returns
    -------
    fluence : ndarray (J/m^2)
        The fluence of the laser pulse in space.
    """
    field = grid.get_temporal_field()
    intensity = np.abs(epsilon_0 * field**2 / 2 * c)
    fluence = np.squeeze(np.sum(intensity, axis=-1) * grid.dx[-1])

    return fluence


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


def get_spectrum(
    grid, dim, range=None, bins=20, omega0=None, method="sum", ordering="zero_center"
):
    r"""
    Get the frequency spectrum of an envelope or electric field.

    The spectrum can be calculated in three different ways, depending on the
    `method` specified by the user:

    Initially, the spectrum is calculated as the Fourier transform of the
    electric field :math:`E(t)`.

    .. math::

        \int E(t) e^{-i \omega t} dt

    neglecting the negative frequencies. If ``method=="raw"``, no further
    processing is done and the returned spectrum is a complex array with the
    same transverse dimensions as the input grid. The units are
    :math:`\mathrm{V / Hz}`.

    For the other methods, the spectral energy density is calculated as

    .. math::

        \frac{\epsilon_0 c}{2\pi} |\int E(t) e^{-i \omega t} dt| ^ 2

    If ``method=="on_axis"``, a 1D real array with on-axis value of the
    equation above is returned. The units are :math:`\mathrm{J / (rad Hz m^2)}`.

    Otherwise, if ``method=="sum"`` (default), the transverse integral of the
    spectral energy density is calculated:

    .. math::

        \frac{\epsilon_0 c}{2\pi} \int |\int E(t) e^{-i \omega t} dt| ^ 2 dx dy

    The units of this array are :math:`\mathrm{J / (rad Hz)}`

    Parameters
    ----------
    grid : a Grid object.
        It contains an ndarray with the field data from which the
        spectrum is computed, and the associated metadata. The last axis must
        be the longitudinal dimension.

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
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

    ordering : string (optional)
        Order of the frequency array and corresponding spectrum.
        Options are:
        - ``"zero_center"``: np.fft.fftshift is applied so the frequency array is monotonous with 0 at the center.
        - ``"zero_first"``: The frequency array starts with positive frequencies, and negative frequencies are at the end. The array is not monotonous. This is the default with np.fft.ifft.

    Returns
    -------
    spectrum : ndarray
        Array with the spectrum (units and array type depend on ``method``).

    omega : ndarray
        Array with the angular frequencies of the spectrum.
    """
    spectral_field, omega = grid.get_spectral_field()
    # multiply by the number of points due to np.fft.fft normalization
    spectral_field *= grid.npoints[-1]

    # Get spectrum.
    if grid.is_envelope:
        # Assume that the FFT of the envelope and the FFT of the complex
        # conjugate of the envelope do not overlap. Then we only need
        # one of them.
        assert omega0 is not None
        spectrum = 0.5 * spectral_field * grid.dx[-1]
        if method == "on_axis":
            nx, ny, _ = spectrum.shape
            spectrum = spectrum[nx // 2, ny // 2] if dim == "xyt" else spectrum[0, 0]
        omega += omega0
    else:
        spectrum = spectral_field * grid.dx[-1]
        if method == "on_axis":
            nx, ny, _ = spectrum.shape
            spectrum = spectrum[nx // 2, ny // 2] if dim == "xyt" else spectrum[0, 0]

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

    assert ordering in ["zero_first", "zero_center"]
    if ordering == "zero_center":
        omega = np.fft.fftshift(omega, axes=-1)
        spectrum = np.fft.fftshift(spectrum, axes=-1)

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

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
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

        h = field if is_hilbert else hilbert_transform(field)

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

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

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


def field_to_envelope(grid, dim, omega0=None, phase_unwrap_nd=False):
    """Convert a field to its complex envelope representation by applying a Hilbert transform.

    Parameters
    ----------
    grid : Grid
        The Grid object on which the field is replaced with an envelope.
        This object is modified by the function.

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    omega0 : scalar
        Central frequency at which the envelope will be defined.
        If None, the central frequency measured from the field is used.

    phase_unwrap_nd : boolean (optional)
        If True, the phase unwrapping is n-dimensional (2- or 3-D depending on dim).
        If False, the phase unwrapping is done in t, treating each transverse cell
        separately. This should be less accurate but faster.
        If set to True, scikit-image must be installed.

    Returns
    -------
    scalar
        If input parameter omega0 was None, return the central angular frequency.
        Otherwise, return None.
    """
    assert not grid.is_envelope

    # Get central wavelength from array
    if omega0 is None:
        _, omg0 = get_frequency(
            grid,
            dim=dim,
            is_hilbert=False,
            phase_unwrap_nd=phase_unwrap_nd,
        )
    else:
        omg0 = omega0
    # Hilbert transform
    field = hilbert_transform(grid.get_temporal_field())
    # Remove carrier frequency
    field *= np.exp(1j * omg0 * grid.axes[-1])
    # Store envelope
    grid.set_is_envelope(True)
    grid.set_temporal_field(field)

    return omg0 if omega0 is None else None


def hilbert_transform(field):
    """Make a hilbert transform of the field.

    Currently the arrays need to be flipped along t (both the input field and
    its transform) to get the imaginary part (and thus the phase) with the
    correct sign.

    Parameters
    ----------
    field : 3d numpy array
        The field whose field should be transformed.
    """
    return hilbert(field[:, :, ::-1])[:, :, ::-1]


def get_grid_cell_volume(grid, dim):
    """Get the volume of the grid cells.

    Parameters
    ----------
    grid : Grid
        The grid form which to compute the cell volume

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

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


def create_grid(array, axes, dim, is_envelope=True, position=0.0):
    """Create a lasy grid from a numpy array.

    Parameters
    ----------
    array : ndarray
        The input field array.

    axes : dict
        Dictionary with the information of the array axes.

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

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
        grid = Grid(dim, lo, hi, npoints, is_envelope=is_envelope, position=position)
        assert np.allclose(grid.axes[0], axes["x"])
        assert np.allclose(grid.axes[1], axes["y"])
        assert np.allclose(grid.axes[2], axes["t"], rtol=1.0e-14)
        assert array.ndim == 3, "Input array should be of dimension 3 [x, y, time]"
        grid.set_temporal_field(array)
    else:  # dim == "rt":
        lo = (axes["r"][0], axes["t"][0])
        hi = (axes["r"][-1], axes["t"][-1])
        npoints = (axes["r"].size, axes["t"].size)
        nm = int((array.shape[0] + 1) / 2)
        grid = Grid(
            dim,
            lo,
            hi,
            npoints,
            n_azimuthal_modes=nm,
            is_envelope=is_envelope,
        )
        assert np.all(grid.axes[0] == axes["r"])
        assert np.allclose(grid.axes[1], axes["t"], rtol=1.0e-14)
        assert array.ndim == 3, (
            "Input array should be of dimension 3 [modes, radius, time]"
        )
        grid.set_temporal_field(array)
    return grid


def export_to_z(dim, grid, omega0, z_axis=None, z0=0.0, t0=0.0, backend="NP"):
    """
    Export laser pulse to spatial domain from temporal domain (internal LASY representation).

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
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

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
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


def get_w0(grid, dim):
    r"""
    Calculate the laser waist.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    grid : a Grid object.
        It contains an ndarray (V/m) with the value of the envelope field and the associated metadata that defines the points at which the laser is defined.

    Returns
    -------
    sigma : Standard deviation of a**2 in m
    """
    field = grid.get_temporal_field()
    if dim == "xyt":
        Nx, Ny, Nt = field.shape
        A2 = (np.abs(field[Nx // 2 - 1, :, :]) ** 2).sum(-1)
        ax = grid.axes[1]
    else:
        A2 = (np.abs(field[0, :, :]) ** 2).sum(-1)
        ax = grid.axes[0]
        if ax[0] > 0:
            A2 = np.r_[A2[::-1], A2]
            ax = np.r_[-ax[::-1], ax]
        else:
            A2 = np.r_[A2[::-1][:-1], A2]
            ax = np.r_[-ax[::-1][:-1], ax]

    sigma = 2 * np.sqrt(np.average(ax**2, weights=A2))

    return sigma


def get_phi2(dim, grid):
    r"""
    Calculate the second derivative of the temporal phase of the laser.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    grid : a Grid object.
        It contains an ndarray (V/m) with the value of the envelope field and the associated metadata that defines the points at which the laser is defined.

    Returns
    -------
    phi2 : Second derivative of temporal phase :math:`\Phi^{(2)} = \frac{d\omega_0}{dt} = \frac{d^2\Phi(t)}{dt^2}` in (second^-2)
    """
    env = grid.get_temporal_field()
    env_abs2 = np.abs(env**2)
    # Calculate group-delayed dispersion
    phi_envelop = np.unwrap(np.angle(env), axis=2)
    pphi_pt = np.gradient(phi_envelop, grid.dx[-1], axis=2)
    pphi_pt2 = np.gradient(pphi_pt, grid.dx[-1], axis=2)
    phi2 = np.average(pphi_pt2, weights=env_abs2)

    return phi2


def get_zeta(dim, grid, k0):
    r"""
    Calculate the spatial chirp of the laser.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    grid : a Grid object.
        It contains an ndarray (V/m) with the value of the envelope field and the associated metadata that defines the points at which the laser is defined.

    Returns
    -------
    zeta_x, zeta_y : Spatial chirp in :math:`\zeta=\frac{dx_0}{d\omega}` (meter * second)
    nu_x, nu_y: Spatial chirp in :math:`\nu=\frac{d\omega_0}{dx}` (meter^-1 * second^-1)
    """
    assert dim == "xyt", "No spatial chirp for axis-symmetric dimension."
    w0 = get_w0(grid, dim)
    tau = 2 * get_duration(grid, dim)
    env_spec, spectral_axis = grid.get_spectral_field()
    env_spec_abs2 = np.abs(env_spec**2)
    # Get the spectral axis
    omega = spectral_axis + k0 * c
    # Calculate dx0 and dy0 in (x,y,omega) space
    weight_x_3d = np.transpose(env_spec_abs2, (2, 1, 0))
    weight_y_3d = np.transpose(env_spec_abs2, (2, 0, 1))
    weight_x_2d = np.sum(weight_x_3d, axis=2)
    weight_y_2d = np.sum(weight_y_3d, axis=2)
    # Calculate xda and yda, avoiding division by zero
    xda = np.where(
        weight_x_2d != 0, np.sum(grid.axes[0] * weight_x_3d, axis=2) / weight_x_2d, 0
    )
    yda = np.where(
        weight_y_2d != 0, np.sum(grid.axes[1] * weight_y_3d, axis=2) / weight_y_2d, 0
    )
    # Calculate spatial chirp zeta
    derivative_x_zeta = np.gradient(xda, omega, axis=0)
    derivative_y_zeta = np.gradient(yda, omega, axis=0)
    weight_x_2d = np.mean(env_spec_abs2, axis=0)
    weight_y_2d = np.mean(env_spec_abs2, axis=1)
    zeta_x = np.average(derivative_x_zeta.T, weights=weight_x_2d)
    zeta_y = np.average(derivative_y_zeta.T, weights=weight_y_2d)
    nu_x = 4 * zeta_x / (w0**2 * tau**2 + 4 * zeta_x**2)
    nu_y = 4 * zeta_y / (w0**2 * tau**2 + 4 * zeta_y**2)
    return [zeta_x, zeta_y], [nu_x, nu_y]


def get_beta(dim, grid, k0):
    r"""
    Calculate the angular dispersion of the laser.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    grid : a Grid object.
        It contains an ndarray (V/m) with
        the value of the envelope field and the associated metadata that defines the points at which the laser is defined.

    Returns
    -------
    beta_x, beta_y : Angular dispersion in :math:` \beta = \frac{d\theta_0}{d\omega}` (second)
    """
    assert dim == "xyt", "No angular chirp for axis-symmetric dimension."
    env_spec, spectral_axis = grid.get_spectral_field()
    env_spec_abs2 = np.abs(env_spec**2)
    # Get the spectral axis
    omega = spectral_axis + k0 * c
    # Calculate angular dispersion beta
    phi_envelop_abs = np.unwrap(
        np.array(np.arctan2(env_spec.imag, env_spec.real)), axis=2
    )
    angle_x = np.gradient(phi_envelop_abs, grid.dx[1], axis=1) / k0
    angle_y = np.gradient(phi_envelop_abs, grid.dx[0], axis=0) / k0
    derivative_x_beta = np.gradient(angle_y, omega, axis=2)
    derivative_y_beta = np.gradient(angle_x, omega, axis=2)
    beta_x = np.average(derivative_x_beta, weights=env_spec_abs2)
    beta_y = np.average(derivative_y_beta, weights=env_spec_abs2)
    return [beta_x, beta_y]


def get_pft(dim, grid):
    r"""
    Calculate the pulse front tilt (PFT) of the laser.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    grid : a Grid object.
        It contains an ndarray (V/m) with
        the value of the envelope field and the associated metadata
        that defines the points at which the laser is defined.

    Returns
    -------
    pft_x, pft_y : Pulse front tilt in :math:`p=\frac{dt_0}{dx}` (second * meter^-1).
    """
    assert dim == "xyt", "No pulse front tilt for cylindrical symmetry."
    env = grid.get_temporal_field()
    env_abs2 = np.abs(env**2)
    weight_xy_2d = np.mean(env_abs2, axis=2)
    z_centroids = np.sum(grid.axes[2] * env_abs2, axis=2) / np.sum(env_abs2, axis=2)
    derivative_x_pft = np.gradient(z_centroids, axis=0) / grid.dx[0]
    derivative_y_pft = np.gradient(z_centroids, axis=1) / grid.dx[1]
    pft_x = np.average(derivative_x_pft, weights=weight_xy_2d)
    pft_y = np.average(derivative_y_pft, weights=weight_xy_2d)
    return [pft_x, pft_y]


def get_propation_angle(dim, grid, k0):
    r"""
    Calculate the propagating angle of the laser.

    Parameters
    ----------
    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    grid : a Grid object.
        It contains an ndarray (V/m) with the value of the envelope field and the associated metadata that defines the points at which the laser is defined.

    Returns
    -------
    angle_x, angle_y : Propagating angle in :math:`p = \frac{k_x}{k_z}` or :math:`p = \frac{k_y}{k_z}` (in radians).
    """
    assert dim == "xyt", "Propagation is always on-axis for axis-symmetric dimension."
    env = grid.get_temporal_field()
    env_abs2 = np.abs(env**2)
    phi_envelop_abs = np.unwrap(np.angle(env), axis=2)
    pphi_px = np.gradient(phi_envelop_abs, grid.dx[1], axis=1)
    pphi_py = np.gradient(phi_envelop_abs, grid.dx[0], axis=0)
    angle_x = np.average(pphi_px, weights=env_abs2) / k0
    angle_y = np.average(pphi_py, weights=env_abs2) / k0
    return [angle_x, angle_y]


def get_spectral_phase(grid, dim, omega0, method="sum", ordering="zero_center"):
    r"""
    Calculate the spectral phase of a pulse in a given grid.

    Depending on the chosen calculation method, the spectral phase can be calculated in two different ways.

    If `method==sum` (default), the spectral field of the laser is spatially integrated before extracting the phase, i.e.

    .. math::

        \varphi(\omega) = \arg\left( \int E(x,y,\omega) dx dy \right)

    If `method==on-axis`, the on-axis spectral phase is calculated:

    .. math::

        \varphi(\omega) = \arg\left( E(x=0,y=0,\omega) \right)

    Parameters
    ----------
    grid : Grid
        The grid with the field to analyze.

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    omega0 : float
        Central angular frequency of the field

    method : string, optional
        Method of calculating the spectral phase. Options are:

        - ``'sum'``: Calculates the spectral phase of the spatially summed field (default).
        - ``'on-axis'``: Calculates the on-axis spectral phase.

    ordering : string (optional)
        Order of the frequency array and corresponding spectral phase.
        Options are:
        - ``"zero_center"``: np.fft.fftshift is applied so the frequency array is monotonous with 0 at the center.
        - ``"zero_first"``: The frequency array starts with positive frequencies, and negative frequencies are at the end. The array is not monotonous. This is the default with np.fft.ifft.


    Returns
    -------
    phase: ndarray of floats (1D)
        Spectral phase of the pulse in the specified units and calculation method (in rad/s)

    omega: ndarray of floats (1D)
        Angular frequencies at which the phase is defined (in rad/s)

    """
    # Field must be envelope
    assert grid.is_envelope

    # get the spectral field
    field_spectral, omega = grid.get_spectral_field()

    # if method=='on-axis' get the on-axis field envelope, and calculate its phase
    assert method in ["on-axis", "sum"]
    if method == "on-axis":
        if dim == "xyt":
            Nx = grid.npoints[0]
            Ny = grid.npoints[1]
            phase = np.angle(field_spectral[Nx // 2, Ny // 2, :])
        else:  # dim=='rt'
            phase = np.angle(field_spectral[0, 0, :])

    # if method=='sum' integrate the field spatially before getting the phase from it
    else:  # method='sum'
        # grid cell volume required for integration
        dV = get_grid_cell_volume(grid, dim)

        if dim == "xyt":
            summed_field = np.sum(field_spectral * dV, axis=(0, 1))
        else:  # dim=='rt'
            summed_field = np.sum(field_spectral * dV[None, :, None], axis=(0, 1))

        phase = np.angle(summed_field)

    # create omega array (angular frequencies)
    assert ordering in ["zero_first", "zero_center"]
    if ordering == "zero_center":
        omega = np.fft.fftshift(omega, axes=-1)
        phase = np.fft.fftshift(phase, axes=-1)
    omega += omega0

    # unwrap the phase
    phase = np.unwrap(phase)

    # return the phase and omega arrays
    return phase, omega


def get_dispersion(grid, dim, omega0, order, omega_eval=None, method="sum"):
    r"""
    Calculate the n-th order dispersion polynomial of the laser.

    .. math::
        \Phi^{(n)} = \frac{\partial^n \phi(\omega)}{\partial \omega^n}

    where n is the order to which the disperison is calculated (in s^n/rad).

    E.g. order=1, calculates the group delay (GD), order=2 calculates group delay dispersion (GDD) and order=3 calculates the third-order dispersion (TOD).

    Parameters
    ----------
    grid : Grid
        The grid with the field to analyze.

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    omega0 : float
        Angular frequency at which the laser envelope is defined.

    order : integer
        Dispersion polynomial order that should be calculated.

    omega_eval : float, optional
        Central angular frequency at which the dispersion polynomial is calculated, if `None`, `omega0` is used.

    method : string, optional
        Method of retrieving the phase that is used for calculating the dispersion. Options are:

        - ``'sum'``: Calculates the spectral phase of the spatially summed field (default).
        - ``'on-axis'``: Calculates the on-axis spectral phase.

    Returns
    -------
    disp: ndarray of floats (1D)
        n-th order dispersion over the entire spectral range (in s^n/rad)

    disp0: float
        n-th order dispersion at the center frequency (in s^n/rad)

    """
    # calculate the spectral phase of the laser pulse
    phase, omega = get_spectral_phase(grid, dim, method=method, omega0=omega0)

    # calculate the n-th order derivative wrt. angular frequency
    disp = np.gradient(phase, omega, axis=-1)
    for _ in range(order - 1):
        disp = np.gradient(disp, omega, axis=-1)

    # get the dispersion at the specified frequency of the envelope's frequency
    omega_eval = omega_eval if omega_eval is not None else omega0

    disp0 = interp1d(omega, disp, bounds_error=True)(omega_eval)

    return disp, disp0


def get_bandwidth(grid, dim, method="sum", level=None, unit="rad/s", omega0=None):
    """Calculate the spectral width of a pulse in a given grid.

    By default, the bandwidth is calculated as the rms width of the spatially summed spectrum, in rad/s.
    Optionally, the bandwidth can also be calculated on-axis, at a given intensity level or in meters.

    Parameters
    ----------
    grid : Grid
        The grid with the envelope to analyze.

    dim : string
        Dimensionality of the array. Options are:

        - ``'xyt'``: The laser pulse is represented on a 3D grid:
                    Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - ``'rt'`` : The laser pulse is represented on a 2D grid:
                    Cylindrical (r) transversely, and temporal (t) longitudinally.

    method : string, optional
        Method to calculate the bandwidth. Options are:

        - ``'sum'``: Calculates the width of the spatially summed spectrum.
        - ``'on-axis'``: Calculates the width of the on-axis spectrum.

    level : float, optional
        Intensity level at which the bandwidth is calculated. If None, i.e. by default,
        the bandwidth is calculated as the rms width of the spectral intensity.

    unit : string, optional
        Unit in which the bandwidth should be returned.
        Options are:

        - ``'rad/s'``: Radians per second.
        - ``'m'``: meters.

    omega0 : float, optional
        Central angular frequency of the pulse.
        Only required if ``unit='m'``, i.e. the bandwidth should be converted to meters.

    Returns
    -------
    float
        Spectral bandwidth of the pulse in the specified units and calculation method.
    """
    # Get the volume of each grid cell and spectral field
    dV = get_grid_cell_volume(grid, dim)
    field, omega = grid.get_spectral_field()

    # Choose axis along which to calculate the bandwidth
    if unit == "m":  # convert omega to wavelength
        assert omega0, "'omega0' must be provided to calculate bandwidth in meters."
        width_axis = 2 * np.pi * c / (omega + omega0)
    else:  # keep omega as that axis
        width_axis = omega

    # Calculate weights of each grid cell (amplitude of the field).
    if dim == "xyt":
        spectral_intensity = np.abs(field) ** 2 * dV
    else:  # dim == "rt":
        spectral_intensity = np.abs(field) ** 2 * dV[np.newaxis, :, np.newaxis]

    # Selecte the method to calculate the bandwidth
    if method == "sum":
        spectral_intensity = np.sum(spectral_intensity, axis=(0, 1))
    else:
        if dim == "xyt":
            spectral_intensity = spectral_intensity[
                grid.npoints[0] // 2, grid.npoints[1] // 2, :
            ]
        else:  # dim=='rt'
            spectral_intensity = spectral_intensity[0, 0, :]

    if level:
        # sort omega/wavelength axis and spectral intensity
        order = np.argsort(width_axis)
        width_axis = width_axis[order]
        spectral_intensity = spectral_intensity[order]

        # find intensity threshold
        threshold = np.max(spectral_intensity) * level

        # find indices that mark the range in which spectral intensity >= threshold
        idcs = np.where(spectral_intensity >= threshold)[0]
        i_min, i_max = idcs[0], idcs[-1]

        # calculate positions of lower and upper bounds
        lower_bound = np.interp(
            threshold,
            spectral_intensity[i_min - 1 : i_min + 1],
            width_axis[i_min - 1 : i_min + 1],
        )
        upper_bound = np.interp(
            threshold,
            spectral_intensity[i_max : i_max + 2][::-1],
            width_axis[i_max : i_max + 2][::-1],
        )

        # calculate bandwidth
        bandwidth = upper_bound - lower_bound

    else:  # default case, calculate rms width
        bandwidth = weighted_std(width_axis, spectral_intensity)

    return bandwidth
