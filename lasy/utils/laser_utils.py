import numpy as np
from scipy.constants import c, epsilon_0, e, m_e
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from skimage.restoration import unwrap_phase

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
        It contains a ndarrays (V/m) with
        the value of the envelope field and the associated metadata
        that defines the points at which evaluate the laser

    Returns
    -------
    energy: float (in Joules)
    """
    # This uses the following volume integral:
    # $E_{laser} = \int dV \;\frac{\epsilon_0}{2} | E_{env} |^2$
    # which assumes that we can average over the oscilations at the
    # specified laser wavelength.
    # This probably needs to be generalized for few-cycle laser pulses.

    envelope = grid.field

    dV = get_grid_cell_volume(grid, dim)

    if dim == "xyt":
        energy = ((dV * epsilon_0 * 0.5) * abs(envelope) ** 2).sum()
    elif dim == "rt":
        energy = (
            dV[np.newaxis, :, np.newaxis]
            * epsilon_0
            * 0.5
            * abs(envelope[:, :, :]) ** 2
        ).sum()

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
        Energy of the laser pulse after normalization

    grid: a Grid object
        Contains value of the laser envelope and metadata
    """
    if energy is None:
        return

    current_energy = compute_laser_energy(dim, grid)
    norm_factor = (energy / current_energy) ** 0.5
    grid.field *= norm_factor


def normalize_peak_field_amplitude(amplitude, grid):
    """
    Normalize energy of the laser pulse contained in grid.

    Parameters
    ----------
    amplitude : scalar (V/m)
        Peak field amplitude of the laser pulse after normalization

    grid : a Grid object
        Contains value of the laser envelope and metadata
    """
    if amplitude is None:
        return
    grid.field *= amplitude / np.abs(grid.field).max()


def normalize_peak_intensity(peak_intensity, grid):
    """
    Normalize energy of the laser pulse contained in grid.

    Parameters
    ----------
    peak_intensity : scalar (W/m^2)
        Peak field amplitude of the laser pulse after normalization

    grid : a Grid object
        Contains value of the laser envelope and metadata
    """
    if peak_intensity is None:
        return
    intensity = np.abs(epsilon_0 * grid.field**2 / 2 * c)
    input_peak_intensity = intensity.max()

    grid.field *= np.sqrt(peak_intensity / input_peak_intensity)


def get_full_field(laser, theta=0, slice=0, slice_axis="x", Nt=None):
    """
    Reconstruct the laser pulse with carrier frequency on the default grid.

    Parameters
    ----------
    theta : float (rad) (optional)
        Azimuthal angle
    slice : float (optional)
        Normalised position of the slice from -0.5 to 0.5
    Nt: int (optional)
        Number of time points on which field should be sampled. If is None,
        the orignal time grid is used, otherwise field is interpolated on a
        new grid.

    Returns
    -------
        Et : ndarray (V/m)
            The reconstructed field, with shape (Nr, Nt) (for `rt`)
            or (Nx, Nt) (for `xyt`)
        extent : ndarray (Tmin, Tmax, Xmin, Xmax)
            Physical extent of the reconstructed field
    """
    omega0 = laser.profile.omega0
    env = laser.grid.field.copy()
    time_axis = laser.grid.axes[-1]

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


def get_spectrum(grid, dim, bins=20, range=None, omega0=None, phase_unwrap_1d=None):
    """
    Get the the frequency spectrum of an envelope.

    Parameters
    ----------
    grid : a Grid object.
        It contains a ndarrays with the field data from which the
        spectrum is computed, and the associated metadata. The last axis must
        be the longitudinal dimension.
        Can be the full electric field or the envelope.

    dim : string (optional)
        Dimensionality of the array. Only used if is_envelope is False.
        Options are:

        - 'xyt': The laser pulse is represented on a 3D grid:
                 Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - 'rt' : The laser pulse is represented on a 2D grid:
                 Cylindrical (r) transversely, and temporal (t) longitudinally.

    bins : int (optional)
        Number of bins of the spectrum.

    range : list of float (optional)
        List of two values indicating the minimum and maximum frequency of the
        spectrum.

    omega0 : scalar
        Angular frequency at which the envelope is defined.
        Required if an only if is_envelope is True.

    phase_unwrap_1d : boolean (optional)
        Whether the phase unwrapping is done in 1D.
        This is not recommended, as the unwrapping will not be accurate,
        but it might be the only practical solution when dim is 'xyt'.

    Returns
    -------
    spectrum : ndarray of doubles
        Array with the spectrum amplitudes.

    omega_spectrum : scalar
        Array with the frequencies of the spectrum.
    """
    # Get frequency array.
    omega, central_omega = get_frequency(
        grid=grid,
        dim=dim,
        is_envelope=True,
        omega0=omega0,
        phase_unwrap_1d=phase_unwrap_1d,
    )
    # Calculate weights of each frequency (amplitude of the field).
    dV = get_grid_cell_volume(grid, dim)
    if dim == "xyt":
        weights = np.abs(grid.field) * dV
    elif dim == "rt":
        weights = np.abs(grid.field) * dV[np.newaxis, :, np.newaxis]
    # Get weighted spectrum.
    # Neglects the 2 first and last time slices, whose values seems to be
    # slightly off (maybe due to lower order derivative at the edges).
    spectrum, edges = np.histogram(
        a=np.squeeze(omega)[..., 2:-2],
        weights=np.squeeze(weights[..., 2:-2]),
        bins=bins,
        range=range,
    )
    omega_spectrum = edges[1:] - (edges[1] - edges[0]) / 2
    return spectrum, omega_spectrum


def get_frequency(
    grid,
    dim=None,
    is_envelope=True,
    is_hilbert=False,
    omega0=None,
    phase_unwrap_1d=None,
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
        Dimensionality of the array. Only used if is_envelope is False.
        Options are:

        - 'xyt': The laser pulse is represented on a 3D grid:
                 Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - 'rt' : The laser pulse is represented on a 2D grid:
                 Cylindrical (r) transversely, and temporal (t) longitudinally.

    is_envelope : bool (optional)
        Whether the field provided uses the envelope representation, as used
        internally in lasy. If False, field is assumed to represent the
        the electric field.

    is_hilbert : boolean (optional)
        If True, the field argument is assumed to be a Hilbert transform, and
        is used through the computation. Otherwise, the Hilbert transform is
        calculated in the function.

    omega0 : scalar
        Angular frequency at which the envelope is defined.
        Required if an only if is_envelope is True.

    phase_unwrap_1d : boolean (optional)
        Whether the phase unwrapping is done in 1D.
        This is not recommended, as the unwrapping will not be accurate,
        but it might be the only practical solution when dim is 'xyt'.

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
    # Assumes t is last dimension!
    if is_envelope:
        assert omega0 is not None
        phase = np.unwrap(np.angle(grid.field))
        omega = omega0 + np.gradient(-phase, grid.axes[-1], axis=-1, edge_order=2)
        central_omega = np.average(omega, weights=np.abs(grid.field))
    else:
        assert dim in ["xyt", "rt"]
        if dim == "xyt" and not phase_unwrap_1d:
            print("WARNING: using 3D phase unwrapping, this can be expensive")

        h = grid.field if is_hilbert else hilbert_transform(grid)
        h = np.squeeze(grid.field)
        if phase_unwrap_1d:
            phase = np.unwrap(np.angle(h))
        else:
            phase = unwrap_phase(np.angle(h))
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
    """Get envelope duration, measured as RMS.

    Parameters
    ----------
    grid : Grid
        The grid with the envelope to analyze.
    dim : str
        Dimensionality of the grid.

    Returns
    -------
    float
        RMS duration of the envelope in seconds.
    """
    # Calculate weights of each frequency (amplitude of the field).
    dV = get_grid_cell_volume(grid, dim)
    if dim == "xyt":
        weights = np.abs(grid.field) * dV
    elif dim == "rt":
        weights = np.abs(grid.field) * dV[np.newaxis, :, np.newaxis]
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
    omega, _ = get_frequency(grid, is_envelope=True, omega0=omega0)
    return -1j * e * grid.field / (m_e * omega * c)


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
    if direct:
        A = (
            -np.gradient(grid.field, grid.axes[-1], axis=-1, edge_order=2)
            + 1j * omega0 * grid.field
        )
        return m_e * c / e * A
    else:
        omega, _ = get_frequency(grid, is_envelope=True, omega0=omega0)
        return 1j * m_e * omega * c * grid.field / e


def field_to_envelope(grid, dim, phase_unwrap_1d):
    """Get the complex envelope of a field by applying a Hilbert transform.

    Parameters
    ----------
    grid : Grid
        The field from which to extract the envelope.
    dim : str
        Dimensions of the field. Possible values are `'xyt'` or `'rt'`.
    phase_unwrap_1d : bool
        Whether the phase unwrapping is done in 1D. This is not recommended,
        as the unwrapping will not be accurate, but it might be the only
        practical solution when dim is 'xyt'.

    Returns
    -------
    tuple
        A tuple with the envelope array and the central wavelength.
    """
    # hilbert transform needs inverted time axis.
    grid.field = hilbert_transform(grid)

    # Get central wavelength from array
    omg_h, omg0_h = get_frequency(
        grid,
        dim=dim,
        is_envelope=False,
        is_hilbert=True,
        phase_unwrap_1d=phase_unwrap_1d,
    )
    grid.field *= np.exp(1j * omg0_h * grid.axes[-1])

    return grid, omg0_h


def hilbert_transform(grid):
    """Make a hilbert transform of the grid field.

    Currently the arrays need to be flipped along t (both the input field and
    its transform) to get the imaginary part (and thus the phase) with the
    correct sign.

    Parameters
    ----------
    grid : Grid
        The lasy grid whose field should be transformed.
    """
    return hilbert(grid.field[:, :, ::-1])[:, :, ::-1]


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
    elif dim == "rt":
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


def create_grid(array, axes, dim):
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
        grid = Grid(dim, lo, hi, npoints)
        assert np.all(grid.axes[0] == axes["x"])
        assert np.all(grid.axes[1] == axes["y"])
        assert np.all(grid.axes[2] == axes["t"])
        assert grid.field.shape == array.shape
        grid.field = array
    else:  # dim == "rt":
        lo = (axes["r"][0], axes["t"][0])
        hi = (axes["r"][-1], axes["t"][-1])
        npoints = (axes["r"].size, axes["t"].size)
        grid = Grid(dim, lo, hi, npoints, n_azimuthal_modes=1)
        assert np.all(grid.axes[0] == axes["r"])
        assert np.allclose(grid.axes[1], axes["t"], rtol=1.0e-14)
        assert grid.field.shape == array[np.newaxis].shape
        grid.field = array[np.newaxis]
    return grid
