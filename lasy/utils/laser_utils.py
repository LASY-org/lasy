import numpy as np
from scipy.constants import c, epsilon_0
from scipy.interpolate import interp1d


def compute_laser_energy(dim, grid):
    """
    Computes the total laser energy that corresponds to the current
    envelope data. This is used mainly for normalization purposes.

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
        the value of the envelope field and an object of type
        lasy.utils.Box that defines the points at which evaluate the laser

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
    box = grid.box

    dz = box.dx[-1] * c

    if dim == "xyt":
        dV = box.dx[0] * box.dx[1] * dz
        energy = ((dV * epsilon_0 * 0.5) * abs(envelope) ** 2).sum()
    elif dim == "rt":
        r = box.axes[0]
        dr = box.dx[0]
        # 1D array that computes the volume of radial cells
        dV = np.pi * ((r + 0.5 * dr) ** 2 - (r - 0.5 * dr) ** 2) * dz
        energy = (
            dV[np.newaxis, :, np.newaxis]
            * epsilon_0
            * 0.5
            * abs(envelope[:, :, :]) ** 2
        ).sum()

    return energy


def normalize_energy(dim, energy, grid):
    """
    Normalize energy of the laser pulse contained in grid

    Parameters
    -----------
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
    Normalize energy of the laser pulse contained in grid

    Parameters
    ----------
    amplitude : scalar (V/m)
        Peak field amplitude of the laser pulse after normalization

    grid : a Grid object
        Contains value of the laser envelope and metadata
    """

    if amplitude is None:
        return
    grid.field = grid.field / np.abs(grid.field).max() * amplitude


def normalize_peak_intensity(peak_intensity, grid):
    """
    Normalize energy of the laser pulse contained in grid

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
    Reconstruct the laser pulse with carrier frequency on the default grid

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
    field = laser.field.field.copy()
    time_axis = laser.box.axes[-1]

    if laser.dim == "rt":
        azimuthal_phase = np.exp(-1j * laser.box.azimuthal_modes * theta)
        field_upper = field * azimuthal_phase[:, None, None]
        field_upper = field_upper.sum(0)
        azimuthal_phase = np.exp(1j * laser.box.azimuthal_modes * theta)
        field_lower = field * azimuthal_phase[:, None, None]
        field_lower = field_lower.sum(0)
        field = np.vstack((field_lower[::-1][:-1], field_upper))
    elif slice_axis == "x":
        Nx_middle = field.shape[0] // 2 - 1
        Nx_slice = int((1 + slice) * Nx_middle)
        field = field[Nx_slice, :]
    elif slice_axis == "y":
        Ny_middle = field.shape[1] // 2 - 1
        Ny_slice = int((1 + slice) * Ny_middle)
        field = field[:, Ny_slice, :]
    else:
        return None

    if Nt is not None:
        Nr = field.shape[0]
        time_axis_new = np.linspace(laser.box.lo[-1], laser.box.hi[-1], Nt)
        field_new = np.zeros((Nr, Nt), dtype=field.dtype)

        for ir in range(Nr):
            interp_fu_abs = interp1d(time_axis, np.abs(field[ir]))
            slice_abs = interp_fu_abs(time_axis_new)
            interp_fu_angl = interp1d(time_axis, np.unwrap(np.angle(field[ir])))
            slice_angl = interp_fu_angl(time_axis_new)
            field_new[ir] = slice_abs * np.exp(1j * slice_angl)

        time_axis = time_axis_new
        field = field_new

    field *= np.exp(-1j * omega0 * time_axis[None, :])
    field = np.real(field)

    if laser.dim == "rt":
        ext = np.array(
            [laser.box.lo[-1], laser.box.hi[-1], -laser.box.hi[0], laser.box.hi[0]]
        )
    else:
        ext = np.array(
            [laser.box.lo[-1], laser.box.hi[-1], laser.box.lo[0], laser.box.hi[0]]
        )

    return field, ext
