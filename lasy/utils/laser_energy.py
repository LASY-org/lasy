import numpy as np
import scipy.constants as scc

def compute_laser_energy(dim, grid):
    """
    Computes the total laser energy that corresponds to the current
    envelope data. This is used mainly for normalization purposes.

    Parameters:
    -----------
    dim: string
        Dimensionality of the array. Options are:
        - 'xyt': The laser pulse is represented on a 3D grid:
                 Cartesian (x,y) transversely, and temporal (t) longitudinally.
        - 'rt' : The laser pulse is represented on a 2D grid:
                 Cylindrical (r) transversely, and temporal (t) longitudinally.

    grid: a Grid object. It contains a ndarrays (V/m) with
          the value of the envelope field and an object of type
          lasy.utils.Box that defines the points at which evaluate the laser

    Returns:
    --------
    energy: float (in Joules)
    """
    # This uses the following volume integral:
    # $E_{laser} = \int dV \;\frac{\epsilon_0}{2} | E_{env} |^2$
    # which assumes that we can average over the oscilations at the
    # specified laser wavelength.
    # This probably needs to be generalized for few-cycle laser pulses.

    envelope = grid.field
    box = grid.box

    dz = box.dx[-1]*scc.c

    if dim == 'xyt':
        dV = box.dx[0] * box.dx[1] * dz
        energy = ((dV * scc.epsilon_0 * 0.5) * \
                abs(envelope)**2).sum()
    elif dim == 'rt':
        r = box.axes[0]
        dr = box.dx[0]
        # 1D array that computes the volume of radial cells
        dV = np.pi*( (r+0.5*dr)**2 - (r-0.5*dr)**2 ) * dz
        energy = (dV[np.newaxis,:,np.newaxis] * scc.epsilon_0 * 0.5 * \
                abs(envelope[:,:,:])**2).sum()

    return energy

def normalize_energy(dim, energy, grid):
    """
    Normalize energy of the laser pulse contained in grid

    Parameters
    -----------
    energy: scalar (J)
        Energy of the laser pulse after normalization

    grid: a Grid object
        Contains value of the laser envelope and metadata
    """

    if energy is None:
        return

    current_energy = compute_laser_energy(dim, grid)
    norm_factor = (energy/current_energy)**.5
    grid.field *= norm_factor
