import numpy as np
import scipy.constants as scc

def compute_laser_energy(grid):
    """
    Computes the total laser energy that corresponds to the current
    envelope data. This is used mainly for normalization purposes.

    Parameters:
    -----------
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

    if box.dim == 'xyt':
        dV = box.dx[0] * box.dx[1] * dz
        energy = ((dV * scc.epsilon_0 * 0.5) * \
                abs(envelope)**2).sum()
    elif box.dim == 'rt':
        r = box.axes[0]
        dr = box.dx[0]
        # 1D array that computes the volume of radial cells
        dV = np.pi*( (r+0.5*dr)**2 - (r-0.5*dr)**2 ) * dz
        energy = (dV[np.newaxis,:,np.newaxis] * scc.epsilon_0 * 0.5 * \
                abs(envelope[:,:,:])**2).sum()

    return energy

def normalize_energy(energy, grid):
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

    current_energy = compute_laser_energy(grid)
    norm_factor = (energy/current_energy)**.5
    grid.field *= norm_factor
