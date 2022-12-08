import numpy as np
import scipy.constants as scc

def compute_laser_energy(envelope, box):
    """
    Computes the total laser energy that corresponds to the current
    envelope data. This is used mainly for normalization purposes.

    Parameters:
    -----------
    envelope: ndarrays (V/m)
        Contains the value of the envelope field

    box: an object of type lasy.utils.Box
        Defines the points at which evaluate the laser

    Returns:
    --------
    energy: float (in Joules)
    """
    # This uses the following volume integral:
    # $E_{laser} = \int dV \;\frac{\epsilon_0}{2} | E_{env} |^2$
    # which assumes that we can average over the oscilations at the
    # specified laser wavelength.
    # This probably needs to be generalized for few-cycle laser pulses.
    dz = box.dx[-1] * scc.c # (Last dimension is time)

    if box.dim == 'xyt':
        dV = box.dx[0] * box.dx[1] * dz
        energy = ((dV * scc.epsilon_0 * 0.5) * \
                abs(envelope)**2).sum()
    elif box.dim == 'rt':
        r = box.axes[0]
        dr = box.dx[0]
        # 1D array that computes the volume of radial cells
        dV = np.pi*( (r+0.5*dr)**2 - (r-0.5*dr)**2 ) * dz
        energy = (dV[:,np.newaxis] * scc.epsilon_0 * 0.5 * \
                abs(envelope[0,:,:])**2).sum()
        # TODO: generalize for higher-order modes
        assert envelope.shape[0] == 1

    return energy
