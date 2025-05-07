import copy

import numpy as np


def gerchberg_saxton_algo(
    laserPos1,
    laserPos2,
    dz,
    condition="max_iterations",
    max_iterations=10,
    amplitude_error=1.0,
    debug=False,
):
    """
    Implement the Gerchberg-Saxton Algorithm.

    Given two laser profiles and a distance betweent them, calculate
    the spatial phase profile of the laser in both planes using the
    Gerchberg-Saxton algorithm. Returns the phase of the laser in both
    planes.

    Parameters
    ----------
    laserPos1, laserPos2 : instance of Laser

    dz : float (meters)
        Distance between the two laser pulses

    condition : string
        The condition on which to stop the iterative loop.
        Options are: 'max_iterations', stopping after a fixed number of
        iterations or 'amplitude_error', waiting for the residual of
        the e-field amplitude from laserPos1 to fall below 'amplitude_error'

    max_iterations : int
        Maximum number of iterations to perform

    amplitude_error : float
        Residual value for amplitude given as a fraction of the maximum
        initial amplitude value for laserPos1.

    debug : boolean (default: False)
        if True, the error at each iteration is printed to standard output

    Returns
    -------
    phase1, phase2 : ndarray of floats (rad)
        Phase profiles of the laser pulse at the locations where
        laserPos1 and laserPos2 are defined.
    """
    laser1 = copy.deepcopy(laserPos1)
    laser2 = copy.deepcopy(laserPos2)
    amp1 = np.abs(laser1.grid.get_temporal_field())
    amp1_summed = np.sum(amp1)
    amp2 = np.abs(laser2.grid.get_temporal_field())
    phase1 = np.zeros_like(amp1)

    if condition == "max_iterations":

        def breakout(i):
            return i < max_iterations

        cond = 0
    elif condition == "amplitude_error":

        def breakout(amp):
            return amp / amp1_summed > amplitude_error

        cond = 9e30

    i = 0
    while breakout(cond):
        laser1.grid.set_temporal_field(amp1 * np.exp(1j * phase1))
        laser1.propagate(dz, verbose=False)

        phase2 = np.angle(laser1.grid.get_temporal_field())
        laser2.grid.set_temporal_field(amp2 * np.exp(1j * phase2))
        laser2.propagate(-dz, verbose=False)

        field2 = laser2.grid.get_temporal_field()
        phase1 = np.angle(field2)
        amp1_calc = np.abs(field2)
        amp_error_summed = np.sum(np.abs(amp1_calc) - amp1)
        if debug:
            i += 1
            print(
                "Iteration %i : Amplitude Error (summed) = %.2e"
                % (i, amp_error_summed / amp1_summed)
            )
        if condition == "max_iterations":
            cond += 1
        else:
            cond = amp_error_summed

    return phase1, phase2, amp_error_summed
