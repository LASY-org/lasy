import copy
import numpy as np


def gerchberg_saxton_algo(
    laserPos1,
    laserPos2,
    dz,
    condition="max_itterations",
    max_itterations=10,
    amplitude_error=1.0,
    debug=False,
):
    """
    Given two laser profiles and a distance betweent them, calculate
    the spatial phase profile of the laser in both planes using the
    Gerchberg-Saxton algorithm.

    Parameters
    ----------
    laserPos1, laserPos2 : instance of Laser

    dz : float (meters)
        Distance between the two laser pulses

    condition : string
        The condition on which to stop the itterative loop.
        Options are: 'max_itterations', stopping after a fixed number of
        itterations or 'amplitude_error', waiting for the residual of
        the e-field amplitude from laserPos1 to fall below 'amplitude_error'

    max_itterations : int
        Maximum number of itterations to perform

    amplitude_error : float
        Residual value for amplitude given as a fraction of the maximum
        initial amplitude value for laserPos1.

    Returns
    -------
    phase1, phase2 : ndarray of floats (rad)
        Phase profile of the laser pulse
    """
    laser1 = copy.deepcopy(laserPos1)
    laser2 = copy.deepcopy(laserPos2)
    amp1 = np.abs(laser1.field.field)
    amp1_summed = np.sum(amp1)
    amp2 = np.abs(laser2.field.field)
    phase1 = np.zeros_like(amp1)

    if condition == "max_itterations":
        breakout = lambda i: i < max_itterations
        cond = 0
    elif condition == "amplitude_error":
        breakout = lambda amp: amp / amp1_summed > amplitude_error
        cond = 9e30

    i = 0
    while breakout(cond):
        laser1.field.field = amp1 * np.exp(1j * phase1)
        laser1.propagate(dz)

        phase2 = np.angle(laser1.field.field)
        laser2.field.field = amp2 * np.exp(1j * phase2)
        laser2.propagate(-dz)

        phase1 = np.angle(laser2.field.field)
        amp1_calc = np.abs(laser2.field.field)
        amp_error_summed = np.sum(np.abs(amp1_calc) - amp1)
        if debug:
            i += 1
            print(
                "Itteration %i : Amplitude Error (summed) = %.2e"
                % (i, amp_error_summed / amp1_summed)
            )
        if condition == "max_itterations":
            cond += 1
        else:
            cond = amp_error_summed

    return phase1, phase2, amp_error_summed
