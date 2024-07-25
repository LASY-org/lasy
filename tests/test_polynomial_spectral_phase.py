# -*- coding: utf-8 -*-
"""
This test checks the implementation of the polynomial spectral phase
by initializing a Gaussian pulse (with flat spectral phase),
adding spectral phase to it, and checking the corresponding
temporal shape of the laser pulse again analytical formulas.
"""

import numpy as np

from lasy.laser import Laser
from lasy.optical_elements import PolynomialSpectralPhase
from lasy.profiles.gaussian_profile import GaussianProfile

# Laser parameters
wavelength = 0.8e-6
w0 = 5.0e-6  # m
pol = (1, 0)
laser_energy = 1.0  # J
t_peak = 0.0e-15  # s
tau = 15.0e-15  # s
gaussian_profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)

# Grid parameters
dim = "xyt"
lo = (-12e-6, -12e-6, -100e-15)
hi = (+12e-6, +12e-6, +100e-15)
npoints = (100, 100, 200)


def test_gdd():
    """
    Add GDD to the laser and compare the on-axis field with the
    analytical formula for a Gaussian pulse with GDD.
    """
    gdd = 200e-30
    dazzler = PolynomialSpectralPhase(gdd=gdd)

    # Initialize the laser
    dim = "xyt"
    lo = (-12e-6, -12e-6, -100e-15)
    hi = (+12e-6, +12e-6, +100e-15)
    npoints = (100, 100, 200)
    laser = Laser(dim, lo, hi, npoints, gaussian_profile)

    # Get field before and after dazzler
    E_before = laser.grid.get_temporal_field()
    laser.apply_optics(dazzler)
    E_after = laser.grid.get_temporal_field()

    # Compute the analtical expression in real space for a Gaussian
    t = np.linspace(laser.grid.lo[-1], laser.grid.hi[-1], laser.grid.npoints[-1])
    E0 = E_before.max()
    stretch_factor = 1 - 2j * gdd / tau**2
    E_analytical = (
        E0 * np.exp(-1.0 / stretch_factor * (t / tau) ** 2) / stretch_factor**0.5
    )

    # Compare the on-axis field with the analytical formula
    tol = 1.2e-3
    assert np.all(
        abs(E_after[50, 50, :] - E_analytical) / abs(E_analytical).max() < tol
    )


def test_tod():
    """
    Add TOD to the laser and compare the on-axis field with the
    analytical formula from the stationary phase approximation.
    """
    tod = 5e-42
    dazzler = PolynomialSpectralPhase(tod=tod)

    # Initialize the laser
    dim = "xyt"
    lo = (-12e-6, -12e-6, -50e-15)
    hi = (+12e-6, +12e-6, +250e-15)
    npoints = (100, 100, 400)
    laser = Laser(dim, lo, hi, npoints, gaussian_profile)
    t = np.linspace(laser.grid.lo[-1], laser.grid.hi[-1], laser.grid.npoints[-1])

    # Get field before and after dazzler
    E_before = laser.grid.get_temporal_field()
    laser.apply_optics(dazzler)
    E_after = laser.grid.get_temporal_field()

    # TOD typically results in several post-pulses with decreasing amplitude.
    # The stationary phase approximation cannot predict the oscillations that corresponds
    # to discrete post-pulses, but it predicts their average, decreasing amplitude.
    # Thus, here we extract the amplitude of the peaks of the post-pulses
    # and multiply by 0.5 to find the average amplitude.
    on_axis_env = abs(E_after[50, 50, :])
    peak_positions = []
    for i in range(1, len(on_axis_env) - 1):
        if (on_axis_env[i] > on_axis_env[i + 1]) and (
            on_axis_env[i] > on_axis_env[i - 1]
        ):
            peak_positions.append(i)
    # Skip first maximum, for which the stationary phase is not adapted
    peak_positions = np.array(peak_positions[1:])
    assert len(peak_positions) > 10  # Check that there are multiple post-pulses
    avg_amplitude = on_axis_env[peak_positions] * 0.5

    # Compute the analtical expression using the stationary phase approximation
    E0 = E_before.max()

    def stationary_phase_approx(t):
        w_stat = (
            2 * t * (t > 0) / tod
        ) ** 0.5  # Omega for which the derivative of the phase is 0
        return abs(
            E0
            * np.exp(-(w_stat**2) * tau**2 / 4)
            / (1 + 2j * tod * w_stat / tau**2) ** 0.5
        )

    predictions = stationary_phase_approx(t[peak_positions])

    # Compare the on-axis field with the analytical formula
    tol = 1.3e-3
    assert np.all(abs(avg_amplitude - predictions) / abs(E0) < tol)
