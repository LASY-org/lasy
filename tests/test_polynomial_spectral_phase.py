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

    # Only compare data in the post-pulse region (t>t_peak+tau),
    # where the stationary phase approximation is valid
    E_compare = abs( E_after[50, 50, t>t_peak + tau] )
    t = t[t>t_peak + tau]

    E0 = abs(E_before[50,50]).max()
    prediction = abs(
            2* E0 * tau/(8*tod*t)**.25
            * np.exp( - tau**2*t/(2*tod) )
            * np.cos( 2*t/3 * (2*t/tod)**.5
            - tau**4/(8*tod)*(2*t/tod)**.5 - np.pi/4 )
        )

    # Compare the on-axis field with the analytical formula
    tol = 3e-2
    assert np.all( abs(E_compare - prediction)/abs(E0) < tol )
