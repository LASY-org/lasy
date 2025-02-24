# -*- coding: utf-8 -*-
"""Test the polynomial spectral phase implementation.

Test checks the implementation of the polynomial spectral phase
by initializing a Gaussian pulse (with flat spectral phase),
adding spectral phase to it, and checking the corresponding
temporal shape of the laser pulse against analytical formulas.
"""

import numpy as np
from scipy.constants import c

from lasy.laser import Laser
from lasy.optical_elements import PolynomialSpectralPhase
from lasy.profiles.gaussian_profile import GaussianProfile

# Laser parameters
wavelength = 0.8e-6
omega0 = 2 * np.pi * c / wavelength
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


def test_delay():
    """Add and compare group delay for laser.

    Add group delay to the laser and compare the on-axis field with the
    analytical formula for a Gaussian pulse with group delay.
    """
    delay = 50e-15
    dazzler = PolynomialSpectralPhase(omega0, delay=delay)

    # Initialize the laser
    dim = "xyt"
    lo = (-12e-6, -12e-6, -100e-15)
    hi = (+12e-6, +12e-6, +100e-15)
    npoints = (100, 100, 400)
    laser = Laser(dim, lo, hi, npoints, gaussian_profile)

    # Get field before and after dazzler
    E_before = laser.grid.get_temporal_field()
    laser.apply_optics(dazzler)
    E_after = laser.grid.get_temporal_field()

    # Extract peak field before dazzler
    E0 = abs(E_before[50, 50]).max()

    # Compute the analtical expression in real space for a Gaussian
    t = np.linspace(laser.grid.lo[-1], laser.grid.hi[-1], laser.grid.npoints[-1])

    E_analytical = E0 * np.exp(-1.0 * ((t - delay) / tau) ** 2)

    # Compare the on-axis field with the analytical formula
    tol = 1.2e-3
    assert np.all(
        abs(E_after[50, 50, :] - E_analytical) / abs(E_analytical).max() < tol
    )


def test_gdd():
    """Add and compare GDD laser.

    Add GDD to the laser and compare the on-axis field with the
    analytical formula for a Gaussian pulse with GDD.
    """
    gdd = 200e-30
    dazzler = PolynomialSpectralPhase(omega0, gdd=gdd)

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
    # Extract peak field before dazzler
    E0 = abs(E_before[50, 50]).max()

    # Compute the analtical expression in real space for a Gaussian
    t = np.linspace(laser.grid.lo[-1], laser.grid.hi[-1], laser.grid.npoints[-1])
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
    """Add and compare TOD laser.

    Add TOD to the laser and compare the on-axis field with the
    analytical formula from the stationary phase approximation.
    """
    tod = 5e-42
    dazzler = PolynomialSpectralPhase(omega0, tod=tod)

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
    # Extract peak field before dazzler
    E0 = abs(E_before[50, 50]).max()

    # Compare data in the post-pulse region to the stationary phase approximation.
    # The stationary phase approximation result was obtained under the additional
    # assumption t >> tau^4/tod, so we only compare the data in this region
    E_compare = abs(E_after[50, 50, t > t_peak + 2 * tau**4 / tod])  # On-axis field
    t = t[t > t_peak + 2 * tau**4 / tod]
    prediction = abs(
        2
        * E0
        * tau
        / (8 * tod * t) ** 0.25
        * np.exp(-(tau**2) * t / (2 * tod))
        * np.cos(
            2 * t / 3 * (2 * t / tod) ** 0.5
            - tau**4 / (8 * tod) * (2 * t / tod) ** 0.5
            - np.pi / 4
        )
    )

    # Compare the on-axis field with the analytical formula
    tol = 2.4e-2
    assert np.all(abs(E_compare - prediction) / abs(E0) < tol)
