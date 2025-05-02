# -*- coding: utf-8 -*-
"""Checking the implementation of the resampling propagator.

Initializing a Gaussian pulse in the near field, and
propagating it through a parabolic mirror, and then to
the focal position (radial axis is resampled to accomodate the new size of the beam);
we then check that the waist has the expected value in the far field (i.e. in the focal plane)
"""

import numpy as np

from lasy.laser import Grid, Laser
from lasy.optical_elements import ParabolicMirror
from lasy.profiles.combined_profile import CombinedLongitudinalTransverseProfile
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.profiles.longitudinal.gaussian_profile import GaussianLongitudinalProfile
from lasy.profiles.transverse.laguerre_gaussian_profile import (
    LaguerreGaussianTransverseProfile,
)

# Laser parameters
wavelength = 0.8e-6
w0 = 5.0e-3  # m, initialized in near field
pol = (1, 0)
laser_energy = 1.0  # J
t_peak = 0.0e-15  # s
tau = 30.0e-15  # s

# Define the initial grid for the laser
dim = "rt"
lo = (0e-3, -90e-15)
hi = (15e-3, +90e-15)
npoints = (500, 100)


def get_w0(laser, m):
    # Calculate the laser waist
    field = laser.grid.get_temporal_field()
    A2 = (np.abs(field[m, :, :]) ** 2).sum(-1)
    ax = laser.grid.axes[0]
    if ax[0] > 0:
        A2 = np.r_[A2[::-1], A2]
        ax = np.r_[-ax[::-1], ax]
    else:
        A2 = np.r_[A2[::-1][:-1], A2]
        ax = np.r_[-ax[::-1][:-1], ax]

    sigma = 2 * np.sqrt(np.average(ax**2, weights=A2))

    return sigma


def check_resampling(laser, new_grid, m=0):
    # Focus down the laser and propagate
    f0 = 2.0  # focal distance in m
    laser.apply_optics(ParabolicMirror(f=f0))
    laser.propagate(f0, nr_boundary=128, grid_out=new_grid)  # resample the radial grid

    # Check that the value is the expected one in the near field
    w0_num = get_w0(laser, m)
    assert m in [0, 1]
    w0_theor = wavelength * f0 / (np.pi * w0)
    if m == 1:
        w0_theor *= np.sqrt(3)
    err = 2 * np.abs(w0_theor - w0_num) / (w0_theor + w0_num)
    assert err < 1e-3


def test_resampling_gaussian():
    # Initialize the laser (gaussian profile)
    gaussian_profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)
    laser = Laser(dim, lo, hi, npoints, gaussian_profile)

    # Define the new grid for the laser
    new_r_max = 300e-6
    npoints_new = (50, 100)
    new_grid = Grid(
        dim,
        lo,
        (new_r_max, hi[1]),
        npoints_new,
        n_azimuthal_modes=laser.grid.n_azimuthal_modes,
    )
    # Check resampling propagator
    check_resampling(laser, new_grid)


def test_resampling_multimode_gaussian():
    # Initialize the laser (gaussian profile)
    gaussian_profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)
    laser = Laser(
        dim, lo, hi, npoints, gaussian_profile, n_azimuthal_modes=3, n_theta_evals=20
    )

    # Define the new grid for the laser
    new_r_max = 300e-6
    npoints_new = (50, 100)
    new_grid = Grid(
        dim,
        lo,
        (new_r_max, hi[1]),
        npoints_new,
        n_azimuthal_modes=laser.grid.n_azimuthal_modes,
    )
    # Check resampling propagator
    check_resampling(laser, new_grid)


def test_resampling_laguerre():
    # Initialize the laser (LaguerreGaussian)
    p = 0  # Radial order of Generalized Laguerre polynomial
    m = 1  # Phase Rotation

    LongitProfile = GaussianLongitudinalProfile(wavelength, tau, t_peak)
    TransvProfile = LaguerreGaussianTransverseProfile(w0, p, m, wavelength=800e-9)
    pulseProfile = CombinedLongitudinalTransverseProfile(
        wavelength, pol, LongitProfile, TransvProfile, laser_energy=laser_energy
    )

    laser = Laser(dim, lo, hi, npoints, pulseProfile, n_azimuthal_modes=2)

    # Define the new grid for the laser
    new_r_max = 300e-6
    npoints_new = (50, 100)
    new_grid = Grid(
        dim,
        lo,
        (new_r_max, hi[1]),
        npoints_new,
        n_azimuthal_modes=laser.grid.n_azimuthal_modes,
    )
    # Check resampling propagator
    check_resampling(laser, new_grid, m)
