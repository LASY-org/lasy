"""Checking the implementation of the Fresnel Chrip-Z propagator.

Initializing a Gaussian pulse in the near field, and
propagating it through a parabolic mirror, and then to
the focal position (transverse plane is resampled to accomodate the new size of the beam);
we then check that the waist has the expected value in the far field (i.e. in the focal plane)
"""

import numpy as np

from lasy.laser import Grid, Laser
from lasy.optical_elements import ParabolicMirror
from lasy.profiles.combined_profile import CombinedLongitudinalTransverseProfile
from lasy.profiles.longitudinal.continuous_wave_profile import ContinuousWaveProfile
from lasy.profiles.transverse.laguerre_gaussian_profile import (
    LaguerreGaussianTransverseProfile,
)
from lasy.propagators.fresnel_chirpztransform_propagator import FresnelChirpZPropagator
from lasy.utils.laser_utils import get_w0

# Laser parameters
w0 = 5.0e-3  # m, initialized in near field
pol = (1, 0)
peak_fluence = 1e4  # W/m^2
dim = "xyt"


def check_resampling(laser, new_grid, m=0, wavelength=800e-9):
    # Focus down the laser and propagate
    f0 = 2.0  # focal distance in m
    laser.apply_optics(ParabolicMirror(f=f0))
    laser.add_propagator(FresnelChirpZPropagator())
    laser.propagate(f0, grid_out=new_grid)  # resample the radial grid

    # Check that the value is the expected one in the near field
    w0_num = get_w0(laser.grid, laser.dim)
    assert m in [0, 1]
    w0_theor = wavelength * f0 / (np.pi * w0)
    if m == 1:
        w0_theor *= np.sqrt(3)
    err = 2 * np.abs(w0_theor - w0_num) / (w0_theor + w0_num)

    assert err < 1e-3


def resampling_laguerre_CW(m=0, wavelength=800e-9):
    # Initialize the laser (LaguerreGaussian)
    p = 0  # Radial order of Generalized Laguerre polynomial

    LongitProfile = ContinuousWaveProfile(wavelength)
    TransvProfile = LaguerreGaussianTransverseProfile(w0, p, m, wavelength=wavelength)
    pulseProfile = CombinedLongitudinalTransverseProfile(
        wavelength, pol, LongitProfile, TransvProfile, peak_fluence=1
    )

    lo = (-15e-3, -15e-3, -1)
    hi = (15e-3, 15e-3, 1)
    npoints = (1024, 1024, 1)

    laser = Laser(dim, lo, hi, npoints, pulseProfile)

    # Define the new grid for the laser
    new_r_max = 250e-6
    npoints_new = (1024, 1024, npoints[2])
    new_grid = Grid(
        dim, (-new_r_max, -new_r_max, lo[2]), (new_r_max, new_r_max, hi[2]), npoints_new
    )
    # Check resampling propagator
    check_resampling(laser, new_grid, m, wavelength)


def test_gaussian_633nm():
    # Test a gaussian beam at a wavelength of 633nm
    resampling_laguerre_CW(m=0, wavelength=633e-9)


def test_gaussian_800nm():
    # Test a gaussian beam at a wavelength of 800 nm
    resampling_laguerre_CW(m=0, wavelength=800e-9)


def test_gaussian_m1_633nm():
    # Test a gaussian beam at a wavelength of 633nm
    resampling_laguerre_CW(m=1, wavelength=633e-9)


def test_gaussian_m1_800nm():
    # Test a gaussian beam at a wavelength of 800 nm
    resampling_laguerre_CW(m=1, wavelength=800e-9)
