"""Test the implementation of the HG recostruction.

Test checks the implementation of the HG reconstruction. It does so
by initializing a super-Gaussian pulse and denoising it. It then
checks that the error remains positive and less than the predefined
value.
"""
from copy import deepcopy

import numpy as np

from lasy.laser import Laser
from lasy.profiles.combined_profile import CombinedLongitudinalTransverseProfile
from lasy.profiles.transverse.super_gaussian_profile import (
    SuperGaussianTransverseProfile,
)
from lasy.profiles.longitudinal import ContinuousWaveProfile
from lasy.utils.mode_decomposition import *


def test_denoise_hg_reconstruction():
    # Parameters
    dimensions = 'xyt'
    pol = (0,1)
    peak_fluence = 1.0
    lambda0 = 800e-9
    w0 = 20e-6
    shape_parameter = 3
    lo = [-2e-4, -2e-4, None]
    hi = [2e-4, 2e-4, None]
    num_points = [1000,1000,1]

    
    # Define the transverse profile
    longitudinal_profile = ContinuousWaveProfile(lambda0)
    transverse_profile = SuperGaussianTransverseProfile(
        w0, shape_parameter
    )  # Super-Gaussian profile
    laser_profile = CombinedLongitudinalTransverseProfile(
    lambda0, pol, longitudinal_profile, transverse_profile, peak_fluence=peak_fluence,
)

    laser_raw = Laser(dimensions, lo, hi, num_points, laser_profile)

    # Calculate the decomposition and waist of the laser pulse
    modes = hermite_gauss_decomposition(laser_raw, w0, w0, 10, 10)
    
    laser_cleaned = deepcopy(laser_raw)  # Make a copy of the input grid
    hermite_gauss_composition(laser_cleaned, w0, w0, modes)

    # Calculate the error
    x = np.linspace(-5 * w0, 5 * w0, 500)
    X, Y = np.meshgrid(x, x)
    
    # Original profile
    prof1 = laser_raw.grid.get_temporal_field()
    
    # Reconstructed profile
    prof2 = laser_cleaned.grid.get_temporal_field()

    error = np.sum(np.abs(prof2 - prof1) ** 2) / np.sum(np.abs(prof1) ** 2)
    print(error)
    assert error < 0.02