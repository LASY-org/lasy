"""Test the implementation of the HG recostruction.

Test checks the implementation of the HG reconstruction. It does so
by initializing a super-Gaussian pulse and denoising it. It then
checks that the error remains positive and less than the predefined
value.
"""

import numpy as np

from lasy.profiles.transverse.super_gaussian_profile import (
    GaussianTransverseProfile,
)
from lasy.utils.denoise import hg_reconstruction


def test_denoise_hg_reconstruction():
    # Parameters
    waist = 20e-6
    shape_parameter = 3
    wavelength = 8e-7    
    resolution = 0.2e-6
    lo = [-2e-4, -2e-4]
    hi = [2e-4, 2e-4]
    
    # Define the transverse profile
    transverse_profile = SuperGaussianTransverseProfile(
        waist, shape_parameter
    )  # Super-Gaussian profile
    transverse_profile_cleaned, waist = hg_reconstruction(
        transverse_profile, wavelength, resolution, lo, hi
    )  # Denoised profile

    # Calculate the error
    x = np.linspace(-5 * waist[0], 5 * waist[0], 500)
    X, Y = np.meshgrid(x, x)
    prof1 = np.abs(transverse_profile.evaluate(X, Y)) ** 2
    prof2 = np.abs(transverse_profile_cleaned.evaluate(X, Y)) ** 2
    error = (prof1 - prof2) / np.max(prof1)
    assert 0 < np.max(error) < 0.2
