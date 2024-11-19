"""Test the implementation of the denoise.

Test checks the implementation of the denoise
by initializing a super-Gaussian pulse and denoise it. It then
checks that the error remains positive and less than the predefined
value.
"""

import numpy as np

from lasy.profiles.transverse.super_gaussian_profile import (
    SuperGaussianTransverseProfile,
)
from lasy.utils.denoise import denoise_transverse_hg
from lasy.profiles.transverse.super_gaussian_profile import SuperGaussianTransverseProfile

def test_denoise_transverse_hg():

    # Parameters
    waist = 20e-6
    shape_parameter = 3

    # Define the transverse profile
    transverse_profile = SuperGaussianTransverseProfile(
        waist, shape_parameter
    )  # Super-Gaussian profile
    transverse_profile_cleaned, waist, l = denoise_transverse_hg(
        transverse_profile
    )  # Denoised profile

    # Calculate the error
    x = np.linspace(-5 * waist, 5 * waist, 500)
    X, Y = np.meshgrid(x, x)
    prof1 = np.abs(transverse_profile.evaluate(X, Y)) ** 2
    prof2 = np.abs(transverse_profile_cleaned.evaluate(X, Y)) ** 2
    error = (prof1 - prof2) / np.max(prof1)
    assert 0 < np.max(error) < 0.2