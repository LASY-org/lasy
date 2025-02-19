"""Test the implementation of the HG recostruction.

Test checks the implementation of the HG reconstruction. It does so
by initializing a super-Gaussian pulse and denoising it. It then
checks that the error remains positive and less than the predefined
value.
"""

import numpy as np

from lasy.profiles.transverse.super_gaussian_profile import (
    SuperGaussianTransverseProfile,
)
from lasy.utils.denoise import hg_reconstruction


def test_denoise_hg_reconstruction():
    # Parameters
    w0 = 20e-6
    shape_parameter = 3
    wavelength = 8e-7
    resolution = 0.2e-6
    lo = [-2e-4, -2e-4]
    hi = [2e-4, 2e-4]

    # Define the transverse profile
    transverse_profile = SuperGaussianTransverseProfile(
        w0, shape_parameter
    )  # Super-Gaussian profile
    transverse_profile_cleaned, w0x, w0y = hg_reconstruction(
        transverse_profile, wavelength, resolution, lo, hi
    )  # Denoised profile

    # Calculate the error
    x = np.linspace(-5 * w0x, 5 * w0x, 500)
    X, Y = np.meshgrid(x, x)
    prof1 = transverse_profile.evaluate(X, Y)
    prof2 = transverse_profile_cleaned.evaluate(X, Y)
    error = np.sum(np.abs(prof2 - prof1) ** 2) / np.sum(np.abs(prof1) ** 2)
    assert error < 0.02
