# -*- coding: utf-8 -*-

import pytest
import numpy as np

from lasy.laser import Laser
from lasy.profiles.profile import Profile, SummedProfile, ScaledProfile
from lasy.profiles import CombinedLongitudinalTransverseProfile, GaussianProfile
from lasy.profiles.longitudinal import GaussianLongitudinalProfile
from lasy.profiles.transverse import (
    LaguerreGaussianTransverseProfile,
    SuperGaussianTransverseProfile,
)


class MockProfile(Profile):
    """
    A mock Profile class that always returns a constant value.
    """

    def __init__(self, wavelength, pol, value):
        super().__init__(wavelength, pol)
        self.value = value

    def evaluate(self, x, y, t):
        return np.ones_like(x, dtype="complex128") * self.value


@pytest.fixture(scope="function")
def gaussian():
    # Cases with Gaussian laser
    wavelength = 0.8e-6
    pol = (1, 0)
    laser_energy = 1.0  # J
    t_peak = 0.0e-15  # s
    tau = 30.0e-15  # s
    w0 = 5.0e-6  # m
    profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)

    return profile


def test_profile_gaussian_3d_cartesian(gaussian):
    # - 3D Cartesian case
    dim = "xyt"
    lo = (-10e-6, -10e-6, -60e-15)
    hi = (+10e-6, +10e-6, +60e-15)
    npoints = (100, 100, 100)

    laser = Laser(dim, lo, hi, npoints, gaussian)
    laser.write_to_file("gaussianlaser3d")
    laser.propagate(1e-6)
    laser.write_to_file("gaussianlaser3d")


def test_profile_gaussian_cylindrical(gaussian):
    # - Cylindrical case
    dim = "rt"
    lo = (0e-6, -60e-15)
    hi = (10e-6, +60e-15)
    npoints = (50, 100)

    laser = Laser(dim, lo, hi, npoints, gaussian)
    laser.write_to_file("gaussianlaserRZ")
    laser.propagate(1e-6)
    laser.write_to_file("gaussianlaserRZ")


def test_profile_laguerre_gauss():
    # Case with Laguerre-Gauss laser
    wavelength = 0.8e-6
    pol = (1, 0)
    laser_energy = 1.0  # J
    t_peak = 0.0e-15  # s
    tau = 30.0e-15  # s
    w0 = 5.0e-6  # m
    profile = CombinedLongitudinalTransverseProfile(
        wavelength,
        pol,
        laser_energy,
        GaussianLongitudinalProfile(wavelength, tau, t_peak),
        LaguerreGaussianTransverseProfile(w0, p=0, m=1),
    )

    # - Cylindrical case
    dim = "rt"
    lo = (0e-6, -60e-15)
    hi = (10e-6, +60e-15)
    npoints = (50, 100)

    laser = Laser(dim, lo, hi, npoints, profile, n_azimuthal_modes=2)
    laser.write_to_file("laguerrelaserRZ")
    laser.propagate(1e-6)
    laser.write_to_file("laguerrelaserRZ")


def test_profile_super_gauss():
    # Case with super-Gaussian laser
    wavelength = 0.8e-6
    pol = (1, 0)
    laser_energy = 1.0  # J
    t_peak = 0.0e-15  # s
    tau = 30.0e-15  # s
    w0 = 5.0e-6  # m
    profile = CombinedLongitudinalTransverseProfile(
        wavelength,
        pol,
        laser_energy,
        GaussianLongitudinalProfile(wavelength, tau, t_peak),
        SuperGaussianTransverseProfile(w0, n_order=10),
    )

    # - Cylindrical case
    dim = "rt"
    lo = (0e-6, -60e-15)
    hi = (10e-6, +60e-15)
    npoints = (50, 100)

    laser = Laser(dim, lo, hi, npoints, profile, n_azimuthal_modes=2)
    laser.write_to_file("superGaussianlaserRZ")
    laser.propagate(1)
    laser.write_to_file("superGaussianlaserRZ")

    return profile


def test_add_profiles():
    # Add the two profiles together
    profile_1 = MockProfile(0.8e-6, (1, 0), 1.0)
    profile_2 = MockProfile(0.8e-6, (1, 0), 2.0)
    summed_profile = profile_1 + profile_2
    # Check that the result is a SummedProfile object
    assert isinstance(summed_profile, SummedProfile)
    # Check that the profiles are stored correctly
    assert summed_profile.profiles[0] == profile_1
    assert summed_profile.profiles[1] == profile_2
    # Check that the evaluate method works
    assert np.allclose(summed_profile.evaluate(0, 0, 0), 3.0)


def test_add_error_if_not_all_profiles():
    profile_1 = MockProfile(0.8e-6, (1, 0), 1.0)
    with pytest.raises(AssertionError):
        profile_1 + 1.0


def test_add_error_if_different_wavelength():
    profile_1 = MockProfile(0.8e-6, (1, 0), 1.0)
    profile_2 = MockProfile(0.8e-6, (1, 0), 2.0)
    profile_3 = MockProfile(0.9e-6, (1, 0), 2.0)
    summed_profile = profile_1 + profile_2
    with pytest.raises(AssertionError):
        summed_profile + profile_3


def test_add_error_if_different_polarisation():
    profile_1 = MockProfile(0.8e-6, (1, 0), 1.0)
    profile_2 = MockProfile(0.8e-6, (1, 0), 2.0)
    profile_3 = MockProfile(0.8e-6, (0, 1), 2.0)
    summed_profile = profile_1 + profile_2
    with pytest.raises(AssertionError):
        summed_profile + profile_3


def test_scale_profiles():
    # Add the two profiles together
    profile_1 = MockProfile(0.8e-6, (1, 0), 1.0)
    scaled_profile = 2.0 * profile_1
    scaled_profile_right = profile_1 * 2.0
    # Check that the result is a ScaledProfile object
    assert isinstance(scaled_profile, ScaledProfile)
    # Check that the profiles are stored correctly
    assert scaled_profile.profile == profile_1
    # Check that the evaluate method works
    assert np.allclose(scaled_profile.evaluate(0, 0, 0), 2.0)
    assert np.allclose(scaled_profile_right.evaluate(0, 0, 0), 2.0)


def test_scale_error_if_not_scalar():
    profile_1 = MockProfile(0.8e-6, (1, 0), 1.0)
    with pytest.raises(AssertionError):
        profile_1 * profile_1
    with pytest.raises(AssertionError):
        profile_1 * [1.0, 2.0]
