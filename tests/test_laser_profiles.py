# -*- coding: utf-8 -*-

import numpy as np
import pytest
from scipy.constants import c

from lasy.laser import Laser
from lasy.profiles import FromArrayProfile, GaussianProfile, SpeckleProfile
from lasy.profiles.longitudinal import (
    CosineLongitudinalProfile,
    GaussianLongitudinalProfile,
    SuperGaussianLongitudinalProfile,
)
from lasy.profiles.profile import Profile, ScaledProfile, SummedProfile
from lasy.profiles.transverse import (
    GaussianTransverseProfile,
    HermiteGaussianTransverseProfile,
    JincTransverseProfile,
    LaguerreGaussianTransverseProfile,
    ScaledTransverseProfile,
    SummedTransverseProfile,
    SuperGaussianTransverseProfile,
    TransverseProfile,
    TransverseProfileFromData,
)
from lasy.utils.exp_data_utils import find_center_of_mass


class MockProfile(Profile):
    """
    A mock Profile class that always returns a constant value.
    """

    def __init__(self, wavelength, pol, value):
        super().__init__(wavelength, pol)
        self.value = value

    def evaluate(self, x, y, t):
        return np.ones_like(x, dtype="complex128") * self.value


class MockTransverseProfile(TransverseProfile):
    """
    A mock TransverseProfile class that always returns a constant value.
    """

    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self, x, y):
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


def test_transverse_profiles_rt():
    npoints = 4000
    w0 = 10.0e-6
    r = np.linspace(0, 12 * w0, npoints)

    # GaussianTransverseProfile
    print("GaussianTransverseProfile")
    std_th = w0 / np.sqrt(2)
    profile = GaussianTransverseProfile(w0)
    field = profile.evaluate(r, np.zeros_like(r))
    std = np.sqrt(np.average(r**2, weights=np.abs(field)))
    print("std_th = ", std_th)
    print("std = ", std)
    assert np.abs(std - std_th) / std_th < 0.01

    # LaguerreGaussianTransverseProfile
    print("LaguerreGaussianTransverseProfile")
    p = 2
    m = 0
    std_th = np.sqrt(5 / 2) * w0
    profile = LaguerreGaussianTransverseProfile(w0, p, m)
    field = profile.evaluate(r, np.zeros_like(r))
    std = np.sqrt(np.average(r**2, weights=r * np.abs(field) ** 2))
    print("std_th = ", std_th)
    print("std = ", std)
    assert np.abs(std - std_th) / std_th < 0.01

    # SuperGaussianTransverseProfile
    print("SuperGaussianTransverseProfile")
    n_order = 100  # close to flat-top, compared with flat-top theory
    std_th = w0 / np.sqrt(3)
    profile = SuperGaussianTransverseProfile(w0, n_order)
    field = profile.evaluate(r, np.zeros_like(r))
    std = np.sqrt(np.average(r**2, weights=np.abs(field)))
    print("std_th = ", std_th)
    print("std = ", std)
    assert np.abs(std - std_th) / std_th < 0.01

    # JincTransverseProfile
    print("JincTransverseProfile")
    profile = JincTransverseProfile(w0)
    std_th = 1.5 * w0  # Just measured from this test
    field = profile.evaluate(r, np.zeros_like(r))
    std = np.sqrt(np.average(r**2, weights=abs(field) ** 2))
    print("std_th = ", std_th)
    print("std = ", std)
    assert np.abs(std - std_th) / std_th < 0.1


def test_transverse_profiles_3d():
    npoints = 200
    w0 = 10.0e-6

    # HermiteGaussianTransverseProfile
    print("HermiteGaussianTransverseProfile")
    n_x = 2
    n_y = 2
    std_th = np.sqrt(5.0 / 4) * w0
    profile = HermiteGaussianTransverseProfile(w0, n_x, n_y)
    x = np.linspace(-4 * w0, 4 * w0, npoints)
    y = np.zeros_like(x)
    field = profile.evaluate(x, y)
    std = np.sqrt(np.average(x**2, weights=np.abs(field) ** 2))
    print("std_th = ", std_th)
    print("std = ", std)
    assert np.abs(std - std_th) / std_th < 0.01

    # TransverseProfileFromData
    print("TransverseProfileFromData")
    lo = (-40.0e-6, -40.0e-6)
    hi = (40.0e-6, 40.0e-6)
    x = np.linspace(lo[0], hi[0], 200)
    y = np.linspace(lo[1], hi[1], 100)
    dx = x[1] - x[0]
    w0 = 10.0e-6
    x0 = 10.0e-6
    X, Y = np.meshgrid(x, y, indexing="ij")
    intensity_data = np.exp(-((X - x0) ** 2 + Y**2) / w0**2)
    profile = TransverseProfileFromData(intensity_data, lo, hi)
    field = profile.evaluate(X, Y)
    x0_test = find_center_of_mass(field**2)[0] * dx + lo[0]
    print("beam center, theory: ", x0)
    print("beam center from profile ", x0_test)
    assert (x0_test - x0) / x0 < 0.1


def test_longitudinal_profiles():
    npoints = 10000

    wavelength = 800e-9
    tau_fwhm = 30.0e-15
    t_peak = 1.0 * tau_fwhm
    cep_phase = 0.5 * np.pi
    omega_0 = 2.0 * np.pi * c / wavelength

    t = np.linspace(t_peak - 4 * tau_fwhm, t_peak + 4 * tau_fwhm, npoints)

    # GaussianLongitudinalProfile
    print("GaussianLongitudinalProfile")
    tau = tau_fwhm / np.sqrt(2 * np.log(2))
    profile_gaussian = GaussianLongitudinalProfile(wavelength, tau, t_peak, cep_phase)
    field_gaussian = profile_gaussian.evaluate(t)

    std_gauss = np.sqrt(np.average((t - t_peak) ** 2, weights=np.abs(field_gaussian)))
    std_gauss_th = tau / np.sqrt(2.0)
    print("std_th = ", std_gauss_th)
    print("std = ", std_gauss)
    assert np.abs(std_gauss - std_gauss_th) / std_gauss_th < 0.01

    t_peak_gaussian = t[np.argmax(np.abs(field_gaussian))]
    print("t_peak_th = ", t_peak)
    print("t_peak = ", t_peak_gaussian)
    assert np.abs(t_peak_gaussian - t_peak) / t_peak < 0.01

    ff_gaussian = field_gaussian * np.exp(-1.0j * omega_0 * t)
    cep_phase_gaussian = np.angle(ff_gaussian[np.argmax(np.abs(field_gaussian))])
    print("cep_phase_th = ", cep_phase)
    print("cep_phase = ", cep_phase_gaussian)
    assert np.abs(cep_phase_gaussian - cep_phase) / cep_phase < 0.02

    # SuperGaussianLongitudinalProfile
    print("SuperGaussianLongitudinalProfile")
    n_order = 2  # ordinary gaussian
    tau = tau_fwhm / np.sqrt(2 * np.power(np.log(2), n_order / 2))
    profile_super_gaussian = SuperGaussianLongitudinalProfile(
        wavelength, tau, t_peak, n_order, cep_phase
    )
    field_super_gaussian = profile_super_gaussian.evaluate(t)

    std_super_gauss = np.sqrt(
        np.average((t - t_peak) ** 2, weights=np.abs(field_super_gaussian))
    )
    std_super_gauss_th = tau / np.sqrt(2.0)
    print("std_th = ", std_super_gauss_th)
    print("std = ", std_super_gauss)
    assert np.abs(std_super_gauss - std_super_gauss_th) / std_super_gauss_th < 0.01

    t_peak_super_gaussian = t[np.argmax(np.abs(field_super_gaussian))]
    print("t_peak_th = ", t_peak)
    print("t_peak = ", t_peak_super_gaussian)
    assert np.abs(t_peak_super_gaussian - t_peak) / t_peak < 0.01

    ff_super_gaussian = field_super_gaussian * np.exp(-1.0j * omega_0 * t)
    cep_phase_super_gaussian = np.angle(
        ff_super_gaussian[np.argmax(np.abs(field_super_gaussian))]
    )
    print("cep_phase_th = ", cep_phase)
    print("cep_phase = ", cep_phase_super_gaussian)
    assert np.abs(cep_phase_super_gaussian - cep_phase) / cep_phase < 0.02

    # CosineLongitudinalProfile
    print("CosineLongitudinalProfile")
    profile_cos = CosineLongitudinalProfile(wavelength, tau_fwhm, t_peak, cep_phase)
    field_cos = profile_cos.evaluate(t)

    std_cos = np.sqrt(np.average((t - t_peak) ** 2, weights=np.abs(field_cos)))
    std_cos_th = tau_fwhm * np.sqrt(1 - 8 / np.pi**2)
    print("std_th = ", std_cos_th)
    print("std = ", std_cos)
    assert np.abs(std_cos - std_cos_th) / std_cos_th < 0.01

    t_peak_cos = t[np.argmax(np.abs(field_cos))]
    print("t_peak_th = ", t_peak)
    print("t_peak = ", t_peak_cos)
    assert np.abs(t_peak_cos - t_peak) / t_peak < 0.01

    ff_cos = field_cos * np.exp(-1.0j * omega_0 * t)
    cep_phase_cos = np.angle(ff_cos[np.argmax(np.abs(field_cos))])
    print("cep_phase_th = ", cep_phase)
    print("cep_phase = ", cep_phase_cos)
    assert np.abs(cep_phase_cos - cep_phase) / cep_phase < 0.02


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


def test_from_array_profile():
    # Create a 3D numpy array, use it to create a LASY profile,
    # and check that the resulting profile has the correct width
    lo = (-10e-6, -20e-6, -30e-15)
    hi = (+10e-6, +20e-6, +30e-15)
    npoints = (16, 32, 64)
    x = np.linspace(lo[0], hi[0], npoints[0])
    y = np.linspace(lo[1], hi[1], npoints[1])
    t = np.linspace(lo[2], hi[2], npoints[2])
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    e0 = 4.0e12  # Approx a0 = 1 for wavelength = 800 nm
    wx = 3.0e-6
    wy = 5.0e-6
    tau = 5.0e-15
    E = 1j * e0 * np.exp(-(X**2) / wx**2 - Y**2 / wy**2 - t**2 / tau**2)
    wavelength = 0.8e-6
    pol = (1, 0)
    axes = {"x": x, "y": y, "t": t}
    dim = "xyt"

    profile = FromArrayProfile(
        wavelength=wavelength, pol=pol, array=E, dim=dim, axes=axes
    )
    laser = Laser(dim, lo, hi, npoints, profile)
    laser.write_to_file("fromArray")

    F = profile.evaluate(X, Y, T)
    width = (
        np.sqrt(
            np.sum(np.abs(F) ** 2 * x.reshape((x.size, 1, 1)) ** 2)
            / np.sum(np.abs(F) ** 2)
        )
        * 2
    )
    print("theory width  : ", wx)
    print("Measured width: ", width)
    assert np.abs((width - wx) / wx) < 1.0e-5


def test_speckle_profile():
    # - speckled laser case
    print("SpeckledProfile")
    wavelength = 0.351e-6  # Laser wavelength in meters
    polarization = (1, 0)  # Linearly polarized in the x direction
    laser_energy = 1.0  # J (this is the laser energy stored in the box defined by `lo` and `hi` below)
    focal_length = 3.5  # m
    beam_aperture = [0.35, 0.5]  # m
    n_beamlets = [24, 32]
    temporal_smoothing_type = "GP ISI"
    relative_laser_bandwidth = 0.005

    profile = SpeckleProfile(
        wavelength,
        polarization,
        laser_energy,
        focal_length,
        beam_aperture,
        n_beamlets,
        temporal_smoothing_type=temporal_smoothing_type,
        relative_laser_bandwidth=relative_laser_bandwidth,
    )
    dimensions = "xyt"
    dx = wavelength * focal_length / beam_aperture[0]
    dy = wavelength * focal_length / beam_aperture[1]
    Lx = 1.8 * dx * n_beamlets[0]
    Ly = 3.1 * dy * n_beamlets[1]
    nu_laser = c / wavelength
    t_max = 50 / nu_laser
    lo = (0, 0, 0)
    hi = (Lx, Ly, t_max)
    npoints = (200, 250, 2)

    laser = Laser(dimensions, lo, hi, npoints, profile)
    laser.write_to_file("speckledProfile")


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


def test_add_transverse_profiles():
    # Add the two profiles together
    trans_profile_1 = MockTransverseProfile(1.0)
    trans_profile_2 = MockTransverseProfile(2.0)
    summed_trans_profile = trans_profile_1 + trans_profile_2
    # Check that the result is a SummedTransverseProfile object
    assert isinstance(summed_trans_profile, SummedTransverseProfile)
    # Check that the profiles are stored correctly
    assert summed_trans_profile.transverse_profiles[0] == trans_profile_1
    assert summed_trans_profile.transverse_profiles[1] == trans_profile_2
    # Check that the evaluate method works
    assert np.allclose(summed_trans_profile.evaluate(0, 0), 3.0)


def test_add_transverse_error_if_not_all_transverse_profiles():
    trans_profile_1 = MockTransverseProfile(1.0)
    with pytest.raises(AssertionError):
        trans_profile_1 + 1.0


def test_scale_transverse_profiles():
    # Add the two profiles together
    trans_profile_1 = MockTransverseProfile(1.0)
    scaled_trans_profile = 2.0 * trans_profile_1
    scaled_trans_profile_right = trans_profile_1 * 2.0
    # Check that the result is a ScaledProfile object
    assert isinstance(scaled_trans_profile, ScaledTransverseProfile)
    assert isinstance(scaled_trans_profile_right, ScaledTransverseProfile)
    # Check that the profiles are stored correctly
    assert scaled_trans_profile.transverse_profile == trans_profile_1
    # Check that the evaluate method works
    assert np.allclose(scaled_trans_profile.evaluate(0, 0), 2.0)
    assert np.allclose(scaled_trans_profile.evaluate(0, 0), 2.0)


def test_scale_trans_error_if_not_scalar():
    trans_profile_1 = MockTransverseProfile(1.0)
    with pytest.raises(AssertionError):
        trans_profile_1 * trans_profile_1
    with pytest.raises(AssertionError):
        trans_profile_1 * [1.0, 2.0]
