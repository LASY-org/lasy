import numpy as np

from scipy.constants import c
from lasy.laser import Laser
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.utils.laser_utils import (
    get_spectrum,
    compute_laser_energy,
    get_t_peak,
    get_duration,
)


def get_gaussian_profile():
    # Cases with Gaussian laser
    wavelength = 0.8e-6
    pol = (1, 0)
    laser_energy = 1.0  # J
    t_peak = 0.0e-15  # s
    tau = 30.0e-15  # s
    w0 = 5.0e-6  # m
    profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)

    return profile


def get_spatial_chirp_profile():
    # Cases with Gaussian laser
    wavelength = 0.8e-6
    pol = (1, 0)
    laser_energy = 1.0  # J
    t_peak = 0.0e-15  # s
    tau = 30.0e-15  # s
    w0 = 5.0e-6  # m
    a = 0.0
    b = tau * w0 / 2  # m.s
    profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak, a, b)

    return profile


def get_angular_dispersion_profile():
    # Cases with Gaussian laser
    wavelength = 0.8e-6
    pol = (1, 0)
    laser_energy = 1.0  # J
    t_peak = 0.0e-15  # s
    tau = 30.0e-15  # s
    w0 = 5.0e-6  # m
    a = tau / w0  # s/m
    profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak, a)

    return profile


def get_gaussian_laser(dim):
    # - Cylindrical case
    if dim == "rt":
        lo = (0e-6, -100e-15)
        hi = (25e-6, +100e-15)
        npoints = (100, 200)
    else:  # dim == "xyt":
        lo = (-25e-6, -25e-6, -100e-15)
        hi = (+25e-6, +25e-6, +100e-15)
        npoints = (100, 100, 200)
    return Laser(dim, lo, hi, npoints, get_gaussian_profile())


def get_spatial_chirp_laser(dim):
    # - Cylindrical case
    if dim == "rt":
        lo = (0e-6, -150e-15)
        hi = (35e-6, +150e-15)
        npoints = (200, 300)
    else:  # dim == "xyt":
        lo = (-35e-6, -35e-6, -150e-15)
        hi = (+35e-6, +35e-6, +150e-15)
        npoints = (200, 200, 300)
    return Laser(dim, lo, hi, npoints, get_spatial_chirp_profile())


def get_angular_dispersion_laser(dim):
    # - Cylindrical case
    if dim == "rt":
        lo = (0e-6, -150e-15)
        hi = (25e-6, +150e-15)
        npoints = (100, 300)
    else:  # dim == "xyt":
        lo = (-25e-6, -25e-6, -150e-15)
        hi = (+25e-6, +25e-6, +150e-15)
        npoints = (100, 100, 300)
    return Laser(dim, lo, hi, npoints, get_angular_dispersion_profile())


def test_laser_analysis_utils():
    """Test the different laser analysis utilities in both geometries."""
    for dim in ["xyt", "rt"]:
        laser = get_gaussian_laser(dim)

        # Check that energy computed from spectrum agrees with `compute_laser_energy`.
        spectrum, omega = get_spectrum(
            laser.grid, dim, is_envelope=True, omega0=laser.profile.omega0
        )
        d_omega = omega[1] - omega[0]
        spectrum_energy = np.sum(spectrum) * d_omega
        energy = compute_laser_energy(dim, laser.grid)
        np.testing.assert_approx_equal(spectrum_energy, energy, significant=10)

        # Check that laser central time agrees with the given one.
        t_peak_rms = get_t_peak(laser.grid, dim)
        np.testing.assert_approx_equal(t_peak_rms, laser.profile.t_peak, significant=3)

        # Check that laser duration agrees with the given one.
        tau_rms = get_duration(laser.grid, dim)
        np.testing.assert_approx_equal(2 * tau_rms, laser.profile.tau, significant=3)

        laser_sc = get_spatial_chirp_laser(dim)
        # Check that laser central time agrees with the given one with angular dispersion
        omega, freq_sc_rms = get_frequency(laser_sc.grid, dim)
        omega0 = 2 * np.pi * c / laser_sc.wavelength
        # freq_sc_expected = 0.0
        # np.testing.assert_approx_equal(freq_sc_rms, freq_sc_expected, significant=3)

        laser_ad = get_angular_dispersion_laser(dim)
        # Check that laser central time agrees with the given one with angular dispersion
        t_peak_ad_rms = get_t_peak(laser_ad.grid, dim)
        # np.testing.assert_approx_equal(t_peak_ad_rms, laser_ad.profile.a*laser_ad.profile.w0, significant=3)


if __name__ == "__main__":
    test_laser_analysis_utils()
