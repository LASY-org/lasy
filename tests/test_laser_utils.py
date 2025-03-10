import numpy as np
from scipy.constants import c, epsilon_0

from lasy.laser import Laser
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.utils.laser_utils import compute_laser_energy, get_duration, get_spectrum


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


def get_gaussian_laser(dim):
    # - Cylindrical case
    if dim == "rt":
        lo = (0e-6, -60e-15)
        hi = (25e-6, +60e-15)
        npoints = (100, 100)
    else:  # dim == "xyt":
        lo = (-25e-6, -25e-6, -60e-15)
        hi = (+25e-6, +25e-6, +60e-15)
        npoints = (100, 100, 100)
    return Laser(dim, lo, hi, npoints, get_gaussian_profile())


def test_laser_analysis_utils():
    """Test the different laser analysis utilities in both geometries."""
    for dim in ["xyt", "rt"]:
        laser = get_gaussian_laser(dim)

        # Check that energy computed from spectrum agrees with `compute_laser_energy`.
        spectrum, omega = get_spectrum(laser.grid, dim, omega0=laser.profile.omega0)
        d_omega = omega[1] - omega[0]
        spectrum_energy = np.sum(spectrum) * d_omega
        energy = compute_laser_energy(dim, laser.grid)
        np.testing.assert_approx_equal(spectrum_energy, energy, significant=10)

        # Check that laser duration agrees with the given one.
        tau_rms = get_duration(laser.grid, dim)
        np.testing.assert_approx_equal(2 * tau_rms, laser.profile.tau, significant=3)


def test_laser_normalization_utils():
    """Test the different laser normalization utilities in both geometries."""
    for dim in ["xyt", "rt"]:
        laser = get_gaussian_laser(dim)

        # Check energy normalization
        laser.normalize(1, kind="energy")
        energy = compute_laser_energy(dim, laser.grid)
        np.testing.assert_approx_equal(1, energy, significant=10)

        # Check peak field normalization
        laser.normalize(1, kind="field")
        field = laser.grid.get_temporal_field()
        np.testing.assert_approx_equal(1, np.abs(field.max()), significant=10)

        # Check peak intensity normalization
        laser.normalize(1, kind="intensity")
        field = laser.grid.get_temporal_field()
        intensity = np.abs(epsilon_0 * field**2 / 2 * c)
        np.testing.assert_approx_equal(1, intensity.max(), significant=10)

        # Check average intensity normalization
        laser.normalize(1, kind="average_intensity")
        field = laser.grid.get_temporal_field()
        intensity = np.abs(epsilon_0 * field**2 / 2 * c)
        np.testing.assert_approx_equal(1, intensity.mean(), significant=10)


if __name__ == "__main__":
    test_laser_analysis_utils()
    test_laser_normalization_utils()
