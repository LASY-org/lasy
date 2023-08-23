import numpy as np

from lasy.laser import Laser
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.utils.laser_utils import get_spectrum, compute_laser_energy, get_duration


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
        spectrum, omega = get_spectrum(
            laser.grid, dim, is_envelope=True, omega0=laser.profile.omega0
        )
        d_omega = omega[1] - omega[0]
        spectrum_energy = np.sum(spectrum) * d_omega
        energy = compute_laser_energy(dim, laser.grid)
        np.testing.assert_approx_equal(spectrum_energy, energy, significant=10)

        # Check that laser duration agrees with the given one.
        tau_rms = get_duration(laser.grid, dim)
        np.testing.assert_approx_equal(
            2 * tau_rms, laser.profile.long_profile.tau, significant=3
        )


if __name__ == "__main__":
    test_laser_analysis_utils()
