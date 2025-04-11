"""
Test the implementation of continuous wave laser and plane wave laser.

This test file verifies the correct implementation of these cases of the laser object and
checks the implementation of the peak_fluence and peak_power normalizations
as well as some measurement functionality from laser utils
"""

import numpy as np
from scipy.constants import c, epsilon_0

from lasy.laser import Laser
from lasy.profiles import CombinedLongitudinalTransverseProfile
from lasy.profiles.longitudinal import (
    ContinuousWaveProfile,
    GaussianLongitudinalProfile,
)
from lasy.profiles.transverse import GaussianTransverseProfile, PlaneWaveProfile
from lasy.utils.laser_utils import (
    get_duration,
    get_laser_power,
    get_w0,
)


def test_continuous_wave_laser():
    # Physical Parameters
    peak_fluence = 1e4  # J/m^2
    spot_size = 10e-3
    wavelength = 800e-9
    pol = (1, 0)

    long_prof = ContinuousWaveProfile(wavelength)
    tran_prof = GaussianTransverseProfile(spot_size)
    profile = CombinedLongitudinalTransverseProfile(
        wavelength, pol, long_prof, tran_prof, peak_fluence=peak_fluence
    )

    # Computational Grid
    dim = "xyt"
    lo = (-5 * spot_size, -5 * spot_size, None)
    hi = (5 * spot_size, 5 * spot_size, None)
    npoints = (1000, 1000, 1)

    laser = Laser(dim, lo, hi, npoints, profile)

    field = laser.grid.get_temporal_field()
    intensity = np.abs(epsilon_0 * field**2 / 2 * c)
    fluence = np.sum(intensity, axis=-1) * laser.grid.dx[-1]
    measured_peak_fluence = fluence.max()

    assert np.abs(measured_peak_fluence - peak_fluence) / peak_fluence < 1e-6
    assert np.abs(get_w0(laser.grid, laser.dim) - spot_size) / spot_size < 1e-6


def test_plane_wave_laser():
    # Physical Parameters
    tau = 30e-15
    wavelength = 800e-9
    t_peak = 0.0
    pol = (1, 0)
    peak_power = 1e12  # W

    long_prof = GaussianLongitudinalProfile(wavelength, tau, t_peak)
    tran_prof = PlaneWaveProfile()
    profile = CombinedLongitudinalTransverseProfile(
        wavelength, pol, long_prof, tran_prof, peak_power=peak_power
    )

    # Computational Grid
    dim = "xyt"
    lo = (None, None, -5 * tau)
    hi = (None, None, 5 * tau)
    npoints = (1, 1, 1000)

    laser = Laser(dim, lo, hi, npoints, profile)

    power = get_laser_power(laser.dim, laser.grid)
    measured_peak_power = power.max()

    assert np.abs(measured_peak_power - peak_power) / peak_power < 1e-6
    assert np.abs(2 * get_duration(laser.grid, laser.dim) - tau) / tau < 1e-6
