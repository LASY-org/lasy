"""
Test the implementation of spatio-temporal coupling (STC).

This test verifies the correct implementation of initialization and diagnostics for spatio-temporal coupling in Gaussian lasers. It does so by creating a Gaussian pulse at focus and calculating the STC factors using the implemented functions in `laser.utils`.

Additionally, the correctness is validated by comparing the Gaussian profile with a combined Gaussian profile without STC out of focus.
"""

import numpy as np
import scipy.constants as scc

from lasy.laser import Laser
from lasy.profiles.combined_profile import CombinedLongitudinalTransverseProfile
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.profiles.longitudinal import GaussianLongitudinalProfile
from lasy.profiles.transverse import GaussianTransverseProfile
from lasy.utils.laser_utils import get_beta, get_dispersion, get_zeta

wavelength = 0.6e-6  # m
pol = (1, 0)
laser_energy = 1.0  # J
w0 = 5e-6  # m
tau = 5e-14  # s
t_peak = 0.0  # s
beta = 3e-18  # s
zeta = 2.4e-22  # m * s
phi2 = 2.4e-27  # s ^ 2
stc_theta = scc.pi / 2  # rad
z_r = (np.pi * w0**2) / wavelength
z_foc = 3 * z_r

# Create STC profile.
STCprofile = GaussianProfile(
    wavelength=wavelength,
    pol=pol,
    laser_energy=laser_energy,
    w0=w0,
    tau=tau,
    t_peak=t_peak,
    beta=beta,
    zeta=zeta,
    phi2=phi2,
    stc_theta=stc_theta,
)

# Create laser with given profile in `xyt` geometry.
laser_3d = Laser(
    dim="xyt",
    lo=(-10e-6, -10e-6, -30e-14),
    hi=(10e-6, 10e-6, +30e-14),
    npoints=(100, 100, 200),
    profile=STCprofile,
)

# Create laser out of focus with given profile in `rt` geometry by combined and gaussian profile.
long_profile = GaussianLongitudinalProfile(wavelength, tau, t_peak)
trans_profile = GaussianTransverseProfile(w0, wavelength, z_foc)

combined_profile = CombinedLongitudinalTransverseProfile(
    wavelength, pol, long_profile, trans_profile, laser_energy=laser_energy
)

gaussian_profile = GaussianProfile(
    wavelength=wavelength,
    pol=pol,
    laser_energy=laser_energy,
    w0=w0,
    tau=tau,
    t_peak=t_peak,
    z_foc=z_foc,
)

laser_2d_combined = Laser(
    dim="rt",
    lo=(0e-6, -10e-14),
    hi=(50e-6, +10e-14),
    npoints=(60, 200),
    profile=combined_profile,
)

laser_2d_gaussian = Laser(
    dim="rt",
    lo=(0e-6, -10e-14),
    hi=(50e-6, +10e-14),
    npoints=(60, 200),
    profile=gaussian_profile,
)

# calculate the error of gaussian profile
env_combined = laser_2d_combined.grid.get_temporal_field()
env_gaussian = laser_2d_gaussian.grid.get_temporal_field()

err_real = np.average(
    (np.array(env_combined.real) - np.array(env_gaussian.real))
    / np.array(env_combined.real)
)
err_imag = np.average(
    (np.array(env_combined.imag) - np.array(env_gaussian.imag))
    / np.array(env_combined.imag)
)

gdd_3d, gdd0_3d = get_dispersion(
    grid=laser_3d.grid, dim=laser_3d.dim, omega0=laser_3d.profile.omega0, order=2
)
[zeta_x, zeta_y], [nu_x, nu_y] = get_zeta(
    laser_3d.dim, laser_3d.grid, 2.0 * np.pi / 0.6e-6
)
[beta_x, beta_y] = get_beta(laser_3d.dim, laser_3d.grid, 2.0 * np.pi / 0.6e-6)

assert (err_real + err_imag) < 1e-6

np.testing.assert_approx_equal(gdd0_3d, phi2, significant=2)
np.testing.assert_approx_equal(zeta_y, zeta, significant=2)
np.testing.assert_approx_equal(beta_y, beta, significant=2)
