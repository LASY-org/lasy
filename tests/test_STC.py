"""Test the implementation of the spatio-temporal coupling.

Test checks the implementation of the initialization and diagnostics to spatio-temporal coupling gaussian lasers
by creating a gaussian pulse on focus and calculate the STC factors by the implemented functions in laser.utils.
The correctness is also checked through comparing the gaussian profile and a combined gaussian profile off-focus.
"""

import numpy as np
import scipy.constants as scc

from lasy.laser import Laser
from lasy.profiles.combined_profile import CombinedLongitudinalTransverseProfile
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.profiles.longitudinal import GaussianLongitudinalProfile
from lasy.profiles.transverse import GaussianTransverseProfile
from lasy.utils.laser_utils import get_beta, get_phi2, get_zeta

wavelength = 0.6e-6  # m
pol = (1, 0)
laser_energy = 1.0  # J
w0 = 5e-6  # m
tau = 5e-14  # s
t_peak = 0.0  # s
beta = 3e-18  # s
zeta = 2.4e-22  # m * s
phi2 = 2.4e-24  # s ^ 2
stc_theta = scc.pi / 2  # rad
z_r = (np.pi * w0**2) / wavelength
z_foc = 3 * z_r
# Create STC profile.
profile = GaussianProfile(
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
    lo=(-10e-6, -10e-6, -10e-14),
    hi=(10e-6, 10e-6, +10e-14),
    npoints=(100, 100, 200),
    profile=profile,
)

# Create laser with given profile in `rt` geometry.
long_profile = GaussianLongitudinalProfile(wavelength, tau, t_peak)
trans_profile = GaussianTransverseProfile(w0, wavelength, z_foc)
combined_profile = CombinedLongitudinalTransverseProfile(
    wavelength, pol, laser_energy, long_profile, trans_profile
)

profile_gaussian = GaussianProfile(
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
    profile=profile_gaussian,
)
env_combined = (laser_2d_combined.grid.get_temporal_field())
env_gaussian = (laser_2d_gaussian.grid.get_temporal_field())
err_real = np.average(np.array(env_combined.real)-np.array(env_gaussian.real))

Phi2_3d, phi2_3d = get_phi2(laser_3d.dim, laser_3d.grid)

[zeta_x, zeta_y], [nu_x, nu_y] = get_zeta(
    laser_3d.dim, laser_3d.grid, 2.0 * np.pi / 0.6e-6
)
[beta_x, beta_y] = get_beta(laser_3d.dim, laser_3d.grid, 2.0 * np.pi / 0.6e-6)
np.testing.assert_approx_equal(err_real, 1e-3, significant=1)
np.testing.assert_approx_equal(phi2_3d, 2.4e-24, significant=2)
np.testing.assert_approx_equal(zeta_y, 2.4e-22, significant=2)
np.testing.assert_approx_equal(beta_y, 3e-18, significant=2)
