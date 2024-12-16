import numpy as np
import scipy.constants as scc

from lasy.laser import Laser
from lasy.profiles.gaussian_profile import STCGaussianProfile
from lasy.utils.laser_utils import get_Phi2, get_Zeta

# Create profile.
profile = STCGaussianProfile(
    wavelength=0.6e-6,  # m
    pol=(1, 0),
    laser_energy=1.0,  # J
    w0=5e-6,  # m
    tau=5e-14,  # s
    t_peak=0.0,  # s
    beta=0,
    zeta=2.4e-22,
    phi2=2.4e-24,
    stc_theta=scc.pi / 2,
)
# Create laser with given profile in `xyt` geometry.
laser_3d = Laser(
    dim="xyt",
    lo=(-10e-6, -10e-6, -10e-14),
    hi=(10e-6, 10e-6, +10e-14),
    npoints=(50, 60, 200),
    profile=profile,
)
# Create laser with given profile in `rt` geometry.
laser_2d = Laser(
    dim="rt",
    lo=(-10e-6, -10e-14),
    hi=(10e-6, +10e-14),
    npoints=(60, 200),
    profile=profile,
)

Phi2_3d, phi2_3d = get_Phi2(laser_3d.dim, laser_3d.grid)
Phi2_2d, phi2_2d = get_Phi2(laser_2d.dim, laser_2d.grid)
[zeta_x, zeta_y], [nu_x, nu_y] = get_Zeta(laser_3d.dim, laser_3d.grid, 2.0 * np.pi / 0.6e-6)
np.testing.assert_approx_equal(phi2_3d, 2.4e-24, significant=2)
np.testing.assert_approx_equal(phi2_2d, 2.4e-24, significant=2)

np.testing.assert_approx_equal(zeta_y, 2.4e-24, significant=2)
# np.testing.assert_approx_equal(STC_3d["phi2"], 2.4e-19, significant=2)
# np.testing.assert_approx_equal(STC_3d["beta_y"], 3e-18, significant=2)
# np.testing.assert_approx_equal(STC_3d["zeta_y"], 2.4e-24, significant=2)
