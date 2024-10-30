import numpy as np
import scipy.constants as scc

from lasy.laser import Laser
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.utils.laser_utils import get_STC

# Create profile.
profile = GaussianProfile(
    wavelength=0.6e-6,  # m
    pol=(1, 0),
    laser_energy=1.0,  # J
    w0=5e-6,  # m
    tau=5e-14,  # s
    t_peak=0.0,  # s
    beta=0,
    zeta=2.4e-22,
    phi2=2.4e-22,
    stc_theta=scc.pi / 2,
)
# Create laser with given profile in `rt` geometry.
laser = Laser(
    dim="xyt",
    lo=(-10e-6, -10e-6, -10e-14),
    hi=(10e-6, 10e-6, +10e-14),
    npoints=(50, 60, 70),
    profile=profile,
)
STC = get_STC(laser.dim, laser.grid, k0=2 * scc.pi / 0.6e-6)
np.testing.assert_approx_equal(STC["phi2"], 2.4e-22, significant=2)
np.testing.assert_approx_equal(STC["zeta"], 2.4e-22, significant=2)
