import numpy as np

from lasy.laser import Laser, Grid
from lasy.optical_elements import ParabolicMirror
from lasy.profiles.gaussian_profile import GaussianProfile
import matplotlib.pyplot as plt

wavelength = 0.8e-6
w0 = 5.0e-3  # m, initialized in near field

# The laser is initialized in the near field
pol = (1, 0)
laser_energy = 1.0  # J
t_peak = 0.0e-15  # s
tau = 30.0e-15  # s
gaussian_profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)


def get_w0(laser):
    A2 = (np.abs(laser.grid.field[0, :, :]) ** 2).sum(-1)
    ax = laser.grid.axes[0]
    if ax[0] > 0:
        A2 = np.r_[A2[::-1], A2]
        ax = np.r_[-ax[::-1], ax]
    else:
        A2 = np.r_[A2[::-1][:-1], A2]
        ax = np.r_[-ax[::-1][:-1], ax]

    sigma = 2 * np.sqrt(np.average(ax**2, weights=A2))

    return sigma


def check_resampling(laser):
    # focus down the laser and propagate using resampling 
    f0 = 2.0  # focal distance in m
    laser.apply_optics(ParabolicMirror(f=f0))
    
    new_r_max = 300.e-6
    new_grid = Grid(dim, lo, (new_r_max, hi[1]), npoints, n_azimuthal_modes=laser.grid.n_azimuthal_modes)    
    laser.propagate((f0), grid=new_grid) #resample the radial grid
    
    # Check that the value is the expected one in the near field
    w0_num = get_w0(laser)
    w0_theor = wavelength * f0 / (np.pi * w0)
    err = 2 * np.abs(w0_theor - w0_num) / (w0_theor + w0_num)
    print(err)
    assert err < 1e-3
    
def test_resampling():
    #define the grid for the laser
    dim = "rt"
    lo = (0e-3, -90e-15)
    hi = (15e-3, +90e-15)
    npoints = (250, 30)

    laser = Laser(dim, lo, hi, npoints, gaussian_profile)
    check_resampling(laser)