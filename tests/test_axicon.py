import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy.constants import c, epsilon_0

from lasy.laser import Laser
from lasy.optical_elements import Axicon
from lasy.profiles.gaussian_profile import GaussianProfile

wavelength = 0.8e-6
w0 = 500.0e-5  # m, initialized in near field
pol = (1, 0)
laser_energy = 150.0e-3  # J
t_peak = 0.0e-15  # s
tau = 50.0e-15  # s
gaussian_profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)
gamma = 2.3 * np.pi / 180
z_max = w0 * np.cos(gamma) / np.sin(gamma)


def check_axicon(laser):
    # Propagate only to z_max
    new_laser = Laser(
        laser.dim, laser.grid.lo, laser.grid.hi, laser.grid.npoints, gaussian_profile
    )
    new_laser.apply_optics(Axicon(gamma=gamma))
    new_laser.propagate(z_max)

    # Check the Bessel profile
    error = check_bessel_profile(new_laser, z_max)
    print(f"Error at z_max = {z_max:.2e} m: {error:.4f}")
    assert error < 1.4e-2


def check_bessel_profile(laser, z):
    # Calculate the laser profile
    field = laser.grid.get_temporal_field()
    if laser.dim == "xyt":
        Nx = field.shape[0]
        A2 = 0.5 * epsilon_0 * c * (np.abs(field[Nx // 2 - 1, :, :]) ** 2).max(-1)
        ax = laser.grid.axes[1]
    else:
        A2 = 0.5 * epsilon_0 * c * (np.abs(field[0, :, :]) ** 2).max(-1)
        ax = laser.grid.axes[0]
        if ax[0] > 0:
            A2 = np.r_[A2[::-1], A2]
            ax = np.r_[-ax[::-1], ax]
        else:
            A2 = np.r_[A2[::-1][:-1], A2]
            ax = np.r_[-ax[::-1][:-1], ax]

    # Calculate the expected Bessel profile
    k = 2 * np.pi / wavelength
    r = np.abs(ax)
    kr = k * np.sin(gamma) * r
    power = laser_energy / (np.sqrt(np.pi / 2) * tau)
    expected_profile = (
        4
        * power
        * k
        * np.sin(gamma)
        / w0
        * np.exp(-2 * z**2 / z_max**2)
        * z
        / z_max
        * special.j0(kr) ** 2
    )

    # Normalize profiles

    # Calculate error using normalized profiles
    error = np.max(np.abs(expected_profile - A2)) / abs(A2).max()
    max_error_index = np.argmax(error)
    max_error_location = ax[max_error_index]

    return error


def test_RT_case():
    # Cylindrical case
    dim = "rt"

    original_waist = 10.0e-5
    new_waist = w0
    scale_factor = new_waist / original_waist
    nr = int(600 * scale_factor)
    npoints = (nr, 100)
    lo = (0, -100e-15)
    hi = (new_waist * 5, +100e-15)
    laser = Laser(dim, lo, hi, npoints, gaussian_profile)
    check_axicon(laser)
