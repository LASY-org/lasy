import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy.constants import c, epsilon_0

from lasy.laser import Laser
from lasy.optical_elements import Axicon
from lasy.profiles.gaussian_profile import GaussianProfile

wavelength = 0.8e-6
w0 = 10.0e-5  # m, initialized in near field
pol = (1, 0)
laser_energy = 1.0  # J
t_peak = 0.0e-15  # s
tau = 30.0e-15  # s
gaussian_profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)
gamma = 2.2 * np.pi / 180
z_max = w0 * np.cos(gamma) / np.sin(gamma)


def check_axicon(laser):
    distances = np.linspace(z_max / 1.5, z_max, 2)

    errors = []
    for dist in distances:
        # create a new laser for each distance
        new_laser = Laser(
            laser.dim,
            laser.grid.lo,
            laser.grid.hi,
            laser.grid.npoints,
            gaussian_profile,
        )

        new_laser.apply_optics(Axicon(gamma=gamma))
        new_laser.propagate(dist)
        # Check the Bessel profile
        error = check_bessel_profile(new_laser, dist)
        errors.append(error)
        print(errors)
    assert np.max(errors) < 1e-2


def check_bessel_profile(laser, z):
    # Calculate the laser profile
    field = laser.grid.get_temporal_field()
    if laser.dim == "xyt":
        Nx = field.shape[0]
        A2 = 0.5 * epsilon_0 * c * (np.abs(field[Nx // 2 - 1, :, :]) ** 2).max(-1)
        ax = laser.grid.axes[1]
    else:
        A2 = 0.5 * epsilon_0 * c * (np.abs(field[0, :, :]) ** 2).sum(-1)
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
        * np.exp(-2 * z**2 / (z_max**2))
        * z
        / z_max
        * special.j0(kr) ** 2
    )

    # Normalize profiles
    A2_norm = A2 / np.max(A2)
    expected_profile_norm = expected_profile / np.max(expected_profile)
    # Calculate error
    error1 = np.mean(np.abs(expected_profile_norm - A2_norm))
    plt.figure(figsize=(10, 6))
    plt.plot(ax, expected_profile, label="Expected Profile")
    plt.plot(ax, A2, label="Actual Profile")
    plt.xlim(-1e-4, 1e-4)
    plt.xlabel("Radial distance (m)")
    plt.ylabel("Intensity")
    plt.title(f"Bessel Profile Comparison at z = {z:.2e} m")
    plt.legend()

    # Save the plot
    plt.savefig(f"bessel_profile_z_{z:.2e}.png")
    plt.close()
    return error1  # Return the smallest error


def test_3D_case():
    # 3D case
    dim = "xyt"
    lo = (-20e-5, -20e-5, -100e-15)
    hi = (+20e-5, +20e-5, +100e-15)
    npoints = (1000, 1000, 150)

    laser = Laser(dim, lo, hi, npoints, gaussian_profile)
    check_axicon(laser)


def test_RT_case():
    # Cylindrical case
    dim = "rt"
    lo = (0e-6, -60e-15)
    hi = (15e-3, +60e-15)
    npoints = (1100, 300)

    laser = Laser(dim, lo, hi, npoints, gaussian_profile)
    check_axicon(laser)
