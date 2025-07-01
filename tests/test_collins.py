import numpy as np
from scipy.constants import c

from lasy.laser import Laser
from lasy.profiles import GaussianProfile
from lasy.propagators import ABCD, CollinsDFFTPropagator, CollinsSFFTPropagator
from lasy.utils.laser_utils import get_w0


def make_laser():
    profile = GaussianProfile(
        wavelength=800e-9,
        pol=(1, 0),
        laser_energy=1,
        tau=30e-15 / np.sqrt(2 * np.log(2)),
        w0=500e-6,
        t_peak=0,
    )

    dim = "xyt"
    hi = (2e-3, 2e-3, 50e-15)
    lo = (-2e-3, -2e-3, -50e-15)
    npoints = (49, 49, 100)

    laser = Laser(dim=dim, hi=hi, lo=lo, npoints=npoints, profile=profile)

    return laser


def test_spatial_propagation_SFFT():
    """Verify that the waist of Gaussian beam evolves as expected."""
    laser = make_laser()
    prop = CollinsSFFTPropagator(
        dim=laser.dim,
        omega0=2 * np.pi * c / 800e-9,
    )

    focal_length = 1.0

    zR = (
        focal_length**2 * laser.profile.lambda0 / (np.pi * laser.profile.w0**2)
    )  # Estimated Rayleigh range
    w0 = (
        laser.profile.lambda0 * focal_length / (np.pi * laser.profile.w0)
    )  # Estimated focal spot-size
    print(zR, w0)

    z_pos = (
        np.linspace(-10.0 * zR, 10.0 * zR, 5) + focal_length
    )  # Absolute position from lens
    waists_propagated = []

    i = 0
    abcd = ABCD()
    abcd.add_lens(focal_length)
    for z in z_pos:
        laser = make_laser()
        if i == 0:
            abcd.add_vacuum(z)
            prop.propagate(laser.grid, abcd)
            grid_out = laser.grid
        else:
            abcd.add_vacuum(z - z_pos[i - 1])
            prop.propagate(laser.grid, abcd, grid_out=grid_out)

        waist = get_w0(grid=laser.grid, dim=laser.dim)
        waists_propagated.append(waist)

        laser.show()
        i += 1

    waists_analytical = w0 * np.sqrt(1 + ((z_pos - focal_length) / zR) ** 2)
    print(waists_propagated, waists_analytical, laser.profile.w0)

    assert np.allclose(waists_propagated, waists_analytical, rtol=1e-5, atol=1e-6)


def test_spatial_propagation_DFFT():
    """Verify that the waist of Gaussian beam evolves as expected."""
    laser = make_laser()
    prop = CollinsDFFTPropagator(
        dim=laser.dim,
        omega0=2 * np.pi * c / 800e-9,
    )

    z_pos = np.linspace(-50e-3, 50e-3, 5)
    waists_propagated = []

    for z in z_pos:
        abcd = ABCD()
        abcd.add_vacuum(z)

        laser = make_laser()
        laser.add_propagator(prop)

        prop.propagate(laser.grid, abcd, grid_out=laser.grid)

        waist = get_w0(grid=laser.grid, dim=laser.dim)
        waists_propagated.append(waist)

        laser.show()

    zR = np.pi * laser.profile.w0**2 / (laser.profile.lambda0)
    waists_analytical = laser.profile.w0 * np.sqrt(1 + (z_pos / zR) ** 2)
    print(waists_propagated - waists_analytical, laser.profile.w0)

    assert np.allclose(waists_propagated, waists_analytical, rtol=1e-5, atol=1e-6)
