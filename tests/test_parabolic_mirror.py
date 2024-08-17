# -*- coding: utf-8 -*-
"""Thest the parabolic mirror implementation.

Test checks the implementation of the parabolic mirror
by initializing a Gaussian pulse in the near field, and
propagating it through a parabolic mirror, and then to
the focal position ; we then check that the waist as the
expected value in the far field (i.e. in the focal plane).
"""

import numpy as np

from lasy.laser import Laser
from lasy.optical_elements import ParabolicMirror
from lasy.profiles.gaussian_profile import GaussianProfile

wavelength = 0.8e-6
w0 = 5.0e-3  # m, initialized in near field

# The laser is initialized in the near field
pol = (1, 0)
laser_energy = 1.0  # J
t_peak = 0.0e-15  # s
tau = 30.0e-15  # s
gaussian_profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)


def get_w0(laser):
    # Calculate the laser waist
    field = laser.grid.get_temporal_field()
    if laser.dim == "xyt":
        Nx = field.shape[0]
        A2 = (np.abs(field[Nx // 2 - 1, :, :]) ** 2).sum(-1)
        ax = laser.grid.axes[1]
    else:
        A2 = (np.abs(field[0, :, :]) ** 2).sum(-1)
        ax = laser.grid.axes[0]
        if ax[0] > 0:
            A2 = np.r_[A2[::-1], A2]
            ax = np.r_[-ax[::-1], ax]
        else:
            A2 = np.r_[A2[::-1][:-1], A2]
            ax = np.r_[-ax[::-1][:-1], ax]

    sigma = 2 * np.sqrt(np.average(ax**2, weights=A2))

    return sigma


def check_parabolic_mirror(laser):
    # Propagate laser after parabolic mirror + vacuum
    f0 = 8.0  # focal distance in m
    laser.apply_optics(ParabolicMirror(f=f0))
    laser.propagate(f0)
    # Check that the value is the expected one in the near field
    w0_num = get_w0(laser)
    w0_theor = wavelength * f0 / (np.pi * w0)
    err = 2 * np.abs(w0_theor - w0_num) / (w0_theor + w0_num)
    assert err < 1e-3


def test_3D_case():
    # - 3D case
    # The laser is initialized in the near field
    dim = "xyt"
    lo = (-12e-3, -12e-3, -60e-15)
    hi = (+12e-3, +12e-3, +60e-15)
    npoints = (500, 500, 100)

    laser = Laser(dim, lo, hi, npoints, gaussian_profile)
    check_parabolic_mirror(laser)


def test_RT_case():
    # - Cylindrical case
    # The laser is initialized in the near field
    dim = "rt"
    lo = (0e-6, -60e-15)
    hi = (15e-3, +60e-15)
    npoints = (750, 100)

    laser = Laser(dim, lo, hi, npoints, gaussian_profile)
    check_parabolic_mirror(laser)
