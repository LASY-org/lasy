# -*- coding: utf-8 -*-

import numpy as np
import pytest

from lasy.laser import Laser
from lasy.profiles.gaussian_profile import GaussianProfile


@pytest.fixture(scope="function")
def gaussian():
    # Cases with Gaussian laser
    wavelength = 0.8e-6
    pol = (1, 0)
    laser_energy = 1.0  # J
    t_peak = 0.0e-15  # s
    tau = 30.0e-15  # s
    w0 = 5.0e-6  # m
    profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)

    return profile


def get_w0(laser):
    # Calculate the laser waist
    field = laser.grid.get_temporal_field()
    if laser.dim == "xyt":
        Nx, Ny, Nt = field.shape
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


def check_gaussian_propagation(
    laser, propagation_distance=100e-6, propagation_step=10e-6
):
    # Do the propagation and check evolution of waist with theory
    w0 = laser.profile.w0
    L_R = np.pi * w0**2 / laser.profile.lambda0

    propagated_distance = 0.0
    while propagated_distance <= propagation_distance:
        propagated_distance += propagation_step
        laser.propagate(
            propagation_step,
        )
        w0_num = get_w0(laser)
        w0_theor = w0 * np.sqrt(1 + (propagated_distance / L_R) ** 2)
        err = 2 * np.abs(w0_theor - w0_num) / (w0_theor + w0_num)
        assert err < 1e-3


def test_3D_case(gaussian):
    # - 3D case
    dim = "xyt"
    lo = (-25e-6, -25e-6, -60e-15)
    hi = (+25e-6, +25e-6, +60e-15)
    npoints = (100, 102, 100)

    laser = Laser(dim, lo, hi, npoints, gaussian)
    check_gaussian_propagation(laser)


def test_RT_case(gaussian):
    # - Cylindrical case
    dim = "rt"
    lo = (0e-6, -60e-15)
    hi = (25e-6, +60e-15)
    npoints = (100, 100)

    laser = Laser(dim, lo, hi, npoints, gaussian)
    check_gaussian_propagation(laser)


def test_RT_case_multimode(gaussian):
    # - Cylindrical case
    dim = "rt"
    lo = (0e-6, -60e-15)
    hi = (25e-6, +60e-15)
    npoints = (100, 100)

    # Note: using 3 modes and 20 points is unnecessary here,
    # since the profile is purely cylindrical. This is
    # done here only to make sure that the code is robust.
    laser = Laser(dim, lo, hi, npoints, gaussian, n_azimuthal_modes=3, n_theta_evals=20)
    check_gaussian_propagation(laser)
