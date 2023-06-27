# -*- coding: utf-8 -*-

import pytest

import numpy as np
from lasy.laser import Laser
from lasy.profiles.gaussian_profile import GaussianProfile
from scipy.constants import c


@pytest.fixture(scope="function")
def gaussian():
    # Cases with Gaussian laser
    wavelength = 0.8e-6
    pol = (1, 0)
    laser_energy = 1.0  # J
    t_peak = 0.0e-15  # s
    tau = 30.0e-15  # s
    w0 = 25.0e-6  # m
    profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)

    return profile


def get_laser_z_analytic(profile, z_axis, r_axis):
    w0 = profile.trans_profile.w0
    tau = profile.long_profile.tau
    omega0 = profile.long_profile.omega0
    k0 = omega0 / c
    lambda0 = 2 * np.pi / k0

    L_Ray = np.pi * w0**2 / lambda0
    z_axis_2d = z_axis[None, :]
    r_axis_2d = r_axis[:, None]
    w0_z = w0 * np.sqrt(1 + (z_axis_2d / L_Ray) ** 2)
    R_z_inv = z_axis_2d / (z_axis_2d**2 + L_Ray**2)
    phi_goy = np.arctan2(z_axis_2d, L_Ray)

    Field = (
        w0
        / w0_z
        * np.exp(-(r_axis_2d**2) / w0_z**2)
        * np.exp(-(z_axis_2d**2) / (c * tau) ** 2)
        * np.exp(1j * (k0 * r_axis_2d**2 * R_z_inv / 2 - phi_goy))
    )

    return Field


def check_correctness(laser_t_in, laser_t_out, laser_z_analytic, z_axis):
    laser_z = laser_t_in.export_to_z()
    laser_t_out.import_from_z(laser_z, z_axis)

    ind0 = laser_t_in.field.field.shape[0] // 2 - 1

    laser_t_in_2d = laser_t_in.field.field[ind0]
    laser_t_out_2d = laser_t_out.field.field[ind0]
    laser_z_2d = laser_z[ind0]

    assert np.allclose(laser_t_in_2d, laser_t_out_2d, atol=3e-9)

    assert np.allclose(laser_z_2d, laser_z_analytic, atol=6e-4)


def test_RT_case(gaussian):
    dim = "rt"
    w0 = gaussian.trans_profile.w0
    tau = gaussian.long_profile.tau
    lo = (0, -3.5 * tau)
    hi = (3 * w0, 3.5 * tau)
    npoints = (128, 64)

    laser_t_in = Laser(dim, lo, hi, npoints, gaussian)
    laser_t_out = Laser(dim, lo, hi, npoints, gaussian)

    t_axis = laser_t_in.box.axes[-1]
    r_axis = laser_t_in.box.axes[0]
    z_axis = t_axis * c

    laser_z_analytic = get_laser_z_analytic(gaussian, z_axis, r_axis)

    check_correctness(laser_t_in, laser_t_out, laser_z_analytic, z_axis)


def test_3D_case(gaussian):
    # - 3D case
    dim = "xyt"
    w0 = gaussian.trans_profile.w0
    tau = gaussian.long_profile.tau
    lo = (-3 * w0, -3 * w0, -3.5 * tau)
    hi = (3 * w0, 3 * w0, 3.5 * tau)
    npoints = (128, 128, 64)

    laser_t_in = Laser(dim, lo, hi, npoints, gaussian)
    laser_t_out = Laser(dim, lo, hi, npoints, gaussian)

    t_axis = laser_t_in.box.axes[-1]
    r_axis = np.abs(laser_t_in.box.axes[1])
    z_axis = t_axis * c

    laser_z_analytic = get_laser_z_analytic(gaussian, z_axis, r_axis)

    check_correctness(laser_t_in, laser_t_out, laser_z_analytic, z_axis)
