# -*- coding: utf-8 -*-

import numpy as np
import pytest
from scipy.constants import c

from lasy.laser import Laser
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.utils.laser_utils import export_to_z, import_from_z


@pytest.fixture(scope="function")
def gaussian():
    # Cases with Gaussian laser
    wavelength = 0.8e-6
    pol = (1, 0)
    laser_energy = 1.0  # J
    t_peak = 0.0e-15  # s
    tau = 200.0e-15  # s
    w0 = 5.0e-6  # m
    profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)

    return profile


def get_laser_z_analytic(profile, z_axis, r_axis):
    w0 = profile.w0
    tau = profile.tau
    omega0 = profile.omega0
    k0 = omega0 / c
    lambda0 = 2 * np.pi / k0

    L_Ray = np.pi * w0**2 / lambda0
    z_axis_2d = z_axis[None, :]
    r_axis_2d = r_axis[:, None]
    w0_z = w0 * np.sqrt(1 + (z_axis_2d / L_Ray) ** 2)
    R_z_inv = z_axis_2d / (z_axis_2d**2 + L_Ray**2)
    phi_gouy = np.arctan2(z_axis_2d, L_Ray)

    Field = (
        w0
        / w0_z
        * np.exp(-(r_axis_2d**2) / w0_z**2)
        * np.exp(-(z_axis_2d**2) / (c * tau) ** 2)
        * np.exp(1j * (k0 * r_axis_2d**2 * R_z_inv / 2 - phi_gouy))
    )

    return Field


def check_correctness(laser_t_in, laser_t_out, laser_z_analytic, z_axis):
    laser_z = export_to_z(laser_t_in.dim, laser_t_in.grid, laser_t_in.profile.omega0)
    import_from_z(
        laser_t_out.dim, laser_t_out.grid, laser_t_out.profile.omega0, laser_z, z_axis
    )

    field = laser_t_in.grid.get_temporal_field()
    ind0 = field.shape[0] // 2 - 1

    laser_t_in_2d = field[ind0]
    laser_t_out_2d = field[ind0]
    laser_z_2d = laser_z[ind0]

    assert np.allclose(laser_t_in_2d, laser_t_out_2d, atol=2e-7, rtol=0)

    assert np.allclose(laser_z_2d, laser_z_analytic, atol=1e-3, rtol=0)


def test_RT_case(gaussian):
    dim = "rt"
    w0 = gaussian.w0
    tau = gaussian.tau
    lo = (0, -3.5 * tau)
    hi = (5 * w0, 3.5 * tau)
    npoints = (128, 65)

    laser_t_in = Laser(dim, lo, hi, npoints, gaussian)
    laser_t_out = Laser(dim, lo, hi, npoints, gaussian)
    laser_t_in.normalize(1.0, "field")
    laser_t_out.normalize(1.0, "field")

    t_axis = laser_t_in.grid.axes[-1]
    r_axis = laser_t_in.grid.axes[0]
    z_axis = t_axis * c

    laser_z_analytic = get_laser_z_analytic(gaussian, z_axis, r_axis)

    check_correctness(laser_t_in, laser_t_out, laser_z_analytic, z_axis)


def test_3D_case(gaussian):
    # - 3D case
    dim = "xyt"
    w0 = gaussian.w0
    tau = gaussian.tau
    lo = (-5 * w0, -5 * w0, -3.5 * tau)
    hi = (5 * w0, 5 * w0, 3.5 * tau)
    npoints = (160, 164, 65)

    laser_t_in = Laser(dim, lo, hi, npoints, gaussian)
    laser_t_out = Laser(dim, lo, hi, npoints, gaussian)
    laser_t_in.normalize(1.0, "field")
    laser_t_out.normalize(1.0, "field")

    t_axis = laser_t_in.grid.axes[-1]
    r_axis = np.abs(laser_t_in.grid.axes[1])
    z_axis = t_axis * c

    laser_z_analytic = get_laser_z_analytic(gaussian, z_axis, r_axis)

    check_correctness(laser_t_in, laser_t_out, laser_z_analytic, z_axis)
