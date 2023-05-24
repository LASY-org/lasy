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


def check_correctness(laser_t_in, laser_t_out):
    z_axis = laser_t_in.box.axes[-1] * c
    laser_z = laser_t_in.export_to_z()
    laser_t_out.import_from_z(laser_z, z_axis)

    env_phase_t = -laser_t_in.profile.omega0 * laser_t_in.box.axes[-1]
    env_phase_z = laser_t_in.profile.omega0 / c * z_axis

    ind0 = laser_t_in.field.field.shape[0] // 2 - 1

    Field_orig = np.real(laser_t_in.field.field[ind0] * np.exp(1j * env_phase_t))
    Field_approx = np.real(laser_z[ind0] * np.exp(1j * env_phase_z))[:, ::-1]

    assert np.allclose(
        laser_t_in.field.field[ind0], laser_t_out.field.field[ind0], atol=1e-8
    )
    assert np.allclose(Field_approx, Field_orig, atol=2e-3)


def test_RT_case(gaussian):
    dim = "rt"
    w0 = gaussian.trans_profile.w0
    tau = gaussian.long_profile.tau
    lo = (0, -3.5 * tau)
    hi = (3 * w0, 3.5 * tau)
    npoints = (128, 64)

    laser_t_in = Laser(dim, lo, hi, npoints, gaussian)
    laser_t_out = Laser(dim, lo, hi, npoints, gaussian)
    check_correctness(laser_t_in, laser_t_out)


def test_3D_case(gaussian):
    # - 3D case
    dim = "xyt"
    w0 = gaussian.trans_profile.w0
    tau = gaussian.long_profile.tau
    lo = (-3 * w0, -3 * w0, -3.5 * tau)
    hi = (3 * w0, 3 * w0, 3.5 * tau)
    npoints = (128, 128, 64)
    laser = Laser(dim, lo, hi, npoints, gaussian)

    laser_t_in = Laser(dim, lo, hi, npoints, gaussian)
    laser_t_out = Laser(dim, lo, hi, npoints, gaussian)
    check_correctness(laser_t_in, laser_t_out)
