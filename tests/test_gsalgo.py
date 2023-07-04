# -*- coding: utf-8 -*-

import pytest

import numpy as np
from lasy.laser import Laser
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.utils.zernike import zernike
from lasy.utils.phase_retrieval import gerchberg_saxton_algo
import copy

w0 = 25.0e-6  # m


@pytest.fixture(scope="function")
def gaussian():
    # Cases with Gaussian laser
    wavelength = 0.8e-6
    pol = (1, 0)
    laser_energy = 1.0e-3  # J
    t_peak = 0.0  # s
    tau = 30.0e-15  # s
    profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)

    return profile


def test_3D_case(gaussian):
    # - 3D case
    dim = "xyt"
    lo = (-75e-6, -75e-6, -50e-15)
    hi = (75e-6, 75e-6, 50e-15)
    npoints = (100, 100, 100)

    laser = Laser(dim, lo, hi, npoints, gaussian)

    # Add a phase aberration
    # CALCULATE THE REQUIRED PHASE ABERRATION
    x = np.linspace(lo[0], hi[0], npoints[0])
    y = np.linspace(lo[1], hi[1], npoints[1])
    X, Y = np.meshgrid(x, y)
    pupilRadius = 2 * w0
    phase = -0.2 * zernike(X, Y, (0, 0, pupilRadius), 3)

    R = np.sqrt(X**2 + Y**2)
    phaseMask = np.ones_like(phase)
    phaseMask[R > pupilRadius] = 0

    # NOW ADD THE PHASE TO EACH SLICE OF THE FOCUS
    phase3D = np.repeat(phase[:, :, np.newaxis], npoints[2], axis=2)
    laser.grid.field = np.abs(laser.grid.field) * np.exp(1j * phase3D)

    # PROPAGATE THE FIELD FIELD FOWARDS AND BACKWARDS BY 1 MM
    propDist = 2e-3
    laserForward = copy.deepcopy(laser)
    laserForward.propagate(propDist)
    laserBackward = copy.deepcopy(laser)
    laserBackward.propagate(-propDist)

    # PERFORM GERCHBERG-SAXTON ALGORTIHM TO RETRIEVE PHASE
    _, _, amp_error = gerchberg_saxton_algo(
        laserBackward,
        laserForward,
        2 * propDist,
        condition="max_iterations",
        max_iterations=100,
        debug=True,
    )

    assert amp_error < 1e-6
