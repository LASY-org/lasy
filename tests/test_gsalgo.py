# -*- coding: utf-8 -*-

import copy

import numpy as np
import pytest

from lasy.laser import Laser
from lasy.profiles import CombinedLongitudinalTransverseProfile
from lasy.profiles.longitudinal import ContinuousWaveProfile
from lasy.profiles.transverse import GaussianTransverseProfile
from lasy.utils.phase_retrieval import GerchbergSaxton
from lasy.utils.zernike import zernike


@pytest.fixture(scope="function")
def gaussian():
    peak_fluence = 1.0  # J/m^2
    spot_size = 10e-6
    wavelength = 800e-9
    omega0 = 2 * np.pi * c / wavelength
    pol = (1, 0)

    long_prof = ContinuousWaveProfile(wavelength)
    tran_prof = GaussianTransverseProfile(spot_size)
    profile = CombinedLongitudinalTransverseProfile(
        wavelength, pol, long_prof, tran_prof, peak_fluence=peak_fluence
    )

    return profile


def test_3D_case(gaussian):
    dimensions = "xyt"  # Use Cartesian geometry
    lo = (
        -5.0 * spot_size,
        -5.0 * spot_size,
        None,
    )  # Lower bounds of the simulation box
    hi = (5.0 * spot_size, 5.0 * spot_size, None)  # Upper bounds of the simulation box
    num_points = (256, 256, 1)  # Number of points in each dimension

    laser = Laser(dimensions, lo, hi, num_points, laser_profile)

    # Add a phase aberration
    # CALCULATE THE REQUIRED PHASE ABERRATION
    x = np.linspace(lo[0], hi[0], npoints[0])
    y = np.linspace(lo[1], hi[1], npoints[1])
    X, Y = np.meshgrid(x, y)
    pupilRadius = 20e-6
    phase = -0.2 * zernike(X, Y, (0, 0, pupilRadius), 3)

    R = np.sqrt(X**2 + Y**2)
    phaseMask = np.ones_like(phase)
    phaseMask[R > pupilRadius] = 0

    # NOW ADD THE PHASE TO EACH SLICE OF THE FOCUS
    phase3D = np.repeat(phase[:, :, np.newaxis], npoints[2], axis=2)
    field = laser.grid.get_temporal_field()
    laser.grid.set_temporal_field(np.abs(field) * np.exp(1j * phase3D))

    # PROPAGATE THE FIELD FIELD FOWARDS AND BACKWARDS BY 1 MM
    field = [None] * 2
    propDist = 1e-3
    laserForward = copy.deepcopy(laser)
    laserForward.propagate(propDist)
    laserBackward = copy.deepcopy(laser)
    laserBackward.propagate(-propDist)

    field = [laserBackward, laserForward]
    zVals = [-propDist, propDist]

    # PERFORM GERCHBERG-SAXTON ALGORTIHM TO RETRIEVE PHASE
    gs = GerchbergSaxton(field, zVals, m_max=20, n_max=20, max_iter=50)
    chi2, chi2Grad = gs.retrieve_phase()
    assert chi2 < 5e-5
