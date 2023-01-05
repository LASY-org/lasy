# -*- coding: utf-8 -*-

import pytest
from openpmd_viewer.addons import LpaDiagnostics
from scipy.constants import c

from lasy.laser import Laser
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.profiles.combined_profile import CombinedLongitudinalTransverseProfile
from lasy.profiles.transverse.laguerre_gaussian_profile import LaguerreGaussianTransverseProfile
from lasy.profiles.longitudinal.gaussian_profile import GaussianLongitudinalProfile

wavelength=.8e-6
pol = (1,0)
laser_energy = 1. # J
t_peak = 0.e-15 # s
tau = 30.e-15 # s
w0 = 5.e-6 # m

@pytest.fixture(scope="function")
def gaussian():
    # Cases with Gaussian laser
    profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)
    
    return profile

@pytest.fixture(scope="function")
def check_value(real_value, target):
    #returns the relative error between real_value and target
    return abs(real_value-target)/target


def test_profile_gaussian_3d_cartesian(gaussian):
    # - 3D Cartesian case
    dim = 'xyt'
    lo = (-10e-6, -10e-6, -60e-15)
    hi = (+10e-6, +10e-6, +60e-15)
    npoints=(100,100,100)

    laser = Laser(dim, lo, hi, npoints, gaussian)
    laser.write_to_file('testdata/gaussianlaser3d')
    laser.propagate(1)
    laser.write_to_file('testdata/gaussianlaser3d')

    r_tol = 1e-5

    # load in the data with openPMD-viewer
    ts_3d = LpaDiagnostics('./testdata')

    # check number of dimensions
    Ex_3d_real1, _ = ts_3d.get_field(field='E_real', coord='x', iteration=0,
                                                    slice_across=None)
    assert Ex_3d_real1.ndim == 3

    # check for nans
    assert np.all(Ex_3d_real1) != np.nan

    # calculate waist and pulse duration
    waist_x = ts_3d.get_laser_waist(iteration=0, pol='x')
    waist_y = ts_3d.get_laser_waist(iteration=0, pol='y')

    assert check_value(waist_x, w0) < r_tol
    assert check_value(waist_y, w0) < r_tol

    # check the pulse duration

    tau_retrieved = ts_3d.get_ctau(iteration=0, pol='x')

    assert check_value(tau_retrieved/c, tau) < r_tol


def test_profile_gaussian_cylindrical(gaussian):
    # - Cylindrical case
    dim = 'rt'
    lo = (0e-6, -60e-15)
    hi = (10e-6, +60e-15)
    npoints=(50,100)

    laser = Laser(dim, lo, hi, npoints, gaussian)
    laser.write_to_file('testdata/gaussianlaserRZ')
    laser.propagate(1)
    laser.write_to_file('testdata/gaussianlaserRZ')


def test_profile_laguerre_gauss():
    # Case with Laguerre-Gauss laser
    profile = CombinedLongitudinalTransverseProfile( wavelength, pol, laser_energy,
                GaussianLongitudinalProfile( wavelength, tau, t_peak ),
                LaguerreGaussianTransverseProfile( w0, p=0, m=1 ) )

    # - Cylindrical case
    dim = 'rt'
    lo = (0e-6, -60e-15)
    hi = (10e-6, +60e-15)
    npoints=(50,100)

    laser = Laser(dim, lo, hi, npoints, profile, n_azimuthal_modes=2)
    laser.write_to_file('testdata/laguerrelaserRZ')
    laser.propagate(1)
    laser.write_to_file('testdata/laguerrelaserRZ')
