# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pytest
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c
import numpy as np

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

#@pytest.fixture(scope="function")
def gaussian():
    # Cases with Gaussian laser
    profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)
    
    return profile

def check_value(real_value, target):
    #returns the relative error between real_value and target
    return abs(real_value-target)/target

def weighted_std(data, weights):
    """calculate the weighted std of an array"""
    average = np.average(data, weights=weights)
    variance = np.average((data - average) ** 2, weights=weights)
    return( np.sqrt(variance) )


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

    r_tol = 1e-1

    # load in the data with openPMD-viewer
    ts_3d = OpenPMDTimeSeries('./testdata')

    # check number of dimensions
    Ex_3d_real, _ = ts_3d.get_field(field='E_real', coord='x', iteration=0,
                                    slice_across=None)
    assert Ex_3d_real.ndim == 3

    # check for nans
    assert np.all(Ex_3d_real) != np.nan

    # calculate waist and pulse duration
    Ex_x_real, info_x = ts_3d.get_field(field='E_real', coord='x', iteration=0,
                                    slice_across=['y','t'])
    waist_x = np.sqrt(2) * weighted_std(info_x.x, Ex_x_real)
    print(f'Waist x : {waist_x}')

    Ex_y_real, info_y = ts_3d.get_field(field='E_real', coord='x', iteration=0,
                                    slice_across=['x','t'])
    waist_y = np.sqrt(2) * weighted_std(info_y.y, Ex_y_real)
    print(f'Waist y : {waist_y}')

    assert check_value(waist_x, w0) < r_tol
    assert check_value(waist_y, w0) < r_tol

    # check the pulse duration
    Ex_t_real, info_t = ts_3d.get_field(field='E_real', coord='x', iteration=0,
                                    slice_across=['x','y'])
    tau_retrieved = np.sqrt(2) * weighted_std(info_t.t, Ex_t_real)
    print(f'Pulse duration : {tau_retrieved}')

    assert check_value(tau_retrieved, tau) < r_tol


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

if __name__ == '__main__':
    test_profile_gaussian_3d_cartesian(gaussian())