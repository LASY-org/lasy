# -*- coding: utf-8 -*-
"""Test the fraunhofer far field implementation.

Test checks the implementation of the polynomial spectral phase
by initializing a Gaussian pulse (with flat spectral phase),
adding spectral phase to it, and checking the corresponding
temporal shape of the laser pulse again analytical formulas.
"""

import numpy as np
from scipy.constants import c

from lasy.laser import Laser


from lasy.profiles.combined_profile import CombinedLongitudinalTransverseProfile
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.profiles.longitudinal.gaussian_profile import GaussianLongitudinalProfile

from lasy.utils.grid import Grid, time_axis_indx
from lasy.utils.laser_utils import compute_laser_energy


# Laser parameters
wavelength = 0.8e-6
omega0 = 2 * np.pi * c / wavelength
beam_size = 25e-3   # m
pol = (1, 0)
laser_energy = 1.0  # J
t_peak = 0.0e-15  # s
tau = 50.0e-15  # s
gaussian_profile = GaussianProfile(wavelength, pol, laser_energy, beam_size, tau, t_peak)

# Grid parameters
s_multi = 4
t_multi = 5
num_r = 2**7
num_t = 2**8
dimensions = "xyt"    # Use cartesian geometry
lo = (-s_multi * beam_size,-s_multi * beam_size, -t_multi * tau) # Lower bounds of the simulation box
hi = (s_multi * beam_size,s_multi * beam_size, t_multi * tau)  # Upper bounds of the simulation box
num_points = (num_r,num_r, num_t)    # Number of points in each dimension



def test_farfield():
    """ Calculate far field and compare to anaytical

    Compare transverse electric field profile to analytical for gaussian profile.
    Also check that laser energy is conserved
    """
    
    # Initialize the laser
    laser = Laser(dimensions, lo, hi, num_points, gaussian_profile) # use lasy initialisation
    E0 = abs(laser.grid.get_temporal_field()).max()

    f = beam_size*10
    N_pad = 2**9
    unpad_result = True
    laser.go_to_farfield(f,N_pad,unpad_result) 
    E_xyt = laser.grid.get_temporal_field()
    E_x = abs(E_xyt[:,64,127])
   
    # Compare the on-axis field with the analytical formula
    sigma_u = np.sqrt(beam_size**2/2 -np.sqrt(beam_size**4/4-f**2*wavelength**2/np.pi**2))
    E_x_theory = E0*(beam_size/sigma_u)*np.exp(-laser.grid.axes[0]**2/(sigma_u**2))
    tol = 1.0e-3
    assert np.all(
        abs(E_x - E_x_theory) / abs(E_x_theory).max() < tol
    )

    assert abs(compute_laser_energy(laser.dim, laser.grid)-laser_energy)/laser_energy < tol

