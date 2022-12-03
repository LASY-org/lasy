import numpy as np
from lasy.utils.box import Box
from lasy.gaussian_laser import GaussianLaser

wavelength=.8e-6
pol = (1,0)
laser_energy = 1. # J
t_peak = 0.e-15 # s
tau = 30.e-15 # s
w0 = 5.e-6 # m

# 3D Cartesian case

dim = 'xyt'
lo = (-10e-6, -10e-6, -60e-15)
hi = (+10e-6, +10e-6, +60e-15)
npoints=(100,100,100)
box = Box( dim, lo, hi, npoints)

laser = GaussianLaser(box, wavelength, pol, laser_energy, w0, tau, t_peak)
laser.write_to_file('laser3d')
laser.propagate(1)
laser.write_to_file('laser3d')

# Cylindrical case

dim = 'rt'
lo = (0e-6, -60e-15)
hi = (10e-6, +60e-15)
npoints=(50,100)
box = Box( dim, lo, hi, npoints)

laser = GaussianLaser(box, wavelength, pol, laser_energy, w0, tau, t_peak)
laser.write_to_file('laserRZ')
laser.propagate(1)
laser.write_to_file('laserRZ')
