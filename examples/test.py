import numpy as np
from lasy.laser import Laser
from lasy.laser_profiles.gaussian_laser import GaussianLaser

wavelength=.8e-6
pol = (1,0)
laser_energy = 1. # J
t_peak = 0.e-15 # s
tau = 30.e-15 # s
w0 = 5.e-6 # m
profile = GaussianLaser(wavelength, pol, laser_energy, w0, tau, t_peak)

# 3D Cartesian case

dim = 'xyt'
xlim = (-10e-6, 10e-6)
ylim = (-10e-6, 10e-6)
tlim = (-60e-15, 60e-15)
npoints=(100,100,100)

laser = Laser(dim, npoints, profile, tlim, xlim=xlim, ylim=ylim)
laser.write_to_file('laser3d')
laser.propagate(1)
laser.write_to_file('laser3d')

# Cylindrical case

dim = 'rt'
rlim = (0, 10e-6)
tlim = (-60e-15, 60e-15)
npoints=(50,100)

laser = Laser(dim, npoints, profile, tlim=tlim, rlim=rlim)
laser.write_to_file('laserRZ')
laser.propagate(1)
laser.write_to_file('laserRZ')
