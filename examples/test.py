import numpy as np
from lasy.utils.box import Box
from lasy.gaussian_laser import GaussianLaser

dim = 'xyt'
lo = (-10e-6, -10e-6, -60e-15)
hi = (+10e-6, +10e-6, +60e-15)
npoints=(20,20,20)
box = Box( dim, lo, hi, npoints)

wavelength=.8e-6
pol = (1,0)
laser_energy = 1. # J
t_peak = 0.e-15 # s
tau = 30.e-15 # s
w0 = 5.e-6 # m

laser = GaussianLaser(box, wavelength, pol, laser_energy, w0, tau, t_peak)
laser.write_to_file()
laser.propagate(1)
laser.write_to_file()
