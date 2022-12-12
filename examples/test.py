import numpy as np
import matplotlib.pyplot as plt
from openpmd_viewer import OpenPMDTimeSeries
from lasy.laser import Laser
from lasy.profiles.gaussian_profile import GaussianProfile

wavelength=.8e-6
pol = (1,0)
laser_energy = 1. # J
t_peak = 0.e-15 # s
tau = 30.e-15 # s
w0 = 5.e-6 # m
profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)

# 3D Cartesian case

dim = 'xyt'
lo = (-10e-6, -10e-6, -60e-15)
hi = (+10e-6, +10e-6, +60e-15)
npoints=(100,100,100)

laser = Laser(dim, lo, hi, npoints, profile)
laser.write_to_file('laser3d')
laser.propagate(1)
laser.write_to_file('laser3d')

# Cylindrical case

dim = 'rt'
lo = (0e-6, -60e-15)
hi = (10e-6, +60e-15)
npoints=(50,100)

laser = Laser(dim, lo, hi, npoints, profile)
laser.write_to_file('laserRZ')
laser.propagate(1)
laser.write_to_file('laserRZ')


# load in the data with openPMD-viewer
ts_3d = OpenPMDTimeSeries('../examples/')
ts_circ = OpenPMDTimeSeries('../examples/')

# plot some data
# #3d-geometry
# Slice across y (i.e. in a plane parallel to x-z)
plt.figure()
Ez_real1, info_Ez_real1 = ts_3d.get_field(field='E_real', coord='z', iteration=0,
                                          slice_across='t', plot=True)
plt.savefig('3dfield_Ez_real.png')

# Slice across t (i.e. in a plane parallel to x-z)
plt.figure()
Ey_imag1, info_Ey_imag1 = ts_3d.get_field(field='E_imag', coord='y', iteration=0,
                                          slice_across='x', plot=True)
plt.savefig('3dfield_Ey_imag.png')

# Get the full 3D Cartesian array
Ex_3d_real1, info_Ex_3d_real1 = ts_3d.get_field(field='E_real', coord='x', iteration=0,
                                                slice_across=None)
print(Ex_3d_real1.ndim)

# #RZ-geometry
# Slice across t
plt.figure()
Ez_real2, info_Ez_real2 = ts_circ.get_field(field='E_real', coord='z', iteration=0,
                                            m=0, slice_across='t', plot=True)
plt.savefig('RZfield_Ez_real.png')

# slice across x
plt.figure()
Ey_imag2, info_Ey_imag2 = ts_circ.get_field(field='E_imag', coord='y', iteration=0,
                                            m='all', slice_across='x', plot=True)
plt.savefig('RZfield_Ey_imag.png')

# Get the full 3D radial array
Ex_3d_real2, info_Ex3d_real2 = ts_circ.get_field(field='E_real', coord='x',
                                                 iteration=0)
print(Ex_3d_real2.ndim)
