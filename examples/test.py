from lasy.laser import Laser
from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.profiles.combined_profile import CombinedLongitudinalTransverseProfile
from lasy.profiles.transverse.laguerre_gaussian_profile import LaguerreGaussianTransverseProfile
from lasy.profiles.longitudinal.gaussian_profile import GaussianLongitudinalProfile

## Case with Gaussian laser

wavelength=.8e-6
pol = (1,0)
laser_energy = 1. # J
t_peak = 0.e-15 # s
tau = 30.e-15 # s
w0 = 5.e-6 # m
profile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)

# - 3D Cartesian case
dim = 'xyt'
lo = (-10e-6, -10e-6, -60e-15)
hi = (+10e-6, +10e-6, +60e-15)
npoints=(100,100,100)

laser = Laser(dim, lo, hi, npoints, profile)
laser.write_to_file('gaussianlaser3d')
laser.propagate(1)
laser.write_to_file('gaussianlaser3d')

# - Cylindrical case
dim = 'rt'
lo = (0e-6, -60e-15)
hi = (10e-6, +60e-15)
npoints=(50,100)

laser = Laser(dim, lo, hi, npoints, profile)
laser.write_to_file('gaussianlaserRZ')
laser.propagate(1)
laser.write_to_file('gaussianlaserRZ')

# Case with Laguerre-Gauss laser

wavelength=.8e-6
pol = (1,0)
laser_energy = 1. # J
t_peak = 0.e-15 # s
tau = 30.e-15 # s
w0 = 5.e-6 # m
profile = CombinedLongitudinalTransverseProfile( wavelength, pol, laser_energy,
            GaussianLongitudinalProfile( wavelength, tau, t_peak ),
            LaguerreGaussianTransverseProfile( w0, p=0, m=1 ) )

# - Cylindrical case
dim = 'rt'
lo = (0e-6, -60e-15)
hi = (10e-6, +60e-15)
npoints=(50,100)

laser = Laser(dim, lo, hi, npoints, profile, n_azimuthal_modes=2)
laser.write_to_file('laguerrelaserRZ')
laser.propagate(1)
laser.write_to_file('laguerrelaserRZ')
