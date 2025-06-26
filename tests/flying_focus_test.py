import numpy as np
from scipy.constants import c
from lasy.profiles.flying_focus_profiles import(
    FlyingFocusGaussianProfile,
    FlyingFocusHGProfile,
    FlyingFocusLGProfile,
    FlyingFocusGaussianProfile2D,
    FlyingFocusHGProfile2D
)
from lasy.profiles.gaussian_profile import GaussianProfile
from scipy.special import (hermite, genlaguerre)
from math import factorial

'''
Initializing Constants
'''

# important constants
wavelength = 10e-9 # m
pol = (1, 0) # polarization
w_0 = 10e-2 # waist in meters
w_0x = w_0 # x waist
w_0y = 10 * w_0x # y waist
m = 1 # hermite x modes
n = 2 # hermite y modes
p = 2 # laguerre radial modes
l = 3 # laguerre azimuthal modes
energy = 15 # laser energy in joules
tau = 30e-15 # pulse dir in seconds
t_peak = 2 * tau # peak intensity time in seconds
vf = 0 # focus velocity in meters per second
cep_phase = np.pi / 3 # initial phase in radians
z_r = np.pi * w_0**2 / wavelength # rayleigh length
z_rx = z_r # x rayleigh length
z_ry = np.pi * w_0y**2 / wavelength # y rayleigh length
z_foc = 2 * z_r # position of the initial focal plane in meters
n_order = 2 # supergaussian order, must be int

# computed constants
k = 2 * np.pi / wavelength # wavenumber
omega0 = k * c # angular velocity

# for laser rendering
dimensions = "rt"  # Use cylindrical geometry
lo = (0, -2.5 * tau)  # Lower bounds of the simulation box (r, t)
hi = (5 * w_0, 2.5 * tau)  # Upper bounds of the simulation box (r, t)
num_points = (200, 200)  # Number of points in each dimension (r, t)

# spatio-temporal grid
t = np.linspace(lo[1], hi[1], num_points[1])
x = np.linspace(lo[0], hi[0], num_points[0])
y = np.linspace(10 * lo[0], 10 * hi[0], num_points[0])
X, Y, T = np.meshgrid(x, y, t, indexing='ij')
X, T = np.meshgrid(x, t, indexing='ij')

'''
Defining Analytical Test Solutions
'''

# Gaussian test function
def Gaussian(x, y, t):
    r = np.sqrt(x**2 + y**2)
    z = z_foc - vf * (t - t_peak) # make the focal plane time dependent for flying focus
    q = z + 1.0j * z_r # complex factor
    # Calculate the argument of the complex exponential
    exp_argument = - 1.0j * k * r**2 / (2 * q)
    # Get the profile
    envelope = (
        np.exp(exp_argument) * # transverse profile term
        np.exp(-np.power(((t - t_peak) ** 2) / tau**2, n_order / 2)) * # pulse spread term
        np.exp(1.0j * (cep_phase + omega0 * t_peak)) # phase factor
        * z_r * 1.0j / q # normalization, the 1.0j accounts for a phase built into lasy
    )

    return envelope

# Analytical HG solution
def HermiteGaussian(x, y, t):
    z = z_foc - vf * (t - t_peak) # time dependnet focal plane for flying focus
    q_x = z + 1.0j * z_rx # complex factor for x
    q_y = z + 1.0j * z_ry # complex factor for y
    gouy_x = (m + 1/2) * np.arctan2(z, z_rx) # gouy phase for x
    gouy_y = (n + 1/2) * np.arctan2(z, z_ry) # gouy phase for y
    waist_x = w_0x * np.sqrt(1 + (z / z_rx)**2) # waist on x axis
    waist_y = w_0y * np.sqrt(1 + (z / z_ry)**2) # waist on y axis

    hermite_norm_x = np.sqrt(np.sqrt(2 / np.pi) / (2**m * factorial(m) * waist_x)) # coef from x
    hermite_norm_y = np.sqrt(np.sqrt(2 / np.pi) / (2**n * factorial(n) * waist_y)) # coef from y

    h_x = ( # x solution for HG
        hermite_norm_x * 
        hermite(m)(np.sqrt(2) * x / waist_x) * 
        np.exp(-1.0j * k * x**2 / (2 * q_x)) * 
        np.exp(1.0j * gouy_x)
    ) # y solution for HG
    h_y = (
        hermite_norm_y * 
        hermite(n)(np.sqrt(2) * y / waist_y) * 
        np.exp(-1.0j * k * y**2 / (2 * q_y)) * 
        np.exp(1.0j * gouy_y)
    )

    # Get the profile
    envelope = (
        h_x * h_y * # HG transverse profile
        np.exp(-np.power(((t - t_peak) ** 2 / tau**2), n_order / 2)) * # pulse spread term
        np.exp(1.0j * (cep_phase + omega0 * t_peak)) # phase factor
    )

    return envelope

# Analyic LG Solution
def LaguerreGaussian(x, y, t):
    z = z_foc - vf * (t - t_peak) # time dependent focal plane
    r = np.sqrt(x**2 + y**2) # cylindrical coordinate conversion
    phi = np.arctan2(y, x)
    q = z + 1.0j * z_r # complex factor
    gouy = (2 * p + np.abs(l) + 1) * np.arctan2(z, z_r) # gouy phase
    waist = w_0 * np.sqrt(1 + (z / z_r)**2) # waist calculation

    laguerre_norm = np.sqrt(2 * factorial(p) / (np.pi * factorial(p + np.abs(l)))) / waist # should normalize

    lg = ( # LG solution transverse
        laguerre_norm * 
        (r * np.sqrt(2) / waist)**np.abs(l) * 
        genlaguerre(p, np.abs(l))(2 * r**2 / waist**2) * 
        np.exp(1.0j * gouy) * 
        np.exp(-1.0j * k * r**2 / (2 * q)) * 
        np.exp(-1.0j * l * phi)
    )

    # Get the profile
    envelope = (
        lg * # LG transverse profile
        np.exp(-np.power(((t - t_peak) ** 2 / tau**2), n_order / 2)) * # pulse spread term
        np.exp(1.0j * (cep_phase + omega0 * t_peak)) # phase factor
    )
   
    return envelope

# 2D Gaussian test function
def Gaussian2D(x, t):
    z = z_foc - vf * (t - t_peak) # make the focal plane time dependent for flying focus
    q = z + 1.0j * z_r # complex factor
    # Calculate the argument of the complex exponential
    exp_argument = - 1.0j * k * x**2 / (2 * q)
    # Get the profile
    envelope = (
        np.exp(exp_argument) * # transverse profile term
        np.exp(-np.power(((t - t_peak) ** 2) / tau**2, n_order / 2)) * # pulse spread term
        np.exp(1.0j * (cep_phase + omega0 * t_peak)) # phase factor
        * np.sqrt(1.0j * z_r) / np.sqrt(q) # normalization
    )

    return envelope

# 2d HG test function
def HermiteGaussian2D(x, t):
    z = z_foc - vf * (t - t_peak) # time dependnet focal plane for flying focus
    q_x = z + 1.0j * z_r # complex factor for x
    gouy_x = (m + 1/2) * np.arctan2(z, z_r) # gouy phase for x
    waist_x = w_0 * np.sqrt(1 + (z / z_r)**2) # waist on x axis

    hermite_norm_x = np.sqrt(np.sqrt(2 / np.pi) / (2**m * factorial(m) * waist_x)) # coef from x

    h_x = ( # x solution for HG
        hermite_norm_x * 
        hermite(m)(np.sqrt(2) * x / waist_x) * 
        np.exp(-1.0j * k * x**2 / (2 * q_x)) * 
        np.exp(1.0j * gouy_x)
    )

    # Get the profile
    envelope = (
        h_x *  # HG transverse profile
        np.exp(-np.power(((t - t_peak) ** 2 / tau**2), n_order / 2)) * # pulse spread term
        np.exp(1.0j * (cep_phase + omega0 * t_peak)) # phase factor
    )

    return envelope

'''
Calculating Errors
'''

# Error of FF Gauss against Lasy Gauss
# constants of importance

# initialize gaussian profile
laser_profile_gauss = GaussianProfile(
    wavelength,
    pol,
    energy,
    w_0,
    tau,
    t_peak,
    0,
    0,
    cep_phase,
    z_foc,
)
# initalize flying focus gaussian profile
laser_profile_ff_gauss = FlyingFocusGaussianProfile(

    w_0,
    wavelength,
    pol,
    energy, 
    tau,
    t_peak,
    vf,
    cep_phase,
    z_foc,
    n_order
)

# Norming lasy Gauss
E_lasy_gauss = laser_profile_gauss.evaluate(X, Y, T) 
E_lasy_gauss = E_lasy_gauss / np.max(np.abs(E_lasy_gauss))
# norming FF Gauss
E_lasy_ff_gauss = laser_profile_ff_gauss.evaluate(X, Y, T) 
E_lasy_ff_gauss = E_lasy_ff_gauss / np.max(np.abs(E_lasy_ff_gauss))

vf = 0
n_order = 2 # necessary for consistensy with Lasy Gauss code
rel_error_real = np.max(np.abs(np.real(E_lasy_gauss) - np.real(E_lasy_ff_gauss)))
rel_error_imag = np.max(np.abs(np.imag(E_lasy_gauss) - np.imag(E_lasy_ff_gauss)))
error = 100 * (rel_error_real + rel_error_imag) # error in percent which sums the real and imaginary errors

# error bound
assert error < 1.0e-6

''''''

# Error of FF Gauss against Analytic Gauss with vf != 0
# constants of importance
vf = 0.5 * c
n_order = 5

# initalize flying focus gaussian profile
laser_profile_ff_gauss = FlyingFocusGaussianProfile(
    w_0,
    wavelength,
    pol,
    energy, 
    tau,
    t_peak,
    vf,
    cep_phase,
    z_foc,
    n_order
)
# Norming FF Gauss
E_lasy_ff_gauss = laser_profile_ff_gauss.evaluate(X, Y, T) 
E_lasy_ff_gauss = E_lasy_ff_gauss / np.max(np.abs(E_lasy_ff_gauss))
# analytic redfinition
E_analytical_gauss = Gaussian(X, Y, T) 
E_analytical_gauss = E_analytical_gauss / np.max(np.abs(E_analytical_gauss)) # normed analytical solution for Gauss

# error calculations
rel_error_real = np.max(np.abs(np.real(E_analytical_gauss) - np.real(E_lasy_ff_gauss)))
rel_error_imag = np.max(np.abs(np.imag(E_analytical_gauss) - np.imag(E_lasy_ff_gauss)))
error = 100 * (rel_error_real + rel_error_imag) # error in percent which sums the real and imaginary errors

# error bound
assert error < 1.0e-6

''''''

# Error of FF HG against Gauss
#constants of importance
w_0y = w_0x
m = 0
n = 0
vf = 0
n_order = 2

# initialize flying focus hg profile
laser_profile_ff_hg = FlyingFocusHGProfile(
    w_0x,
    w_0y,
    m,
    n, 
    wavelength,
    pol,
    energy,
    tau,
    t_peak,
    vf,
    cep_phase,
    z_foc,
    n_order
)
# Norming FF HG
E_lasy_ff_hg = laser_profile_ff_hg.evaluate(X, Y, T) 
E_lasy_ff_hg = E_lasy_ff_hg / np.max(np.abs(E_lasy_ff_hg))

# error calculations
rel_error_real = np.max(np.abs(np.real(E_lasy_gauss) - np.real(E_lasy_ff_hg)))
rel_error_imag = np.max(np.abs(np.imag(E_lasy_gauss) - np.imag(E_lasy_ff_hg)))
error = 100 * (rel_error_real + rel_error_imag) # error in percent which sums the real and imaginary errors

# error bound
assert error < 1.0e-6

''''''

# Error of FF HG against Analytic HG
w_0y = 10 * w_0x
m = 3
n = 2
vf = 0.5 * c
n_order = 5

# initialize flying focus hg profile
laser_profile_ff_hg = FlyingFocusHGProfile(
    w_0x,
    w_0y,
    m,
    n, 
    wavelength,
    pol,
    energy,
    tau,
    t_peak,
    vf,
    cep_phase,
    z_foc,
    n_order
)
# Norming FF HG
E_lasy_ff_hg = laser_profile_ff_hg.evaluate(X, Y, T) 
E_lasy_ff_hg = E_lasy_ff_hg / np.max(np.abs(E_lasy_ff_hg))
# analytic redfinition
E_analytical_hg = HermiteGaussian(X, Y, T) 
E_analytical_hg = E_analytical_hg / np.max(np.abs(E_analytical_hg)) # normed analytical solution for HG

# error calculations
rel_error_real = np.max(np.abs(np.real(E_analytical_hg) - np.real(E_lasy_ff_hg)))
rel_error_imag = np.max(np.abs(np.imag(E_analytical_hg) - np.imag(E_lasy_ff_hg)))
error = 100 * (rel_error_real + rel_error_imag) # error in percent which sums the real and imaginary errors

# error bound
assert error < 1.0e-6

''''''

# Error of FF LG against Lasy Gauss
p = 0
l = 0
vf = 0
n_order = 2

# initialize flying focus lg profile
laser_profile_ff_lg = FlyingFocusLGProfile(
    w_0,
    p,
    l,
    wavelength,
    pol,
    energy,
    tau,
    t_peak,
    vf,
    cep_phase,
    z_foc,
    n_order
)
# Norming FF LG
E_lasy_ff_lg = laser_profile_ff_lg.evaluate(X, Y, T) 
E_lasy_ff_lg = E_lasy_ff_lg / np.max(np.abs(E_lasy_ff_lg))

# error calculations
rel_error_real = np.max(np.abs(np.real(E_lasy_gauss) - np.real(E_lasy_ff_lg)))
rel_error_imag = np.max(np.abs(np.imag(E_lasy_gauss) - np.imag(E_lasy_ff_lg)))
error = 100 * (rel_error_real + rel_error_imag) # error in percent which sums the real and imaginary errors

# error bound
assert error < 1.0e-6

''''''

# Error of FF LG against Analytic LG
p = 3
l = 2
vf = 0.5 * c
n_order = 5

# initialize flying focus lg profile
laser_profile_ff_lg = FlyingFocusLGProfile(
    w_0,
    p,
    l,
    wavelength,
    pol,
    energy,
    tau,
    t_peak,
    vf,
    cep_phase,
    z_foc,
    n_order
)
# Norming FF LG
E_lasy_ff_lg = laser_profile_ff_lg.evaluate(X, Y, T) 
E_lasy_ff_lg = E_lasy_ff_lg / np.max(np.abs(E_lasy_ff_lg))
# analytic redfinition
E_analytical_lg = LaguerreGaussian(X, Y, T) 
E_analytical_lg = E_analytical_lg / np.max(np.abs(E_analytical_lg)) # normed analytical solution for LG

# error calculations
rel_error_real = np.max(np.abs(np.real(E_analytical_lg) - np.real(E_lasy_ff_lg)))
rel_error_imag = np.max(np.abs(np.imag(E_analytical_lg) - np.imag(E_lasy_ff_lg)))
error = 100 * (rel_error_real + rel_error_imag) # error in percent which sums the real and imaginary errors

# error bound
assert error < 1.0e-6
''''''

# Error of 2d FF Gauss Against 2d Gauss Analytic
# constants of importance
vf = 0.5 * c
n_order = 5

# initialize 2d FF Gauss test function
E_analytical_ff_gauss_2d = Gaussian2D(X, T)
E_analytical_ff_gauss_2d = E_analytical_ff_gauss_2d / np.max(np.abs(E_analytical_ff_gauss_2d))

# initialize 2d Gauss FF implementation
laser_profile_ff_gauss_2d = FlyingFocusGaussianProfile2D(
    w_0,
    wavelength,
    pol,
    energy,
    tau,
    t_peak,
    vf,
    cep_phase,
    z_foc,
    n_order
)
E_lasy_ff_gauss_2d = laser_profile_ff_gauss_2d.evaluate(X, T)
E_lasy_ff_gauss_2d = E_lasy_ff_gauss_2d / np.max(np.abs(E_lasy_ff_gauss_2d))
# error calculations
rel_error_real = np.max(np.abs(np.real(E_analytical_ff_gauss_2d) - np.real(E_lasy_ff_gauss_2d)))
rel_error_imag = np.max(np.abs(np.imag(E_analytical_ff_gauss_2d) - np.imag(E_lasy_ff_gauss_2d)))
error = 100 * (rel_error_real + rel_error_imag) # error in percent which sums the real and imaginary errors

# error bound
assert error < 1.0e-6

''''''

# Error of 2d FF HG against 2d FF Gauss
# constants of importance
m = 0
vf = 0.5 * c
n_order = 5

# initialize 2d Gauss FF implementation
laser_profile_ff_hg_2d = FlyingFocusHGProfile2D(
    w_0,
    m,
    wavelength,
    pol,
    energy,
    tau,
    t_peak,
    vf,
    cep_phase,
    z_foc,
    n_order
)
E_lasy_ff_hg_2d = laser_profile_ff_hg_2d.evaluate(X, T)
E_lasy_ff_hg_2d = E_lasy_ff_hg_2d / np.max(np.abs(E_lasy_ff_hg_2d))

# error calculations
rel_error_real = np.max(np.abs(np.real(E_lasy_ff_gauss_2d) - np.real(E_lasy_ff_hg_2d)))
rel_error_imag = np.max(np.abs(np.imag(E_lasy_ff_gauss_2d) - np.imag(E_lasy_ff_hg_2d)))
error = 100 * (rel_error_real + rel_error_imag) # error in percent which sums the real and imaginary errors

# error bound
assert error < 1.0e-6

''''''

# Error of 2d FF HG against 2d FF HG analytic
# constants of importance
m = 4
vf = 0.5 * c
n_order = 5

# initialize 2d FF Gauss test function
E_analytical_ff_hg_2d = HermiteGaussian2D(X, T)
E_analytical_ff_hg_2d = E_analytical_ff_hg_2d / np.max(np.abs(E_analytical_ff_hg_2d))

# initialize 2d Gauss FF implementation
laser_profile_ff_hg_2d = FlyingFocusHGProfile2D(
    w_0,
    m,
    wavelength,
    pol,
    energy,
    tau,
    t_peak,
    vf,
    cep_phase,
    z_foc,
    n_order
)
E_lasy_ff_hg_2d = laser_profile_ff_hg_2d.evaluate(X, T)
E_lasy_ff_hg_2d = E_lasy_ff_hg_2d / np.max(np.abs(E_lasy_ff_hg_2d))

# error calculations
rel_error_real = np.max(np.abs(np.real(E_analytical_ff_hg_2d) - np.real(E_lasy_ff_hg_2d)))
rel_error_imag = np.max(np.abs(np.imag(E_analytical_ff_hg_2d) - np.imag(E_lasy_ff_hg_2d)))
error = 100 * (rel_error_real + rel_error_imag) # error in percent which sums the real and imaginary errors

# error bound
assert error < 1.0e-6