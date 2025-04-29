import numpy as np

from lasy.laser import Laser
from lasy.profiles import GaussianProfile
from lasy.propagators import AngularSpectrumDFFTPropagator
<<<<<<< HEAD
import numpy as np
from scipy.constants import c
=======
from lasy.utils.laser_utils import get_duration, get_w0
>>>>>>> 6047ebc78956349d6fd87041bb3c4649f789fd8f


def make_laser():
    profile = GaussianProfile(
        wavelength=800e-9,
        pol=(1, 0),
        laser_energy=1,
        tau=30e-15 / np.sqrt(2 * np.log(2)),
        w0=100e-6,
        t_peak=0,
    )

<<<<<<< HEAD
    dim = 'xyt'
    hi = (2e-3, 2e-3, 500e-15)
    lo = (-2e-3, -2e-3, -500e-15)
    npoints = (100, 100, 100)
=======
    dim = "xyt"
    hi = (5e-3, 5e-3, 300e-15)
    lo = (-5e-3, -5e-3, -300e-15)
    npoints = (100, 100, 201)
>>>>>>> 6047ebc78956349d6fd87041bb3c4649f789fd8f

    laser = Laser(dim=dim, hi=hi, lo=lo, npoints=npoints, profile=profile)

    return laser


def test_spatial_propagation():
<<<<<<< HEAD
    prop = AngularSpectrumDFFTPropagator(omega0=2*np.pi*c/800e-9, n=1.)
=======
    prop = AngularSpectrumDFFTPropagator(n=1.0)
>>>>>>> 6047ebc78956349d6fd87041bb3c4649f789fd8f

    z_pos = np.linspace(-5e-3, 50e-3, 10)
    waists_propagated = []

    for z in z_pos:
        laser = make_laser()
        prop.propagate(distance=z, grid=laser.grid, dim=laser.dim)
        waist = get_w0(grid=laser.grid, dim=laser.dim)
        waists_propagated.append(waist)

<<<<<<< HEAD
    zR = np.pi*laser.profile.w0**2/(laser.profile.lambda0)
    waists_analytical = laser.profile.w0 * np.sqrt(1 + (z_pos/zR)**2)
=======
    zR = np.pi * laser.profile.w0**2 / laser.profile.laser_wavelength

    waists_analytical = laser.profile.w0 * np.sqrt(1 + (z_pos / zR) ** 2)
>>>>>>> 6047ebc78956349d6fd87041bb3c4649f789fd8f

    assert np.allclose(waists_propagated, waists_analytical, rtol=1e-5, atol=1e-6)


def n_fusedsilica(wavelength):
    """
    Sellmeier equation for fused silica
    """
<<<<<<< HEAD
    x = wavelength * 1e6
    return (1+0.6961663/(1-(0.0684043/x)**2)+0.4079426/(1-(0.1162414/x)**2)+0.8974794/(1-(9.896161/x)**2))**.5


def test_temporal_propagation():

    prop = AngularSpectrumDFFTPropagator(omega0=2*np.pi*c/800e-9, n=n_fusedsilica)
=======
    return (
        1
        + 0.6961663 / (1 - (0.0684043 / (wavelength * 1e-6)) ** 2)
        + 0.4079426 / (1 - (0.1162414 / (wavelength * 1e-6)) ** 2)
        + 0.8974794 / (1 - (9.896161 / (wavelength * 1e-6)) ** 2)
    ) ** 0.5


def test_temporal_propagation():
    prop = AngularSpectrumDFFTPropagator(n=n_fusedsilica)
>>>>>>> 6047ebc78956349d6fd87041bb3c4649f789fd8f

    z_pos = np.linspace(0, 10e-3, 10)
    durations_propagated = []

    for z in z_pos:
        laser = make_laser()
        prop.propagate(distance=z, grid=laser.grid, dim=laser.dim)
        duration = get_duration(grid=laser.grid, dim=laser.dim) * 2*np.sqrt(2*np.log(2))
        durations_propagated.append(duration)

    gdd = z_pos * 36.163e-27
    tau_initial = laser.profile.tau * np.sqrt(2 * np.log(2))
    duration_analytical = tau_initial * np.sqrt(
        1 + (4 * np.log(2) * gdd / tau_initial**2) ** 2
    )

    assert np.allclose(durations_propagated, duration_analytical, rtol=1e-5, atol=1e-15)
