import numpy as np

from lasy.laser import Laser
from lasy.profiles import GaussianProfile
from lasy.propagators import AngularSpectrumPropagator
from lasy.utils.laser_utils import field_to_vector_potential


def test_fftw():
    profile = GaussianProfile(
        wavelength=800e-9,
        laser_energy=13.92,
        tau=90e-15,
        w0=50.0e-6,
        pol=(0, 1),
        t_peak=0,
        z_foc=0.0314,
    )
    dim = "xyt"
    hi = (1e-3, 1.0e-3, 180e-15)
    lo = (-1e-3, -1.0e-3, -180e-15)
    npoints = (200, 500, 500)

    laser = Laser(dim=dim, hi=hi, lo=lo, npoints=npoints, profile=profile)
    laserp = Laser(dim=dim, hi=hi, lo=lo, npoints=npoints, profile=profile)
    linear_propagator = AngularSpectrumPropagator(
        omega0=profile.omega0, n=1, dim=laser.dim
    )
    dz = 0.02  # length of the individual propagation steps
    laserp.add_propagator(linear_propagator)
    laser.add_propagator(linear_propagator)
    laserp.propagate(dz, use_fftw=True)
    laser.propagate(dz, use_fftw=True)
    Ar = field_to_vector_potential(laser.grid, laser.profile.omega0)
    Arp = field_to_vector_potential(laserp.grid, laserp.profile.omega0)
    assert np.allclose(Ar, Arp, rtol=1e-6, atol=1e-9)
