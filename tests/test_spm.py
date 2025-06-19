import numpy as np
from scipy.constants import c, epsilon_0

from lasy.laser import Laser
from lasy.profiles import GaussianProfile
from lasy.propagators.nonlinear_phase_shift import NonlinearKerrStep
from lasy.utils.laser_utils import get_bandwidth


def make_laser():
    profile = GaussianProfile(
        wavelength=800e-9,
        pol=(1, 0),
        laser_energy=100e-3,
        tau=50e-15 / np.sqrt(2 * np.log(2)),
        w0=10e-3,
        t_peak=0,
    )

    dim = "xyt"
    hi = (20e-3, 20e-3, 250e-15)
    lo = (-20e-3, -20e-3, -250e-15)
    npoints = (51, 51, 101)

    laser = Laser(dim=dim, hi=hi, lo=lo, npoints=npoints, profile=profile)

    return laser


def test_nonlinear_step():
    """
    Compare the numerically calculated spectral broadening to the analytical equation for broadening of Gaussian pulses.

    The analytical description is taken from 'Nonlinear Fiber Optics, G. Agrawal, 3rd ed., p.104.'
    """
    # define refractive index and nonlinear refractive index
    n2 = 1e-20

    # create laser
    laser = make_laser()

    # create propagator
    NLprop = NonlinearKerrStep(n2=n2, k0=laser.profile.omega0 / c)

    # create range of distances over which to test the spectral broadening
    z_pos = np.linspace(0, 15e-3, 5)

    bandwidths_propagated = []

    # iterate over z-steps and calculate the on-axis bandwidth
    for z in z_pos:
        laser = make_laser()
        laser.grid = NLprop.apply(distance=z, grid_in=laser.grid)
        bandwidth = (
            get_bandwidth(grid=laser.grid, dim=laser.dim, method="on-axis") * 2
        )  # times 2 to convert half-width to full width
        bandwidths_propagated.append(bandwidth)

    # calculate peak intensity and max phase shift for analytical broadening calculation
    peak_intensity = np.max(
        0.5 * c * epsilon_0 * abs(laser.grid.get_temporal_field()) ** 2
    )
    phase_shift = 2 * np.pi / laser.profile.lambda0 * n2 * z_pos * peak_intensity

    initial_bandwidth = bandwidths_propagated[0]

    # calculate broadening according to Agrawal book
    bandwidths_analytical = initial_bandwidth * np.sqrt(
        1 + 4 / (3 * 3**0.5) * (phase_shift) ** 2
    )

    assert np.allclose(bandwidths_propagated, bandwidths_analytical, rtol=1e-3)
