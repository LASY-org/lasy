"""Test the implementation of the axiparabola.

Test checks the implementation of the axiparabola
by initializing a super-Gaussian pulse in the near field, and
propagating it to the middle of the focal range. It then
checks that the field amplitude remains high over the focal range.
"""

import numpy as np

from lasy.laser import Laser
from lasy.optical_elements import Axiparabola
from lasy.profiles.combined_profile import CombinedLongitudinalTransverseProfile
from lasy.profiles.longitudinal import GaussianLongitudinalProfile
from lasy.profiles.transverse import SuperGaussianTransverseProfile


def test_axiparabola():
    # Define the laser profile
    wavelength = 800e-9  # Laser wavelength in meters
    polarization = (1, 0)  # Linearly polarized in the x direction
    energy = 1.5  # Energy of the laser pulse in joules
    spot_size = 1e-3  # Spot size in the near-field: millimeter-scale
    pulse_duration = 30e-15  # Pulse duration of the laser in seconds
    t_peak = 0.0  # Location of the peak of the laser pulse in time
    laser_profile = CombinedLongitudinalTransverseProfile(
        wavelength,
        polarization,
        GaussianLongitudinalProfile(wavelength, pulse_duration, t_peak),
        SuperGaussianTransverseProfile(spot_size, n_order=16),
        laser_energy=energy,
    )

    # Define the laser on a grid
    dimensions = "rt"  # Use cylindrical geometry
    lo = (0, -2.5 * pulse_duration)  # Lower bounds of the simulation box
    hi = (1.1 * spot_size, 2.5 * pulse_duration)  # Upper bounds of the simulation box
    num_points = (3000, 30)  # Number of points in each dimension
    laser = Laser(dimensions, lo, hi, num_points, laser_profile)

    # Define the parameters of the axiparabola
    f0 = 3e-2  # Focal distance
    delta = 1.5e-2  # Focal range
    R = spot_size  # Radius
    axiparabola = Axiparabola(f0, delta, R)

    # Apply axiparabola and propagate to the middle of the focal range
    laser.apply_optics(axiparabola)
    laser.propagate(f0 + delta / 2)
    E_middle = abs(laser.grid.get_temporal_field()).max()

    # Compute the equivalent Rayleigh length
    import math

    ZR = math.pi * wavelength * f0**2 / spot_size**2
    # Check that the focal range is much longer than the Rayleigh length
    assert delta > 5 * ZR

    laser.propagate(
        -2 * ZR
    )  # Propagate to two Rayleigh lengths before the middle of the focal range
    E_before = abs(laser.grid.get_temporal_field()).max()
    # For a Gaussian beam, the field amplitude should be reduced
    # by a factor of sqrt(5) after two Rayleigh lengths
    # Here, we check that the field remains significantly higher (1.5 times higher)
    assert E_before > E_middle / np.sqrt(5) * 1.5

    laser.propagate(
        4 * ZR
    )  # Propagate to two Rayleigh lengths after the middle of the focal range
    E_after = abs(laser.grid.get_temporal_field()).max()
    # For a Gaussian beam, the field amplitude should be reduced
    # by a factor of sqrt(5) after two Rayleigh lengths
    # Here, we check that the field remains significantly higher (1.5 times higher)
    assert E_after > E_after / np.sqrt(5) * 1.5
