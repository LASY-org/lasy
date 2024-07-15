from lasy.profiles.axiparabola_profile import AxiparabolaProfile
from lasy.profiles.longitudinal_profiles import GaussianLongitudinalProfile
from lasy.laser import Laser

wavelength = 800e-9  # Laser wavelength in meters
polarization = (1, 0)  # Linearly polarized in the x direction
energy = 1.5  # Energy of the laser pulse in joules
spot_size = 25e-6  # Waist of the laser pulse in meters
pulse_duration = 30e-15  # Pulse duration of the laser in seconds
t_peak = 0.0  # Location of the peak of the laser pulse in time

laser_profile = AxiparabolaProfile(
    wavelength,
    polarization,
    energy,
    axiparabola_radius,
    focal_distance,
    focal_range,
    temporal_profile,
)

dimensions = "rt"  # Use cylindrical geometry
lo = (0, -2.5 * pulse_duration)  # Lower bounds of the simulation box
hi = (5 * spot_size, 2.5 * pulse_duration)  # Upper bounds of the simulation box
num_points = (300, 500)  # Number of points in each dimension

laser = Laser(dimensions, lo, hi, num_points, laser_profile)

z_R = 3.14159 * spot_size**2 / wavelength  # The Rayleigh length
laser.propagate(-z_R)  # Propagate the pulse upstream of the focal plane

file_prefix = "test_output"  # The file name will start with this prefix
file_format = "h5"  # Format to be used for the output file

laser.write_to_file(file_prefix, file_format)
