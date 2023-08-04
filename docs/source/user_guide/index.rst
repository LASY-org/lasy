User Guide
==========

Installation
############

To install the code you will need to first clone the repository to your local machine.
Change into the new directory and then run the install command as given below.

..  code-block:: bash
    :caption: Installation Instructions

    git clone https://github.com/LASY-org/lasy.git
    cd lasy
    python3 -m pip install -v .

More installation options and further instructions will be added in due course.


First Example
#############

We will try a simple example to get familiar with the code structure and to verify the installation was successful.
Let's generate a Gaussian pulse at focus, propagate it backwards by one Rayleigh length (the pulse is then located ahead of the focal plane) and then output it to a file.

..  code-block:: python
   :caption: First lets load in the required functions from the library.

   from lasy.profiles.gaussian_profile import GaussianProfile
   from lasy.laser import Laser


..  code-block:: python
   :caption: Next, define the physical parameters of the laser pulse and create the laser profile object.

   wavelength     = 800e-9  # Laser wavelength in meters
   polarization   = (1,0)   # Linearly polarized in the x direction
   energy         = 1.5     # Energy of the laser pulse in joules
   spot_size      = 25e-6   # Waist of the laser pulse in meters
   pulse_duration = 30e-15  # Pulse duration of the laser in seconds
   t_peak         = 0.0     # Location of the peak of the laser pulse in time

   laser_profile = GaussianProfile(wavelength,polarization,energy,spot_size,pulse_duration,t_peak)

..  code-block:: python
   :caption: Now create a full laser object containing the above physical parameters together with the computational settings.

   dimensions     = 'rt'                              # Use cylindrical geometry
   lo             = (0,-2.5*pulse_duration)           # Lower bounds of the simulation box
   hi             = (5*spot_size,2.5*pulse_duration)  # Upper bounds of the simulation box
   num_points     = (300,500)                         # Number of points in each dimension

   laser = Laser(dimensions,lo,hi,num_points,laser_profile)

..  code-block:: python
   :caption: By default, the values of the laser envelope are injected on the focal plan. One can propagate it backwards by one Rayleigh length (optional).

   z_R            = 3.14159*spot_size**2/wavelength    # The Rayleigh length
   laser.propagate(-z_R)                               # Propagate the pulse ahead of the focal plane

..  code-block:: python
   :caption: Output the result to a file. Here we utilise the openPMD standard.

   file_prefix    = 'test_output' # The file name will start with this prefix
   file_format    = 'h5'          # Format to be used for the output file

   laser.write_to_file(file_prefix, file_format)

The generated file may now be viewed or used as a laser input to a variety of other simulation tools.
