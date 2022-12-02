import numpy as np
from lasy.laser import Laser

lo = (1,2,3)
hi = (4,6,8)
dim = 'xyt'
npoints=(2,3,2)

array_in = np.zeros(npoints)
wavelength=.8e-6
pol = (1,0)
laser = Laser(dim, lo, hi, array_in, wavelength, pol)
laser.write_to_file()
laser.propagate(1)
laser.write_to_file()
