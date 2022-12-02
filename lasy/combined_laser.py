from .laser import Laser

class CombinedLongitudinalTransverseLaser(Laser):
    """
    Class that combines a longitudinal and transverse laser profile
    """

    def __init__(self, dim, lo, hi, wavelength, pol,
                 laser_energy, long_profile, trans_profile):
        """
        TODO
        """
        super().__init__(dim, lo, hi, array_in, wavelength, pol)

        self.long_profile = long_profile
        self.trans_profile = trans_profile

        self.fields.fields[:,:,:] = trans_profile.fields[:,:,np.newaxis] * \
                                    long_profile.fields[np.newaxis, :]
