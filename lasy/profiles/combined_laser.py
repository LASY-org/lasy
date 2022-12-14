from .profile import Profile

class CombinedLongitudinalTransverseLaser(Profile):
    """
    Class that combines a longitudinal and transverse laser profile
    """

    def __init__(self, wavelength, pol, laser_energy,
                       long_profile, trans_profile):
        """
        TODO
        """
        super().__init__(wavelength, pol)

        self.long_profile = long_profile
        self.trans_profile = trans_profile

    def evaluate(self, x, y, t):
        """
        TODO
        """
        envelope = self.trans_profile.evaluate(x, y) * \
                    self.long_profile.evaluate(t)
        return envelope
