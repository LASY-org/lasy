from .laser import laser

class GaussianLaser(Laser):
    """
    Derived class for an analytic profile of a Gaussian laser pulse.
    """

    def __init__(dim, lo, hi, wavelength, pol, emax, tau, w0):
        """
        Gaussian laser constructor
        """
        # TODO
