from .laser import laser

class LaguerreGaussianLaser(Laser):
    """
    Derived class for an analytic profile of high-order Gaussian
    laser pulses in rz coordinates.
    """

    def __init__(dim, lo, hi, wavelength, pol, emax, tau, w0):
        """
        Laguerre Gaussian laser constructor

        For this to make sense, it requires an r,theta,z grid
        """
        # TODO
