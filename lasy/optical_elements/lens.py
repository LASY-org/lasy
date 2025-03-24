import numpy as np
from scipy.constants import c

from .optical_element import OpticalElement


class Lens(OpticalElement):
    r"""
    Class for a Lens, derived from ParabolicMirror class, but with some changes in the focal length definition.
    Parameters
    ----------
    R1: ROC first surface (>0 if convex)
    R2: ROC second surface (>0 if concave)
    d: thickness of the thin lens
    n: refractive index
        "SF11" if material is the sf11
        "FS" if the material is fused silica
    """
    def __init__(self, R1, R2, d, n):
        self.R1 = R1
        self.R2 = R2
        self.d = d
        self.n = n

    def amplitude_multiplier(self, x, y, omega):
        """
        Return the amplitude multiplier.
        Parameters
        ----------
        x, y, omega: ndarrays of floats
            Define points on which to evaluate the multiplier.
            These arrays need to all have the same shape.
        Returns
        -------
        multiplier: ndarray of complex numbers
            Contains the value of the multiplier at the specified points
            This array has the same shape as the arrays x, y, omega
            In "rt" x goes to r.max() while y is zero
        """
        
        lam = 2 * np.pi * c / omega *1e6

        # right now the expressions for the refractive index of SF11 and FS are copied and pasted manually from the website "https://refractiveindex.info"
        
        nSF11 = (1+1.73759695/(1-0.013188707/lam**2)+0.313747346/(1-0.0623068142/lam**2)+1.89878101/(1-155.23629/lam**2))**.5
        nFS = (1+0.6961663/(1-(0.0684043/lam)**2)+0.4079426/(1-(0.1162414/lam)**2)+0.8974794/(1-(9.896161/lam)**2))**.5
        
        if self.n == "SF11":
            n=nSF11

        elif self.n == "FS":
            n=nFS
        
        f = 1/((n-1)*( 1/self.R1 - 1/self.R2 + (n-1)*self.d/(n * self.R1 * self.R2)))

        return np.exp(-1j * omega * (x**2 + y**2) / (2 * c * f))