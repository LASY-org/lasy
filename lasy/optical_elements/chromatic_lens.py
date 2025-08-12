import numpy as np
from scipy.constants import c

from .optical_element import OpticalElement


class ChromaticLens(OpticalElement):
    r"""
    Class for a chromatic thin lens, with a varying refractive index depending on the wavelength.

    Examples
    --------
    >>> R1 = 114.5e-3  # 1st ROC
    >>> t1 = 3.4e-3  # lens thickness
    >>> R2 = -114.5e-3  # 2nd ROC
    >>> nFS = (
    ...     lambda x: (
    ...         1
    ...         + 0.6961663 / (1 - (0.0684043 / x) ** 2)
    ...         + 0.4079426 / (1 - (0.1162414 / x) ** 2)
    ...         + 0.8974794 / (1 - (9.896161 / x) ** 2)
    ...     )
    ...     ** 0.5
    ... )
    >>> laser.apply_optics(Lens2(R1=R1, R2=R2, d=t1, n_func=nFS))

    Parameters
    ----------
    R1 : float
        ROC of the first surface (>0 if convex)
    R2 : float
        ROC of the second surface (>0 if concave)
    d : float
        Thickness of the lens used to calculate the total phase shift.
        Note that this optical element still assumes a thin optics.
    n_func : function
        Function that returns the refractive index given the wavelength in microns, taken from the website "https://refractiveindex.info".
        e.g. for Fused Silica:
        nFS = lambda x: (1+0.6961663/(1-(0.0684043/x)**2)+0.4079426/(1-(0.1162414/x)**2)+0.8974794/(1-(9.896161/x)**2))**.5
    """

    def __init__(self, R1, R2, d, n_func):
        self.R1 = R1
        self.R2 = R2
        self.d = d
        self.n_func = n_func

    def amplitude_multiplier(self, x, y, omega):
        """
        Return the amplitude multiplier.

        Parameters
        ----------
        x, y, omega : ndarrays of floats
            Define points on which to evaluate the multiplier. Must have the same shape.

        Returns
        -------
        multiplier : ndarray of complex numbers
            Contains the value of the multiplier at the specified points
        """
        lam = 2 * np.pi * c / omega * 1e6  # Wavelength in microns
        n = self.n_func(lam)

        f = 1 / (
            (n - 1)
            * (1 / self.R1 - 1 / self.R2 + (n - 1) * self.d / (n * self.R1 * self.R2))
        )

        return np.exp(-1j * omega * (x**2 + y**2) / (2 * c * f))
