import numpy as np
from scipy.constants import c, epsilon_0

from .propagator import Propagator


class NonlinearKerrStep():
    """
    Propagator for a nonlinear step with Kerr nonlinearity.
    """

    def __init__(self, n2, n0, k0):
        """
        Initialize the NonlinearKerrStep.

        Parameters
        ----------
        n2 : float
            Nonlinear refractive index.
        n0 : float
            Linear refractive index at the carrier frequency.
        k0 : float
            Wave vector at the carrier frequency.
        """

        self.n2 = n2
        self.n0 = n0
        self.k0 = k0

    def propagate(self, grid, distance):
        """
        Propagate the input grid for nonlinear step.

        Parameters
        ----------
        grid : Grid
            Input grid to be propagated.
        distance : float
            Distance over withh to propagate the field.
        """
        temporal_field = grid.get_temporal_field()
        intensity = 0.5 * c * epsilon_0 * abs(temporal_field) ** 2

        phase = self.n0 * self.n2 * self.k0 * intensity * distance

        grid.set_temporal_field(temporal_field * np.exp(1j * phase))
