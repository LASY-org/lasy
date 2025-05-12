from copy import deepcopy

import numpy as np
from scipy.constants import c, epsilon_0


class NonlinearKerrStep:
    r"""
    Class that represents a propagation step with Kerr nonlinearity.

    This allows to calculate spectral broadening or self-focusing due to self phase modulation.

    .. math::

        E (x,y,t) = E(x,y,t) \times \exp(i\,n_2\,k_0\,I(x,y,t))

    where :math:`I(x,y,t)` is the intensity profile of the pulse.

    Parameters
    ----------
    n2 : float
        Nonlinear (intensity dependent) refractive index.
    k0 : float
        Wave vector at the carrier frequency.
    """

    def __init__(self, n2, k0):
        self.update(n2=n2, k0=k0)

    def update(self, n2, k0):
        """
        Update the nonlinear refractive index and/or the wave vector.

        Parameters
        ----------
        n2 : float
            Nonlinear (intensity dependent) refractive index.
        k0 : float
            Wave vector at the carrier frequency.
        """
        self.n2 = n2 if n2 is not None else self.n2
        self.k0 = k0 if k0 is not None else self.k0

    def apply(self, grid_in, distance, grid_out=None, n2=None, k0=None):
        """
        Apply intensity dependent phase shift to the field.

        Parameters
        ----------
        grid : Grid
            Input grid to which the phase shift is applie.
        distance : float
            Distance over which the pulse propagates the field.
        grid_out : Grid, optional
            Grid object on which the laser pulse after applying the phase
            is defined. Can be different from laser grid before applying.
        """
        self.update(n2=n2, k0=k0)

        if grid_out is None:
            grid_out = deepcopy(grid_in)

        temporal_field = grid_in.get_temporal_field()
        intensity = 0.5 * c * epsilon_0 * abs(temporal_field) ** 2

        phase = self.n2 * self.k0 * intensity * distance

        grid_out.set_temporal_field(temporal_field * np.exp(1j * phase))

        return grid_out
