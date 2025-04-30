import numpy as np
from scipy.constants import c, epsilon_0


class NonlinearKerrStep:
    r"""
    Class that represents a propagation step with Kerr nonlinearity.
    This allows to calculate spectral broadening or self-focusing due to self phase modulation.

    .. math::

        E (x,y,t) = E(x,y,t) \times \exp(i\,n_2\,n_0\,k_0\,I(x,y,t))

    where :math:`I(x,y,t)` is the intensity profile of the pulse.

    Parameters
    ----------
    n2 : float
        Nonlinear refractive index.
    n0 : float
        Linear refractive index at the carrier frequency.
    k0 : float
        Wave vector at the carrier frequency.
    """

    def __init__(self, n2, n0, k0):
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
