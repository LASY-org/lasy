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
        Nonlinear (intensity dependent) refractive index.
    n0 : float
        Linear refractive index at the carrier frequency.
    k0 : float
        Wave vector at the carrier frequency.
    """

    def __init__(self, n2, n0, k0):
        self.n2 = n2
        self.n0 = n0
        self.k0 = k0

    def apply(self, grid_in, distance):
        """
        Apply intensity dependent phase shift to the field.

        Parameters
        ----------
        grid : Grid
            Input grid to which the phase shift is applie.
        distance : float
            Distance over which the pulse propagates the field.
        """
        temporal_field = grid_in.get_temporal_field()
        intensity = 0.5 * c * epsilon_0 * abs(temporal_field) ** 2

        phase = self.n0 * self.n2 * self.k0 * intensity * distance

        grid_in.set_temporal_field(temporal_field * np.exp(1j * phase))
