from .propagator import Propagator
from lasy.utils.laser_utils import get_grid_cell_volume

from scipy.constants import c, epsilon_0
import numpy as np


class NonlinearKerrPropagator(Propagator):
    def __init__(self, n2, n0, k0):
        super().__init__()

        self.n2 = n2  # nonlinear refractive index
        self.n0 = n0  # linear refractive index at the carrier frequency
        self.k0 = k0  # wave vector at the carrier frequency

    def update(self):
        print("empty update for NonlinearKerrPropagator")
        return

    def propagate(self, distance, grid_in, grid_out=None):

        temporal_field = grid_in.get_temporal_field()
        intensity = 0.5 * c * epsilon_0 * abs(temporal_field) ** 2 

        phase = self.n0 * self.n2 * self.k0 * intensity * distance
        
        grid_in.set_temporal_field(temporal_field * np.exp(1j * phase))

        return grid_in