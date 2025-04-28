from .propagator import Propagator
from scipy.constants import c, epsilon_0

class NonlinearKerrPropagator(Propagator):
    def __init__(self, n2, n0, k0):
        super().__init__()
        self.n2 = n2
        self.n0 = n0
        self.k0 = k0

    def update(self):
        print("empty update for NonlinearKerrPropagator")
        pass

    def propagate(self, distance, grid_in, grid_out=None):
        
        field = grid_in
        intensity = 0.5 * c * epsilon_0 * abs(grid) ** 2
        phase = self.n0 * self.n2 * self.k0 * intensity * distance

        return grid_in * np.exp(1j * phase)

