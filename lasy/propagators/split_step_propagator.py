from .propagator import Propagator


class SplitStepPropagator(Propagator):
    def __init__(self):
        super().__init__()
        print("empty init for SplitStepPropagator")

    def update(self):
        print("empty update for SplitStepPropagator")
        pass

    def propagate(self, distance, grid_in, grid_out=None):
        return grid_in