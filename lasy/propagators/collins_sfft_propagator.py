from .single_fft_propagator import SingleFFTPropagator


class CollinsSFFTPropagator(SingleFFTPropagator):
    def __init__(self):
        super().__init__()
        self.update()

        return

    def update(self):
        return

    def propagate(self, distance, grid_in, grid_out=None, abcd=None):
        self.update()

        return deepcopy(grid_in)
